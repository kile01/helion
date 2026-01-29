"""Shared utilities for operation detection in Dynamo/Inductor integration.

This module consolidates operation detection logic used by both:
- helion._compiler._dynamo.variables (Dynamo level)
- helion._compiler.inductor_lowering_extra (Inductor level)
"""

from __future__ import annotations

import torch

# Cache for method name to aten OpOverload resolution
_METHOD_TO_ATEN_VIEW: dict[str, bool | None] = {}

# View ops that can't be auto-detected via PyTorch's is_view:
# - Not in aten namespace (Python builtins/methods)
# - Have is_view=False but may return views in practice
# - Dynamo-specific internal ops
MANUAL_VIEW_OPS = frozenset({
    # Python indexing/attribute access (not in aten)
    "getitem",
    "getattr",
    # Dynamo internal for .data property access
    "_get_data_attr",
    # Dtype conversion methods that return self when dtype matches
    # (not in aten, lowered to aten.to or aten._to_copy)
    "half",
    "float",
    "double",
    "int",
    "long",
    "short",
    "bool",
    "bfloat16",
    # .type() method returns self when type matches
    "type",
    # atleast_*d may return view when dimensions already sufficient
    # (marked is_view=False in aten because they may also copy)
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    # _unsafe_view is semantically a view but lacks aliasing annotation
    "_unsafe_view",
})


def get_op_name(target: object) -> str:
    """Extract the base operation name from a target.

    Works with OpOverload, functions, and string method names.
    """
    if hasattr(target, "__name__"):
        return target.__name__.split(".")[0]
    elif hasattr(target, "name"):
        # Handle torch._ops.OpOverload like 'aten::permute' -> 'permute'
        return target.name().split("::")[-1].split(".")[0]
    elif isinstance(target, str):
        return target
    return ""


def check_aten_is_view(op_name: str) -> bool:
    """Check if an op name corresponds to a view op in aten namespace.

    Caches results for performance.
    """
    if op_name in _METHOD_TO_ATEN_VIEW:
        result = _METHOD_TO_ATEN_VIEW[op_name]
        return result if result is not None else False

    # Try to find in aten namespace
    result = None
    try:
        aten_op = getattr(torch.ops.aten, op_name, None)
        if aten_op is not None:
            # Check default overload first
            if hasattr(aten_op, "default"):
                default = aten_op.default
                if isinstance(default, torch._ops.OpOverload):
                    result = default.is_view
            # If no default, check any overload for is_view=True
            if result is None:
                for overload_name in getattr(aten_op, "overloads", lambda: [])():
                    overload = getattr(aten_op, overload_name, None)
                    if isinstance(overload, torch._ops.OpOverload) and overload.is_view:
                        result = True
                        break
    except Exception:
        pass

    _METHOD_TO_ATEN_VIEW[op_name] = result
    return result if result is not None else False


def is_view_op(target: object, method_name: str | None = None) -> bool:
    """Check if an FX node target represents a view operation.

    Uses PyTorch's schema-based is_view detection when available,
    falling back to MANUAL_VIEW_OPS for ops that can't be auto-detected.

    Args:
        target: The FX node target (OpOverload, function, or string method name)
        method_name: For call_method nodes, the method name string

    Returns:
        True if the operation is a view (shares storage without copying)
    """
    # For OpOverload targets, use PyTorch's is_view property directly
    if isinstance(target, torch._ops.OpOverload):
        # Trust PyTorch's schema-based detection
        if target.is_view:
            return True
        # Check manual overrides for ops with is_view=False that are actually views
        op_name = get_op_name(target)
        return op_name in MANUAL_VIEW_OPS

    # For call_method with string target
    if method_name is not None:
        # Check manual list first
        if method_name in MANUAL_VIEW_OPS:
            return True
        # Try to resolve to aten OpOverload and check is_view
        return check_aten_is_view(method_name)

    # For call_function with function target
    if hasattr(target, "__name__"):
        op_name = target.__name__
        # Check manual list first
        if op_name in MANUAL_VIEW_OPS:
            return True
        # Try to resolve to aten OpOverload and check is_view
        return check_aten_is_view(op_name)

    return False


def is_clone_node(fx_node: torch.fx.Node) -> bool:
    """Check if an FX node represents a clone operation.

    Used at Dynamo level for clone detection before AOT autograd
    eliminates clones.
    """
    if fx_node.op == "call_method" and fx_node.target == "clone":
        return True
    if fx_node.op == "call_function":
        target = fx_node.target
        # Check for aten.clone or similar
        if hasattr(target, "__name__") and target.__name__ == "clone":
            return True
        if hasattr(target, "name") and "clone" in str(target.name()):
            return True
    return False
