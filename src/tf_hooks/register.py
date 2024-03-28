"""Hook registration functions."""

from __future__ import annotations

import types
from typing import Any, Callable, TYPE_CHECKING

from tf_hooks.constants import FORWARD_HOOKS_ATTR, FORWARD_PRE_HOOKS_ATTR, OLD_CALL_ATTR
from tf_hooks.hooks import (
    _all_forward_hooks,
    _all_forward_pre_hooks,
    TFForwardHook,
    TFForwardPreHook,
)

if TYPE_CHECKING:
    import tensorflow as tf


def _call_with_hooks(layer: tf.keras.layers.Layer, *args, **kwargs) -> Any:
    """Call a layer with execution of its hooks.

    This layer's `call` method will be replaced by this during hook registration.

    Should this replace `layer.call`, the layer then will:
    1. Execute hooks on inputs (forward pre-hook)
    2. Pass hook-processed or original inputs through the original `layer.call`
    3. Execute hooks on inputs and outputs (forward hook)
    4. Pass hook-processed or original outputs onto the next layer.

    Args:
        layer (tf.keras.layers.Layer): The layer.

    Returns:
        The layer's output.
    """
    forward_hooks_called_ids = set()
    if hasattr(layer, FORWARD_PRE_HOOKS_ATTR):
        for hook in getattr(layer, FORWARD_PRE_HOOKS_ATTR).values():
            args, kwargs = hook(args, kwargs)

    # In case layer runs into a error, and an always-called forward hook is used.
    # Set output as None to start with so that it is defined for the forward hook.
    output = None
    try:
        output = getattr(layer, OLD_CALL_ATTR)(*args, **kwargs)

        if hasattr(layer, FORWARD_HOOKS_ATTR):
            for hook in getattr(layer, FORWARD_HOOKS_ATTR).values():
                output = hook(args, kwargs, output)
                forward_hooks_called_ids.add(hook.id)

        return output

    except Exception:
        if hasattr(layer, FORWARD_HOOKS_ATTR):
            for hook in getattr(layer, FORWARD_HOOKS_ATTR).values():
                if hook.always_call and hook.id not in forward_hooks_called_ids:
                    output = hook(args, kwargs, output)

        # Raise original exception
        raise


def _switch_calls_if_needed(layer: tf.keras.layers.Layer) -> None:
    """Replace `layer.call` with `_call_with_hooks` if needed."""
    if not hasattr(layer, OLD_CALL_ATTR):
        setattr(layer, OLD_CALL_ATTR, layer.call)
        layer.call = types.MethodType(_call_with_hooks, layer)


def register_forward_pre_hook(
    layer: tf.keras.layers.Layer, fn: Callable, prepend: bool = False
) -> TFForwardPreHook:
    """Register a forward pre-hook to a layer.

    The hook executes before a layer is called.

    Args:
        layer (tf.keras.layers.Layer): The layer
        fn (Callable): The function for the hook to use.
            This must have the following signature:

            ```
            prehook(
                layer: tf.keras.layers.Layer,
                args: tuple,
                kwargs: dict,
            ) -> Union[None, Tuple[tuple, dict]]
            ```

            where the outputted tuple should be
            `(processed_args: tuple, processed_kwargs: dict)` if any.
            If `None`, the received args and kwargs will be passed onto the
            next hook or the layer to use. Note that if they are modified
            in-place, those changes will take effect.
        prepend (bool, optional): `True` to execute the hook before
            all existing forward pre-hooks on the layer. Defaults to `False`
            to execute after all existing forward pre-hooks on the layer.

    Returns:
        TFForwardPreHook: The forward pre-hook.
    """
    _switch_calls_if_needed(layer)
    hook = TFForwardPreHook(layer, fn)

    if not hasattr(layer, FORWARD_PRE_HOOKS_ATTR):
        setattr(layer, FORWARD_PRE_HOOKS_ATTR, {})

    pre_hooks = getattr(layer, FORWARD_PRE_HOOKS_ATTR)
    if prepend:
        # Normally, OrderedDict.move_to_end should be used, but
        # Later Keras >= 3.0 converts dicts to keras.src.utils.tracking.TrackedDict,
        # which doe not have move_to_end
        # Note: This only works with Python > 3.7 since
        #  that is when dictionaries started maintaining order.
        tmp_pre_hooks = {hook.id: hook}
        tmp_pre_hooks.update(pre_hooks)
        setattr(layer, FORWARD_PRE_HOOKS_ATTR, tmp_pre_hooks)
    else:
        pre_hooks[hook.id] = hook
    _all_forward_pre_hooks[hook.id] = hook
    return hook


def register_forward_hook(
    layer: tf.keras.layers.Layer,
    fn: Callable,
    prepend: bool = False,
    always_call: bool = False,
) -> TFForwardHook:
    """Register a forward hook to a layer.

    The hook executes after a layer is called.

    Args:
        layer (tf.keras.layers.Layer): The layer
        fn (Callable): The function for the hook to use.
            This must have the following signature:

            ```
            hook(
                layer: tf.keras.layers.Layer,
                args: tuple,
                kwargs: dict,
                outputs: Union[tf.Tensor, tuple],
            ) -> Union[None, tf.Tensor, tuple]
            ```

            where the outputted tuple should be the processed outputs, if any,
            or the singular output tensor.
            If `None`, the received outputs will be passed onto the
            next hook or layer to use. Note that if they are modified
            in-place, those changes will take effect.
        prepend (bool, optional): `True` to execute the hook before
            all existing forward hooks on the layer. Defaults to `False`
            to execute after all existing forward hooks on the layer.
        always_call (bool, optional): `True` to execute the hook regardless
            if an exception is raised when calling the layer. Defaults to `False`
            to not execute the hook.

    Returns:
        TFForwardPreHook: The forward hook.
    """
    _switch_calls_if_needed(layer)
    hook = TFForwardHook(layer, fn, always_call=always_call)
    if not hasattr(layer, FORWARD_HOOKS_ATTR):
        setattr(layer, FORWARD_HOOKS_ATTR, {})

    hooks = getattr(layer, FORWARD_HOOKS_ATTR)
    if prepend:
        # Normally, OrderedDict.move_to_end should be used, but
        # Later Keras >= 3.0 converts dicts to keras.src.utils.tracking.TrackedDict,
        # which doe not have move_to_end
        # Note: This only works with Python > 3.7 since
        #  that is when dictionaries started maintaining order.
        tmp_hooks = {hook.id: hook}
        tmp_hooks.update(hooks)
        setattr(layer, FORWARD_HOOKS_ATTR, tmp_hooks)
    else:
        hooks[hook.id] = hook
    _all_forward_hooks[hook.id] = hook
    return hook
