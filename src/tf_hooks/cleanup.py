"""Hook cleanup utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tf_hooks.constants import FORWARD_HOOKS_ATTR, FORWARD_PRE_HOOKS_ATTR, OLD_CALL_ATTR

if TYPE_CHECKING:
    import tensorflow as tf


def _cleanup_if_needed(layer: tf.keras.layers.Layer) -> None:
    """Cleanup the layer attributes modified by hook registration if needed.

    Args:
        layer (tf.keras.layers.Layer): The TensorFlow layer.
    """
    pre_hook_attr_exists = hasattr(layer, FORWARD_PRE_HOOKS_ATTR)
    if pre_hook_attr_exists and not getattr(layer, FORWARD_PRE_HOOKS_ATTR):
        delattr(layer, FORWARD_PRE_HOOKS_ATTR)
        pre_hook_attr_exists = False

    hook_attr_exists = hasattr(layer, FORWARD_HOOKS_ATTR)
    if hook_attr_exists and not getattr(layer, FORWARD_HOOKS_ATTR):
        delattr(layer, FORWARD_HOOKS_ATTR)
        hook_attr_exists = False

    if not pre_hook_attr_exists and not hook_attr_exists:
        layer.call = getattr(layer, OLD_CALL_ATTR)
        delattr(layer, OLD_CALL_ATTR)
