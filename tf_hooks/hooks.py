"""Hook implementations."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

from tf_hooks.cleanup import _cleanup_if_needed
from tf_hooks.constants import FORWARD_HOOKS_ATTR, FORWARD_PRE_HOOKS_ATTR

if TYPE_CHECKING:
    import tensorflow as tf

_all_forward_pre_hooks = OrderedDict()
_all_forward_hooks = OrderedDict()


class TFHook(ABC):
    """Base class for hooks."""

    def __init__(self, layer: tf.keras.layers.Layer, fn: Callable):
        """Initialize the hook.

        Args:
            layer (tf.keras.layers.Layer): The layer to apply the hook.
            fn (Callable): The function for the hook to use.
        """
        self.layer = layer
        self.fn = fn
        self.id = f"{uuid.uuid4()}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Use the hook.

        Subclasses should use self.fn in __call__
        """
        pass

    @abstractmethod
    def remove(self) -> None:
        """Remove the hook."""
        pass


class TFForwardPreHook(TFHook):
    """Forward pre-hook.

    This hook executes a provided function before the inputs
    are passed to the layer.

    See `TFHook` for initialization arguments.

    The function `fn` provided on init must have the following signature:

        `hook(layer, args, kwargs) -> tuple or None`

        where the outputted tuple should be
        `(processed_args: tuple, processed_kwargs: dict)` if any.
        If `None`, the original args and kwargs will be passed onto the
        next hook or the layer to use.
    """

    def __call__(self, args: tuple, kwargs: dict) -> tuple:
        """Process the layer's inputs before the layer is called.

        Args:
            args (tuple): The layer's received arguments.
            kwargs (dict): The layer's received keyword arguments.

        Returns:
            tuple: A tuple of the processed (args, kwargs) if any,
            otherwise the original inputs.
        """
        hook_out = self.fn(self.layer, args, kwargs)

        if hook_out is None:
            return args, kwargs

        generic_error_msg = (
            "Forward prehook's function output must be tuple of (tuple, dict)"
        )
        assert isinstance(hook_out, tuple), generic_error_msg
        assert len(hook_out) == 2, generic_error_msg
        assert isinstance(hook_out[0], tuple) and isinstance(
            hook_out[1], dict
        ), generic_error_msg
        assert set(hook_out[1].keys()).issuperset(
            kwargs.keys()
        ), "Forward prehook's modified kwarg must have the same keys as the original"

        return hook_out

    def remove(self) -> None:
        """Override `TFHook.remove`."""
        del getattr(self.layer, FORWARD_PRE_HOOKS_ATTR)[self.id]
        del _all_forward_pre_hooks[self.id]
        _cleanup_if_needed(self.layer)


class TFForwardHook(TFHook):
    """Forward hook.

    This hook executes a provided function after a layer is called.

    The function `fn` provided on init must have the following signature:

        `hook(layer, args, kwargs) -> tuple or None`

        where the outputted tuple should be
        `(processed_args: tuple, processed_kwargs: dict)` if any.
        If `None`, the original args and kwargs will be passed onto the
        next hook or the layer to use.
    """

    def __init__(
        self, layer: tf.keras.layers.Layer, fn: Callable, always_call: bool = False
    ):
        """Initialize the forward hook.

        Args:
            layer (tf.keras.layers.Layer): The layer to apply the hook.
            fn (Callable): The function for the hook to use.
                The function must have the following signature:

                `hook(layer, args, kwargs, outputs) -> Any`

                where the outputted tuple should be the processed outputs, if any.
                If `None`, the original outputs will be passed onto the
                next hook or the layer to use.
            always_call (bool, optional): `True` to execute the hook regardless
                if an exception is raised when calling the layer. Defaults to `False`
                to not execute the hook.
        """
        super().__init__(layer, fn)
        self.always_call = always_call

    def __call__(self, args: tuple, kwargs: dict, output: Any) -> Any:
        """Process the layer's inputs and outputs after the layer is called.

        Args:
            args (tuple): The layer's received arguments.
            kwargs (dict): The layer's received keyword arguments.
            output (Any): The layer's output.

        Returns:
            The processed outputs if any, otherwise the original outputs.
        """
        hook_out = self.fn(self.layer, args, kwargs, output)
        if hook_out is None:
            return output
        return hook_out

    def remove(self) -> None:
        """Override `TFHook.remove."""
        del getattr(self.layer, FORWARD_HOOKS_ATTR)[self.id]
        del _all_forward_hooks[self.id]
        _cleanup_if_needed(self.layer)
