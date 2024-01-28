"""Test hook classes."""
from collections import OrderedDict
from unittest import mock

import pytest
import tensorflow as tf
from utils import create_single_layer_model

from tf_hooks.constants import FORWARD_HOOKS_ATTR, FORWARD_PRE_HOOKS_ATTR
from tf_hooks.hooks import (
    _all_forward_hooks,
    _all_forward_pre_hooks,
    TFForwardHook,
    TFForwardPreHook,
)


class TestTFForwardPreHook:
    """Test TFForwardPreHook."""

    def setup_method(self):
        """Set up tests."""
        tf.random.set_seed(42)
        self.layer = tf.keras.layers.Conv2D(4, (3, 3))
        self.model = create_single_layer_model()

    def test_init_single_layer(self):
        """Test initializing hook on an isolated layer."""
        mock_fn = mock.MagicMock(return_value=None)
        prehook = TFForwardPreHook(self.layer, mock_fn)
        assert prehook.layer == self.layer
        assert prehook.fn == mock_fn
        assert isinstance(prehook.id, str)
        assert prehook.id

    def test_model_layer(self):
        """Test initializing hook on a model's layer."""
        mock_fn = mock.MagicMock(return_value=None)
        prehook = TFForwardPreHook(self.layer, mock_fn)
        assert prehook.layer == self.layer
        assert prehook.fn == mock_fn
        assert isinstance(prehook.id, str)
        assert prehook.id

    def test_call_passthrough(self):
        """Test prehook whose function does not return anything."""
        mock_fn = mock.MagicMock(return_value=None)
        prehook = TFForwardPreHook(self.layer, mock_fn)

        test_args = (tf.random.uniform((4, 32, 32, 3)),)
        test_kwargs = {"mock_kwarg": True}
        test_out = prehook(test_args, test_kwargs)

        # Inputs are passed through
        tf.debugging.assert_equal(test_out[0][0], test_args[0])
        assert test_out[1] == test_kwargs

        # Check what mock function was called with
        mock_fn.assert_called_once_with(self.layer, test_args, test_kwargs)

    def test_call_modified_inputs(self):
        """Test prehook whose function does have outputs."""
        mock_modified_args = ("foo", "bar")
        mock_modified_kwargs = {"mock_kwarg": "mock_change"}
        mock_fn = mock.MagicMock(
            return_value=(mock_modified_args, mock_modified_kwargs)
        )

        prehook = TFForwardPreHook(self.layer, mock_fn)

        test_args = (tf.random.uniform((4, 32, 32, 3)),)
        test_kwargs = {"mock_kwarg": True}
        test_out = prehook(test_args, test_kwargs)

        # Inputs have changed
        assert test_out[0] == mock_modified_args
        assert test_out[1] == mock_modified_kwargs

        # Check what mock function was called with
        mock_fn.assert_called_once_with(self.layer, test_args, test_kwargs)

    def test_call_invalid_modified_inputs(self):
        """Test prehook whose function provides invalid outputs."""
        test_args = (tf.random.uniform((4, 32, 32, 3)),)
        test_kwargs = {"mock_kwarg": True}

        # Test invalid function output data type
        mock_fn = mock.MagicMock(return_value="mock_output")
        prehook = TFForwardPreHook(self.layer, mock_fn)
        with pytest.raises(AssertionError):
            prehook(test_args, test_kwargs)

        # Test invalid number of outputs
        mock_fn = mock.MagicMock(
            return_value=("mock_output", "mock_output", "mock_output")
        )
        prehook = TFForwardPreHook(self.layer, mock_fn)
        with pytest.raises(AssertionError):
            prehook(test_args, test_kwargs)

        # Test invalid first item (args) data type
        mock_fn = mock.MagicMock(
            return_value=("mock_output", {"mock_kwarg": "mock_output"})
        )
        prehook = TFForwardPreHook(self.layer, mock_fn)
        with pytest.raises(AssertionError):
            prehook(test_args, test_kwargs)

        # Test invalid second item (kwargs) data type
        mock_fn = mock.MagicMock(return_value=(("mock_output",), "mock_output"))
        prehook = TFForwardPreHook(self.layer, mock_fn)
        with pytest.raises(AssertionError):
            prehook(test_args, test_kwargs)

        # Test invalid keys in returned kwargs
        mock_fn = mock.MagicMock(
            return_value=(("mock_output",), {"mock_output": "mock_output"})
        )
        prehook = TFForwardPreHook(self.layer, mock_fn)
        with pytest.raises(AssertionError):
            prehook(test_args, test_kwargs)

    def test_remove(self):
        """Test prehook removal."""
        layer_prehook_dict = OrderedDict()
        setattr(self.layer, FORWARD_PRE_HOOKS_ATTR, layer_prehook_dict)

        mock_fn = mock.MagicMock()
        prehook = TFForwardPreHook(self.layer, mock_fn)
        prehook_id = prehook.id
        getattr(self.layer, FORWARD_PRE_HOOKS_ATTR)[prehook_id] = prehook
        _all_forward_pre_hooks[prehook_id] = prehook

        mock_cleanup = mock.MagicMock()
        with mock.patch("tf_hooks.hooks._cleanup_if_needed", mock_cleanup):
            prehook.remove()

            # That cleanup actions were done
            mock_cleanup.assert_called_once_with(self.layer)
            assert prehook_id not in getattr(self.layer, FORWARD_PRE_HOOKS_ATTR)
            assert prehook_id not in _all_forward_pre_hooks


class TestTFForwardHook:
    """Test TFForwardHook."""

    def setup_method(self):
        """Set up tests."""
        tf.random.set_seed(42)
        self.layer = tf.keras.layers.Conv2D(4, (3, 3))
        self.model = create_single_layer_model()

    def test_init_single_layer(self):
        """Test initializing hook on an isolated layer."""
        mock_fn = mock.MagicMock(return_value=None)
        hook = TFForwardHook(self.layer, mock_fn)
        assert hook.layer == self.layer
        assert hook.fn == mock_fn
        assert isinstance(hook.id, str)
        assert hook.id
        assert not hook.always_call

        # With always_call
        hook = TFForwardHook(self.layer, mock_fn, always_call=True)
        assert hook.layer == self.layer
        assert hook.fn == mock_fn
        assert isinstance(hook.id, str)
        assert hook.id
        assert hook.always_call

    def test_model_layer(self):
        """Test initializing hook on a model's layer."""
        mock_fn = mock.MagicMock(return_value=None)
        hook = TFForwardHook(self.layer, mock_fn)
        assert hook.layer == self.layer
        assert hook.fn == mock_fn
        assert isinstance(hook.id, str)
        assert hook.id
        assert not hook.always_call

        # With always_call
        hook = TFForwardHook(self.layer, mock_fn, always_call=True)
        assert hook.layer == self.layer
        assert hook.fn == mock_fn
        assert isinstance(hook.id, str)
        assert hook.id
        assert hook.always_call

    def test_call_passthrough(self):
        """Test forward hook whose function does not return anything."""
        mock_fn = mock.MagicMock(return_value=None)
        hook = TFForwardHook(self.layer, mock_fn)

        test_args = (tf.random.uniform((4, 32, 32, 3)),)
        test_kwargs = {"mock_kwarg": True}
        test_output = tf.random.uniform((4, 32, 32, 16))
        test_hook_out = hook(test_args, test_kwargs, test_output)

        # Outputs are passed through
        tf.debugging.assert_equal(test_hook_out, test_output)

        # Check what mock function was called with
        mock_fn.assert_called_once_with(self.layer, test_args, test_kwargs, test_output)

    def test_call_modified_inputs(self):
        """Test hook whose function does have outputs."""
        test_output = tf.random.uniform((4, 32, 32, 16))
        mock_output = 2 * (test_output + 1)
        mock_fn = mock.MagicMock(return_value=mock_output)

        hook = TFForwardHook(self.layer, mock_fn)

        test_args = (tf.random.uniform((4, 32, 32, 3)),)
        test_kwargs = {"mock_kwarg": True}
        test_output = tf.random.uniform((4, 32, 32, 16))
        test_hook_out = hook(test_args, test_kwargs, test_output)

        # Outputs have changed
        tf.debugging.assert_none_equal(test_hook_out, test_output)
        tf.debugging.assert_equal(test_hook_out, mock_output)

        # Check what mock function was called with
        mock_fn.assert_called_once_with(self.layer, test_args, test_kwargs, test_output)

    def test_remove(self):
        """Test hook removal."""
        layer_hook_dict = OrderedDict()
        setattr(self.layer, FORWARD_HOOKS_ATTR, layer_hook_dict)

        mock_fn = mock.MagicMock()
        hook = TFForwardHook(self.layer, mock_fn)
        hook_id = hook.id
        getattr(self.layer, FORWARD_HOOKS_ATTR)[hook_id] = hook
        _all_forward_hooks[hook_id] = hook

        mock_cleanup = mock.MagicMock()
        with mock.patch("tf_hooks.hooks._cleanup_if_needed", mock_cleanup):
            hook.remove()

            # That cleanup actions were done
            mock_cleanup.assert_called_once_with(self.layer)
            assert hook_id not in getattr(self.layer, FORWARD_HOOKS_ATTR)
            assert hook_id not in _all_forward_hooks
