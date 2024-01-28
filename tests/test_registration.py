"""Test hook registration functionality."""
import types
from typing import Union
from unittest import mock

import pytest
import tensorflow as tf
from utils import create_single_layer_model

from tf_hooks.constants import FORWARD_HOOKS_ATTR, FORWARD_PRE_HOOKS_ATTR, OLD_CALL_ATTR
from tf_hooks.hooks import TFForwardHook, TFForwardPreHook
from tf_hooks.register import (
    _call_with_hooks,
    _switch_calls_if_needed,
    register_forward_hook,
    register_forward_pre_hook,
)


class TestSwitchCalls:
    """Test `_switch_calls_if_needed`."""

    def setup_method(self):
        """Set up each test."""
        self.model = create_single_layer_model()
        self.layer = self.model.layers[1]

    def test_switch(self):
        """Test switching of the original call with `_call_with_hooks`."""
        old_call = self.layer.call
        assert not hasattr(self.layer, OLD_CALL_ATTR)
        _switch_calls_if_needed(self.layer)
        assert hasattr(self.layer, OLD_CALL_ATTR)
        assert getattr(self.layer, OLD_CALL_ATTR) == old_call
        assert self.layer.call == types.MethodType(_call_with_hooks, self.layer)

    def test_skip(self):
        """Test skipping switching calls if `_call_with_hooks` already exists."""
        old_call = self.layer.call
        assert not hasattr(self.layer, OLD_CALL_ATTR)
        _switch_calls_if_needed(self.layer)
        # Run it again
        _switch_calls_if_needed(self.layer)
        assert hasattr(self.layer, OLD_CALL_ATTR)
        assert getattr(self.layer, OLD_CALL_ATTR) == old_call
        assert self.layer.call == types.MethodType(_call_with_hooks, self.layer)


def _check_original_layer(layer: tf.keras.layers.Layer):
    """Check original layer for clean state."""
    assert not hasattr(layer, FORWARD_PRE_HOOKS_ATTR)
    assert not hasattr(layer, FORWARD_HOOKS_ATTR)
    assert not hasattr(layer, OLD_CALL_ATTR)


def _check_layer_hook(
    layer: tf.keras.layers.Layer,
    hook_attr_name: str,
    hook: Union[TFForwardHook, TFForwardPreHook],
):
    """Check layer that has hooks registered."""
    assert hasattr(layer, hook_attr_name)
    assert hook.id in getattr(layer, hook_attr_name)
    assert getattr(layer, hook_attr_name)[hook.id] == hook


class TestRegisterHooks:
    """Test hook registration functions.

    This also inherently tests `_call_with_hooks`.
    """

    def setup_method(self):
        """Set up each test."""
        self.model = create_single_layer_model()
        self.layer = self.model.layers[1]

    def test_register_forward_pre_hook(self):
        """Test registering a forward prehook."""
        mock_manager = mock.Mock()
        mock_fn = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        # Check new attribute created and has the hook
        hook = register_forward_pre_hook(self.layer, mock_fn)
        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, hook)

        # Check hook was called before the original call
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_fn, "mock_fn")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_fn(self.layer, (test_input,), {}),
                mock.call.mock_call(test_input),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_register_multiple_forward_pre_hook(self):
        """Test registering more than one forward prehook."""
        mock_manager = mock.Mock()
        mock_fn1 = mock.MagicMock(return_value=None)
        mock_fn2 = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        hook1 = register_forward_pre_hook(self.layer, mock_fn1)
        hook2 = register_forward_pre_hook(self.layer, mock_fn2)

        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, hook1)
        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, hook2)

        # Test order of calls
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_fn1, "mock_fn1")
            mock_manager.attach_mock(mock_fn2, "mock_fn2")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_fn1(self.layer, (test_input,), {}),
                mock.call.mock_fn2(self.layer, (test_input,), {}),
                mock.call.mock_call(test_input),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_register_prepend_forward_pre_hook(self):
        """Test prepending a forward pre-hook to a layer that already has one."""
        mock_manager = mock.Mock()
        mock_fn1 = mock.MagicMock(return_value=None)
        mock_fn2 = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        hook1 = register_forward_pre_hook(self.layer, mock_fn1)
        hook2 = register_forward_pre_hook(self.layer, mock_fn2, prepend=True)

        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, hook1)
        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, hook2)

        # Test order of calls
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_fn1, "mock_fn1")
            mock_manager.attach_mock(mock_fn2, "mock_fn2")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_fn2(self.layer, (test_input,), {}),
                mock.call.mock_fn1(self.layer, (test_input,), {}),
                mock.call.mock_call(test_input),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_register_forward_hook(self):
        """Test registering a forward hook."""
        mock_manager = mock.Mock()
        mock_fn = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        hook = register_forward_hook(self.layer, mock_fn)
        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook)

        # Test order of calls
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_fn, "mock_fn")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_call(test_input),
                mock.call.mock_fn(self.layer, (test_input,), {}, test_output),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_register_multiple_forward_hooks(self):
        """Test registering more than one forward hook."""
        mock_manager = mock.Mock()
        mock_fn1 = mock.MagicMock(return_value=None)
        mock_fn2 = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        hook1 = register_forward_hook(self.layer, mock_fn1)
        hook2 = register_forward_hook(self.layer, mock_fn2)

        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook1)
        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook2)

        # Test order of calls
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_fn1, "mock_fn1")
            mock_manager.attach_mock(mock_fn2, "mock_fn2")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_call(test_input),
                mock.call.mock_fn1(self.layer, (test_input,), {}, test_output),
                mock.call.mock_fn2(self.layer, (test_input,), {}, test_output),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_register_prepend_forward_hooks(self):
        """Test prepending a forward hook to a layer that already has one."""
        mock_manager = mock.Mock()
        mock_fn1 = mock.MagicMock(return_value=None)
        mock_fn2 = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        hook1 = register_forward_hook(self.layer, mock_fn1)
        hook2 = register_forward_hook(self.layer, mock_fn2, prepend=True)

        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook1)
        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook2)

        # Test order of calls
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_fn1, "mock_fn1")
            mock_manager.attach_mock(mock_fn2, "mock_fn2")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_call(test_input),
                mock.call.mock_fn2(self.layer, (test_input,), {}, test_output),
                mock.call.mock_fn1(self.layer, (test_input,), {}, test_output),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_register_both_forward_types(self):
        """Test registering both a forward pre-hook and a forward hook."""
        mock_manager = mock.Mock()
        mock_pre_hook_fn = mock.MagicMock(return_value=None)
        mock_hook_fn = mock.MagicMock(return_value=None)
        _check_original_layer(self.layer)

        test_input = tf.random.uniform((4, 128, 128, 3))
        expected_output = self.model(test_input)

        # Check new attributes created and have both kinds of hooks
        prehook = register_forward_pre_hook(self.layer, mock_pre_hook_fn)
        hook = register_forward_hook(self.layer, mock_hook_fn)
        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, prehook)
        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook)

        # Test order of calls
        with mock.patch.object(
            self.layer, OLD_CALL_ATTR, wraps=getattr(self.layer, OLD_CALL_ATTR)
        ) as mock_call:
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_pre_hook_fn, "mock_pre_hook_fn")
            mock_manager.attach_mock(mock_hook_fn, "mock_hook_fn")

            # Run forward inference with hooks
            test_output = self.model(test_input)

            expected_calls = [
                mock.call.mock_pre_hook_fn(self.layer, (test_input,), {}),
                mock.call.mock_call(test_input),
                mock.call.mock_hook_fn(self.layer, (test_input,), {}, test_output),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

        tf.debugging.assert_equal(test_output, expected_output)

    def test_always_call(self):
        """Test forward hook that is always called if an exception occurs."""
        mock_manager = mock.Mock()
        mock_pre_hook_fn = mock.MagicMock(return_value=None)
        mock_hook_fn1 = mock.MagicMock(return_value=None)
        mock_hook_fn2 = mock.MagicMock(return_value=None)

        # This will be used to patch layer._old_call to raise an exception.
        mock_call = mock.Mock(side_effect=ValueError("test error"))
        _check_original_layer(self.layer)

        prehook = register_forward_pre_hook(self.layer, mock_pre_hook_fn)
        hook1 = register_forward_hook(self.layer, mock_hook_fn1, always_call=True)
        hook2 = register_forward_hook(self.layer, mock_hook_fn2)

        _check_layer_hook(self.layer, FORWARD_PRE_HOOKS_ATTR, prehook)
        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook1)
        _check_layer_hook(self.layer, FORWARD_HOOKS_ATTR, hook2)

        # Use good input to pass TF's shape checks
        test_input = tf.random.uniform((4, 128, 128, 3))

        with mock.patch.object(self.layer, OLD_CALL_ATTR, mock_call):
            mock_manager.attach_mock(mock_call, "mock_call")
            mock_manager.attach_mock(mock_pre_hook_fn, "mock_pre_hook_fn")
            mock_manager.attach_mock(mock_hook_fn1, "mock_hook_fn1")
            mock_manager.attach_mock(mock_hook_fn2, "mock_hook_fn2")

            # Run forward inference with hooks
            with pytest.raises(ValueError):
                self.model(test_input)

            expected_calls = [
                mock.call.mock_pre_hook_fn(self.layer, (test_input,), {}),
                mock.call.mock_call(test_input),
                mock.call.mock_hook_fn1(self.layer, (test_input,), {}, None),
            ]
            mock_manager.assert_has_calls(expected_calls, any_order=False)

            # hook2 is not set to be always called,
            # so mock_hook_fn2 should not be called.
            mock_hook_fn2.assert_not_called()
