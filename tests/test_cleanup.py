"""Test cleanup utility."""
from unittest import mock

from utils import create_single_layer_model

from tf_hooks import register_forward_hook, register_forward_pre_hook
from tf_hooks.constants import FORWARD_HOOKS_ATTR, FORWARD_PRE_HOOKS_ATTR, OLD_CALL_ATTR


class TestCleanup:
    """Test `_cleanup_if_needed` when using `hook.remove()`."""

    def setup_method(self):
        """Set up each test."""
        model = create_single_layer_model()
        self.layer = model.layers[1]
        self.original_call = self.layer.call

        mock_pre_hook_fn1 = mock.MagicMock(return_value=None)
        mock_pre_hook_fn2 = mock.MagicMock(return_value=None)
        mock_hook_fn1 = mock.MagicMock(return_value=None)
        mock_hook_fn2 = mock.MagicMock(return_value=None)

        hooks = {}
        hooks["pre_hook1"] = register_forward_pre_hook(self.layer, mock_pre_hook_fn1)
        hooks["pre_hook2"] = register_forward_pre_hook(self.layer, mock_pre_hook_fn2)
        hooks["hook1"] = register_forward_hook(self.layer, mock_hook_fn1)
        hooks["hook2"] = register_forward_hook(self.layer, mock_hook_fn2)
        self.hooks = hooks

    def test_cleanup_forward_pre_hooks_only(self):
        """Test cleaning up after only the forward pre-hooks are removed."""
        # Remove the first forward pre-hook
        self.hooks["pre_hook1"].remove()
        assert hasattr(self.layer, FORWARD_PRE_HOOKS_ATTR)
        assert hasattr(self.layer, FORWARD_HOOKS_ATTR)
        assert hasattr(self.layer, OLD_CALL_ATTR)
        assert getattr(self.layer, OLD_CALL_ATTR) == self.original_call

        # Remove the remaining forward pre-hook
        self.hooks["pre_hook2"].remove()
        assert not hasattr(self.layer, FORWARD_PRE_HOOKS_ATTR)
        assert hasattr(self.layer, FORWARD_HOOKS_ATTR)
        assert hasattr(self.layer, OLD_CALL_ATTR)
        assert getattr(self.layer, OLD_CALL_ATTR) == self.original_call

    def test_cleanup_forward_hooks_only(self):
        """Test cleaning up after only the forward hooks are removed."""
        # Remove the first forward hook
        self.hooks["hook1"].remove()
        assert hasattr(self.layer, FORWARD_PRE_HOOKS_ATTR)
        assert hasattr(self.layer, FORWARD_HOOKS_ATTR)
        assert hasattr(self.layer, OLD_CALL_ATTR)
        assert getattr(self.layer, OLD_CALL_ATTR) == self.original_call

        # Remove the remaining forward hook
        self.hooks["hook2"].remove()
        assert hasattr(self.layer, FORWARD_PRE_HOOKS_ATTR)
        assert not hasattr(self.layer, FORWARD_HOOKS_ATTR)
        assert hasattr(self.layer, OLD_CALL_ATTR)
        assert getattr(self.layer, OLD_CALL_ATTR) == self.original_call

    def test_cleanup_all_hooks(self):
        """Test cleaning up after all hook types are removed."""
        # Remove all hooks
        for hook in self.hooks.values():
            hook.remove()

        assert not hasattr(self.layer, FORWARD_PRE_HOOKS_ATTR)
        assert not hasattr(self.layer, FORWARD_HOOKS_ATTR)
        assert not hasattr(self.layer, OLD_CALL_ATTR)
        assert self.layer.call == self.original_call
