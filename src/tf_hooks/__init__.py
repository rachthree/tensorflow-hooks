from importlib import metadata

from .register import register_forward_hook, register_forward_pre_hook

__version__ = metadata.version("tensorflow-hooks")

__all__ = ["register_forward_hook", "register_forward_pre_hook"]
