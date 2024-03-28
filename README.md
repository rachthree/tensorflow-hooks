
# tensorflow-hooks: PyTorch-like hooks for TensorFlow Keras layers
One of PyTorch's many strengths are its `torch.nn.Module` hooks.
Inspired by [this issue](https://github.com/tensorflow/tensorflow/issues/33478),
this utility aims to provide a similar functionality as PyTorch's forward pre-hooks and hooks to TensorFlow Keras layers.

Note: Backward hooks are not supported and are not planned to be supported at this time.

## Prerequisites
[TensorFlow](https://www.tensorflow.org/install) should be installed.
`tensorflow-hooks` is tested with versions 2.14.1 and above.

## Installation
Install via `pip` via
```bash
pip install tensorflow-hooks
```

or clone this repo then use
```
pip install .
```

## Using a Forward Pre-hook
A forward pre-hook callable must be:
```python
fn(layer: tf.keras.layers.Layer, args: tuple, kwargs: dict) -> Union[None, Tuple[tuple, dict]]
```

During the layer's forward method, the hook will execute before the layer's forward pass. The hook either returns `None` as a passthrough (or inputs have been modified in-place) or `tuple, dict` to provide the arguments and keyword arguments that the layer will receive.

`tf-hooks` registers the hook and modifies the layer's `call` method via `tf_hooks.register_forward_pre_hook`.

For example:
```python
import tensorflow as tf
from tf_hooks import register_forward_pre_hook

model = tf.keras.applications.resnet50.ResNet50()

def prehook_fn(layer: tf.keras.layers.Layer, args: tuple, kwargs: dict):
    print(f"{layer.name} args received: {args}")
    print(f"{layer.name} kwargs received: {kwargs}")

prehooks = []
for layer in model.layers:
    prehooks.append(register_forward_pre_hook(layer, prehook_fn))

test_input = tf.random.uniform((4, 224, 224, 3))

test_output = model(test_input)
```

The above would result in printing out all the inputs seen by each layer.
If the received arguments / keyword arguments were modified, or new ones provided, this would affect layer computation.

Each item in the `prehooks` list above is a `tf_hook.hooks.TFForwardPreHook`.
To register a hook, use `prehook.remove()`. For example, to remove all the hooks applied above:
```python
for prehook in prehooks:
    prehook.remove()
```

Notes:
* Multiple pre-hooks can be applied to the same layer via using `register_forward_pre_hook` again, and each pre-hook
will execute in the order it was registered for the layer. To prepend a pre-hook, use `prepend=True` when using `register_forward_pre_hook`.


## Using a Forward Hook

A forward hook callable must be:
```python
fn(layer: tf.keras.layers.Layer, args: tuple, kwargs: dict, outputs: Union[tf.Tensor, tuple]) -> Union[None, tf.Tensor, tuple]
```

After the layer's forward method, the hook will execute, using the layer inputs and outputs. The hook either returns `None` as a passthrough (or outputs have been modified in-place) or
whatever objects the hook function returns. These will be provided to the next layer(s).

`tf-hooks` registers the hook and modifies the layer's `call` method via `tf_hooks.register_forward_hook`.

For example:
```python
import tensorflow as tf
from tf_hooks import register_forward_hook
from typing import Union

model = tf.keras.applications.resnet50.ResNet50()

def hook_fn(layer: tf.keras.layers.Layer, args: tuple, kwargs: dict, outputs: Union[tf.Tensor, tuple]):
    print(f"{layer.name} args received: {args}")
    print(f"{layer.name} kwargs received: {kwargs}")
    print(f"{layer.name} outputs: {outputs}")

hooks = []
for layer in model.layers:
    hooks.append(register_forward_hook(layer, hook_fn))

test_input = tf.random.uniform((4, 224, 224, 3))

test_output = model(test_input)
```

The above would result in printing out all layers' inputs and outputs.
If the received outputs were modified, or new ones provided, this would affect downstream layers.

Each item in the `hooks` list above is a `tf_hook.hooks.TFForwardHook`.
To register a hook, use `hook.remove()`. For example, to remove all the hooks applied above:
```python
for hook in hooks:
    hook.remove()
```

Notes:
* Multiple hooks can be applied to the same layer via using `register_forward_hook` again, and each hook
will execute in the order it was registered for the layer. To prepend a hook, use `prepend=True` when using `register_forward_hook`.
* Should you want the hook to always be called even if an exception occurs, use `always_call=True` when using `register_forward_hook`.
