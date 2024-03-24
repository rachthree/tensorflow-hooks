"""Test scenarios."""
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf
from utils import load_io_tensors, save_io_tensors, TFGraph

from tf_hooks import register_forward_hook, register_forward_pre_hook


def _apply_forward_hooks(model: tf.keras.Model, hook_fn: Callable):
    hooks = []
    for layer in model.layers:
        hooks.append(register_forward_hook(layer, hook_fn))
    return hooks


class _SaveIOHook:
    """Test forward hook to save layer inputs and outputs."""

    def __init__(self, save_dir: str):
        """Initialize a _SaveIOHook object.

        Args:
            save_dir (str): Path to save layer inputs and outputs.
        """
        self.save_dir = save_dir

    def __call__(
        self, layer: tf.keras.layers.Layer, args: tuple, kwargs: dict, outputs: Any
    ):
        """Process the layer inputs and outputs.

        Function signature matches what is needed by a forward hook.

        Args:
            layer (tf.keras.layers.Layer): The model layer.
            args (tuple): The arguments seen by the layer.
            kwargs (dict): The keyword arguments seen by the layer.
            outputs (Any): The layer outputs.
        """
        input_path = Path(self.save_dir, f"{layer.name}_inputs.safetensors")
        output_path = Path(self.save_dir, f"{layer.name}_outputs.safetensors")
        input_tensors = {}
        output_tensors = {}
        # Workaround for Add layers
        # Being general is possible but not needed at this time.
        if isinstance(layer, tf.keras.layers.Add):
            args = args[0]

        for i, arg in enumerate(args):
            input_tensors[str(i)] = arg

        if isinstance(outputs, tf.Tensor):
            outputs = [outputs]

        for i, output in enumerate(outputs):
            output_tensors[str(i)] = output

        save_io_tensors(input_tensors, input_path)
        save_io_tensors(output_tensors, output_path)

        return


class TestScenarios:
    """Test different basic scenarios that uses hooks."""

    def setup_method(self):
        """Set up each test."""
        self.model = tf.keras.applications.resnet50.ResNet50()
        self.forward_hooks = []
        self.pre_forward_hooks = []

    def test_save_io(self, tmp_path):
        """Test saving inputs and outputs via hooks."""
        test_input = tf.random.uniform((4, 224, 224, 3))
        expected_output = self.model(test_input)

        forward_hooks = _apply_forward_hooks(self.model, _SaveIOHook(tmp_path))

        # Run inference
        test_output = self.model(test_input)
        tf.debugging.assert_equal(test_output, expected_output)

        # Remove hooks before checking
        for hook in forward_hooks:
            hook.remove()

        # Check inputs and outputs against smaller models
        graph = TFGraph(self.model)

        # tf.keras.layers.InputLayers are not actually called
        layers_to_check = [
            layer
            for layer in self.model.layers
            if layer.name not in graph.input_layer_names
        ]
        assert len(layers_to_check) > 0, "No layers found to check"

        for layer in layers_to_check:
            layer_name = layer.name
            outputs = [layer.output]

            # Get parent layer's outputs as part of the model output
            for parent_node in graph[layer_name].inputs:
                parent_layer = self.model.get_layer(parent_node.name)
                outputs.append(parent_layer.output)

            test_model = tf.keras.Model(self.model.inputs, outputs=outputs)
            test_outputs = test_model(test_input)

            saved_inputs = load_io_tensors(
                Path(f"{tmp_path}", f"{layer.name}_inputs.safetensors")
            )
            saved_outputs = load_io_tensors(
                Path(f"{tmp_path}", f"{layer.name}_outputs.safetensors")
            )

            # Model outputs is
            # [layer_output, parent1_output (input1), parent2_output (input2), etc.]
            # Test layer input
            test_layer_inputs = test_outputs[1:]
            for i, (_, v) in enumerate(saved_inputs.items()):
                tf.debugging.assert_equal(test_layer_inputs[i], v)

            # Test layer output
            test_layer_output = test_outputs[0]
            if isinstance(test_layer_output, tf.Tensor):
                test_layer_output = [test_layer_output]

            for i, (_, v) in enumerate(saved_outputs.items()):
                tf.debugging.assert_equal(test_layer_output[i], v)

    def test_modify_inputs(self):
        """Test modifying inputs via hooks."""
        # Modify inputs for the first non-input layer.
        test_input = tf.random.uniform((4, 224, 224, 3))
        test_modified_input = tf.random.uniform((4, 224, 224, 3))

        def first_layer_prehook(layer, args, kwargs):
            return (test_modified_input,), {}

        test_output = self.model(test_input)
        expected_modified_output = self.model(test_modified_input)
        tf.debugging.assert_none_equal(test_output, expected_modified_output)

        hook = register_forward_pre_hook(self.model.layers[1], first_layer_prehook)

        test_modified_output = self.model(test_input)

        tf.debugging.assert_equal(test_modified_output, expected_modified_output)

        hook.remove()

    def test_modify_outputs(self):
        """Test modifying outputs via hooks."""
        # Modify outputs for the final layer to zero.
        # Test that output is zero.
        test_input = tf.random.uniform((4, 224, 224, 3))
        expected_modified_output = tf.zeros((4, 1000))

        def final_layer_hook(layer, args, kwargs, output):
            return expected_modified_output

        test_output = self.model(test_input)
        tf.debugging.assert_none_equal(test_output, expected_modified_output)

        hook = register_forward_hook(self.model.layers[-1], final_layer_hook)

        test_modified_output = self.model(test_input)
        tf.debugging.assert_equal(test_modified_output, expected_modified_output)

        hook.remove()
