"""Test utilities."""
from collections import deque
from importlib import metadata
from typing import Any, Mapping

import tensorflow as tf
from packaging import version
from safetensors.tensorflow import safe_open, save_file


def create_single_layer_model():
    """Create a TF Keras model with a single layer."""
    x = tf.keras.Input((128, 128, 3))
    y = tf.keras.layers.Conv2D(4, (3, 3))(x)
    return tf.keras.Model(x, y)


def save_io_tensors(io: Mapping[str, tf.Tensor], save_path: str):
    """Save tensors to safetensors file.

    Args:
        io (Mapping[str, tf.Tensor]): dictionary of
            key tensor name / index, value tensor.
        save_path (str): File path to save the tensor mapping.
    """
    save_file(io, save_path)
    return


def load_io_tensors(tensor_io_path: str):
    """Load a safetensors file.

    Args:
        tensor_io_path (str): Path to safetensors file.

    Returns:
        Dictionary of key tensor name / index, value tensor.
    """
    tensors = {}
    with safe_open(tensor_io_path, framework="tf", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


class TFLayerNode:
    """Node describing a TF model layer.

    Used for testing purposes and is not the same
    as a TF Keras Node.
    """

    def __init__(self, name: str):
        """Initialize a TFLayerNode.

        Args:
            name (str): The layer name.
        """
        self.name = name
        self.inputs = []
        self.outputs = []

    def add_child(self, child_node: "TFLayerNode"):
        """Add a child (layer output) to the node.

        Args:
            child_node (TFLayerNode): The output TFLayerNode.
        """
        if child_node not in self.outputs:
            self.outputs.append(child_node)

        if self not in child_node.inputs:
            child_node.inputs.append(self)
        return


class TFGraph:
    """Graph to describe a TF Keras model.

    For Keras 3.x, this uses internal functionality that may break.
    See https://github.com/keras-team/keras/issues/19278
    """

    def __init__(self, model: tf.keras.Model):
        """Initialize a TFGraph.

        Args:
            model (tf.keras.Model): The model.
        """
        self._graph = {}
        self.input_layer_names = []
        self.legacy_keras = version.parse(
            metadata.version("tensorflow")
        ) < version.parse("2.16.0")
        self.traverse(model)

    def __getitem__(self, layer_name: str):
        """Provide the node.

        Args:
            layer_name (str): The layer name.

        Returns:
            The TFLayerNode for the layer.
        """
        return self._graph[layer_name]

    def _get_input_tensor_node(self, tensor: tf.Tensor):
        """Provide the Keras node of an input tensor.

        Args:
            tensor (tf.keras.Tensor): The input tensor.

        Returns:
            The Keras node. Type changes between Keras 3.x and legacy.
        """
        return (
            tensor.node
            if self.legacy_keras
            else tensor._keras_history[0]._inbound_nodes[0]
        )

    def _get_outbound_nodes(self, layer: tf.keras.Layer):
        """Get the outbound nodes of a layer.

        Args:
            layer (tf.keras.Layer): The layer

        Returns:
            List of outbound nodes.
        """
        return layer.outbound_nodes if self.legacy_keras else layer._outbound_nodes

    def _get_node_layer(self, node: Any):
        """Get the layer of a Keras node.

        Args:
            node (Any): The Keras node.
                Type changes between Keras 3.x and legacy.

        Returns:
            The TF Keras layer.
        """
        return node.layer if self.legacy_keras else node.operation

    def traverse(self, model: tf.keras.Model):
        """Traverse the model to generate the graph.

        Args:
            model (tf.keras.Model): The model.
        """
        q = deque()
        for input_tensor in model.inputs:
            # model.inputs is List[KerasTensor]
            # model.inputs[i].node.layer provides the layer
            # For Keras 3.x, this is
            #   model.inputs[i]._keras_history[0]._inbound_nodes[0].operation
            # model.inputs[i].node.layer.outbound_nodes[j] is needed
            # For keras 3.x, this is
            #   model.inputs[i]._keras_history[0]._outbound_nodes[j].operation
            # to find the next layers
            keras_node = self._get_input_tensor_node(input_tensor)
            q.append(keras_node)
            self.input_layer_names.append(self._get_node_layer(keras_node).name)

        while q:
            keras_node = q.popleft()

            layer = self._get_node_layer(keras_node)
            layer_name = layer.name
            if layer_name not in self._graph:
                node = TFLayerNode(layer_name)
                self._graph[layer_name] = node
            else:
                node = self._graph[layer_name]

            for outbound_keras_node in self._get_outbound_nodes(layer):
                output_layer_name = self._get_node_layer(outbound_keras_node).name
                if output_layer_name not in self._graph:
                    output_layer_node = TFLayerNode(output_layer_name)
                    self._graph[output_layer_name] = output_layer_node
                else:
                    output_layer_node = self._graph[output_layer_name]

                node.add_child(output_layer_node)
                q.append(outbound_keras_node)

        return
