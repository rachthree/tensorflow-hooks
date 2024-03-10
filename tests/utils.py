"""Test utilities."""
from collections import deque
from typing import Mapping

import tensorflow as tf
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
    """Graph to describe a TF Keras model."""

    def __init__(self, model: tf.keras.Model):
        """Initialize a TFGraph.

        Args:
            model (tf.keras.Model): The model.
        """
        self._graph = {}
        self.input_layer_names = []
        self.traverse(model)

    def __getitem__(self, layer_name: str):
        """Provide the node.

        Args:
            layer_name (str): The layer name.

        Returns:
            The TFLayerNode for the layer.
        """
        return self._graph[layer_name]

    def traverse(self, model: tf.keras.Model):
        """Traverse the model to generate the graph.

        Args:
            model (tf.keras.Model): The model.
        """
        q = deque()
        for input_tensor in model.inputs:
            # model.inputs is List[KerasTensor]
            # model.inputs[i].node.layer provides the layer
            # model.inputs[i].node.layer.outbound_nodes[j] is needed
            # to find the next layers
            q.append(input_tensor.node)
            self.input_layer_names.append(input_tensor.node.layer.name)

        while q:
            tf_node = q.popleft()

            layer = tf_node.layer
            layer_name = layer.name
            if layer_name not in self._graph:
                node = TFLayerNode(layer_name)
                self._graph[layer_name] = node
            else:
                node = self._graph[layer_name]

            for outbound_tf_node in layer.outbound_nodes:
                output_layer_name = outbound_tf_node.layer.name
                if output_layer_name not in self._graph:
                    output_layer_node = TFLayerNode(output_layer_name)
                    self._graph[output_layer_name] = output_layer_node
                else:
                    output_layer_node = self._graph[output_layer_name]

                node.add_child(output_layer_node)
                q.append(outbound_tf_node)

        return
