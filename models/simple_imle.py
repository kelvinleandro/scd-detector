"""

Model version presented at https://ieeexplore.ieee.org/document/9658706
Implemented initially at https://github.com/likith012/IMLE-Net/blob/main/models/IMLENet.py

The version developed here generalizes the configuration, allowing different inputs for
lstm_units and num_filters.
Besides, it was included other options of attention layers.

"""

from typing import Tuple

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Input,
    Layer,
    ReLU,
    Flatten,
    Dropout,
)

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from typing import Tuple


class MultiHeadAttention(Layer):
    """A class used to build the multi-head attention layer for time series data.

    Attributes
    ----------
    return_sequences: bool, optional
        If False, returns the calculated attention weighted sum of an ECG signal. (default: False)
    num_heads: int, optional
        The number of attention heads. (default: 8)
    dim: int, optional
        The dimension of each attention head. (default: 64)

    Methods
    -------
    build(input_shape)
        Sets the weights for calculating the attention layer.
    call(x)
        Calculates the attention weights.
    get_config()
        Useful for serialization of the attention layer.
    """

    def __init__(
        self,
        return_sequences: bool = False,
        num_heads: int = 8,
        dim: int = 64,
        **kwargs,
    ) -> None:
        self.return_sequences = return_sequences
        self.num_heads = num_heads
        self.dim = dim
        self.depth = dim // num_heads
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        """Builds the attention layer."""
        self.query_weights = [
            self.add_weight(
                name=f"query_weight_{i}",
                shape=(input_shape[-1], self.depth),
                initializer="normal",
            )
            for i in range(self.num_heads)
        ]

        self.key_weights = [
            self.add_weight(
                name=f"key_weight_{i}",
                shape=(input_shape[-1], self.depth),
                initializer="normal",
            )
            for i in range(self.num_heads)
        ]

        self.value_weights = [
            self.add_weight(
                name=f"value_weight_{i}",
                shape=(input_shape[-1], self.depth),
                initializer="normal",
            )
            for i in range(self.num_heads)
        ]

        self.Wo = self.add_weight(
            name="output_weight",
            shape=(
                self.num_heads * self.depth,
                input_shape[-1],
            ),  # Ensuring the output shape matches input shape
            initializer="normal",
        )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the attention weights."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Apply linear projections and split into multiple heads
        queries = [
            K.dot(x, wq) for wq in self.query_weights
        ]  # List of tensors (batch_size, seq_len, depth)
        keys = [
            K.dot(x, wk) for wk in self.key_weights
        ]  # List of tensors (batch_size, seq_len, depth)
        values = [
            K.dot(x, wv) for wv in self.value_weights
        ]  # List of tensors (batch_size, seq_len, depth)

        # Compute scaled dot-product attention for each head
        attention_heads = []
        attention_weights = []
        for i in range(self.num_heads):
            attention_scores = tf.matmul(
                queries[i], keys[i], transpose_b=True
            )  # Shape: (batch_size, seq_len, seq_len)
            attention_scores = attention_scores / tf.math.sqrt(
                tf.cast(self.depth, tf.float32)
            )
            weights = K.softmax(
                attention_scores, axis=-1
            )  # Shape: (batch_size, seq_len, seq_len)
            attention_output = tf.matmul(
                weights, values[i]
            )  # Shape: (batch_size, seq_len, depth)
            attention_heads.append(attention_output)
            attention_weights.append(weights)

        # Concatenate attention heads and apply final linear transformation
        concatenated_heads = tf.concat(
            attention_heads, axis=-1
        )  # Shape: (batch_size, seq_len, num_heads * depth)
        output = K.dot(
            concatenated_heads, self.Wo
        )  # Shape: (batch_size, seq_len, input_dim)

        if self.return_sequences:
            return output, attention_weights

        return K.sum(output, axis=1), attention_weights

    def get_config(self):
        """Returns the config of the attention layer. Useful for serialization."""
        config = super(MultiHeadAttention, self).get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "num_heads": self.num_heads,
                "dim": self.dim,
            }
        )
        return config


class DotProductAttention(Layer):
    """A class used to build the dot-product attention layer.

    Attributes
    ----------
    return_sequences: bool, optional
        If False, returns the calculated attention weighted sum of an input sequence. (default: False)

    Methods
    -------
    build(input_shape)
        Sets the weights for calculating the attention layer.
    call(x)
        Calculates the dot-product attention.
    get_config()
        Useful for serialization of the attention layer.

    """

    def __init__(self, return_sequences: bool = False, **kwargs):
        self.return_sequences = return_sequences
        super(DotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="normal",
            name="WQ",
        )
        self.WK = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="normal",
            name="WK",
        )
        self.WV = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="normal",
            name="WV",
        )
        super(DotProductAttention, self).build(input_shape)

    def call(self, x):
        """Calculates the dot-product attention.

        Parameters
        ----------
        x: tf.Tensor
            The input tensor.

        Returns
        -------
        tf.Tensor
            The attention-weighted sum of the input tensor.
        """

        WQ = K.dot(x, self.WQ)
        WK = K.dot(x, self.WK)
        WV = K.dot(x, self.WV)

        # Implement the dot-product attention mechanism
        attention = tf.matmul(WQ, WK, transpose_b=True)
        attention = attention / K.int_shape(WK)[-1] ** 0.5
        attention = tf.nn.softmax(attention, axis=-1)
        output = tf.matmul(attention, WV)

        if self.return_sequences:
            return output, attention
        else:
            return K.sum(output, axis=1), attention

    def get_config(self):
        """Returns the config of the attention layer. Useful for serialization."""

        base_config = super().get_config()
        config = {"return_sequences": self.return_sequences}
        return dict(list(base_config.items()) + list(config.items()))


class AdditiveAttention(Layer):
    """A class used to build the feed-forward attention layer.

    Attributes
    ----------
    return_sequences: bool, optional
        If False, returns the calculated attention weighted sum of an ECG signal. (default: False)
    dim: int, optional
        The dimension of the attention layer. (default: 64)

    Methods
    -------
    build(input_shape)
        Sets the weights for calculating the attention layer.
    call(x)
        Calculates the attention weights.
    get_config()
        Useful for serialization of the attention layer.

    """

    def __init__(self, return_sequences: bool = False, dim: int = 64, **kwargs) -> None:
        self.return_sequences = return_sequences
        self.dim = dim
        super(AdditiveAttention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        """Builds the attention layer.

        alpha = softmax(V.T * tanh(W.T * x + b))

        Parameters
        ----------
        W: tf.Tensor
            The weights of the attention layer.
        b: tf.Tensor
            The bias of the attention layer.
        V: tf.Tensor
            The secondary weights of the attention layer.

        """

        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], self.dim),
            initializer="normal",
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(self.dim,),
            initializer="zeros",
        )
        self.V = self.add_weight(name="Vatt", shape=(self.dim, 1), initializer="normal")
        super(AdditiveAttention, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the attention weights.

        Parameters
        ----------
        x: tf.Tensor
            The input tensor.

        Returns
        -------
        tf.Tensor
            The attention weighted sum of the input tensor.
        """

        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.dot(e, self.V)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output, a

        return K.sum(output, axis=1), a

    def get_config(self):
        """Returns the config of the attention layer. Useful for serialization."""

        config = super(AdditiveAttention, self).get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "dim": self.dim,
            }
        )

        return config


def residual_block(
    x: tf.Tensor,
    downsample: bool,
    filters: int,
    kernel_size: int = 8,
    dropout_rate: float = 0.5,
) -> tf.Tensor:
    """[summary]

    Parameters
    ----------
    x: tf.Tensor
        The input tensor.
    downsample: bool
        If True, downsamples the input tensor.
    filters: int
        The number of filters in the 1D-convolutional layers.
    kernel_size: int, optional
        The kernel size of the 1D-convolutional layers. (default: 8)

    Returns
    -------
    tf.Tensor
        The output tensor of the residual block.
    """

    y = Conv1D(
        kernel_size=kernel_size,
        strides=(1 if not downsample else 2),
        filters=filters,
        padding="same",
    )(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(dropout_rate)(y)
    y = Conv1D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)

    if downsample:
        x = Conv1D(kernel_size=1, strides=2, filters=filters, padding="same")(x)

    out = Add()([x, y])
    out = BatchNormalization()(out)
    out = ReLU()(out)
    out = Dropout(dropout_rate)(out)

    return out


def build_simple_imle_net(config, attention_layer: Layer, sub=False) -> tf.keras.Model:
    """Builds the IMLE-Net model.

    Parameters
    ----------
    config: imle_config
        The configs for building the model.
    sub: bool, optional
        For sub-diagnostic diseases of MI. (default: False)

    Returns
    -------
    tf.keras.Model
        The keras sequential model.

    """

    inputs = Input(shape=(config.input_channels, config.signal_len, 1), batch_size=None)

    # Beat Level
    x = K.reshape(inputs, (-1, config.beat_len, 1))

    x = Conv1D(
        filters=config.start_filters,
        kernel_size=config.kernel_size,
        padding="same",
    )(x)
    x = Activation("relu")(x)

    num_filters = config.start_filters
    for i in range(len(config.num_blocks_list)):
        num_blocks = config.num_blocks_list[i]
        for j in range(num_blocks):
            x = residual_block(
                x,
                downsample=(j == 0 and i != 0),
                filters=num_filters,
                dropout_rate=config.dropout_rate,
            )
        num_filters *= 2

    x, _ = attention_layer(name="beat_att")(x)

    # Rhythm level
    x = K.reshape(
        x, (-1, int(config.signal_len / config.beat_len), int(num_filters / 2))
    )
    x = Bidirectional(LSTM(config.lstm_units, return_sequences=True))(x)
    x, _ = attention_layer(name="rhythm_att")(x)
    x = Flatten()(x)
    outputs = Dense(config.classes, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    if not sub:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(multi_label=True),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.F1Score(average="macro", threshold=0.5),
            ],
        )
        model._name = "IMLE-Net"

    return model
