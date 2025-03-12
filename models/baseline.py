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
    Dropout,
    Flatten,
    Reshape,
)


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


def build_model(config) -> tf.keras.Model:
    # kerasTensor (simbolico) sendo passado p uma funcao
    # inputs = Input(shape=(1, config.signal_len, 1), batch_size=None)
    # x = K.reshape(inputs, (-1, config.beat_len, 1))
    # x = K.reshape(x, (-1, int(config.signal_len / config.beat_len), config.beat_len))

    inputs = Input(shape=(1, config.signal_len, 1), batch_size=None)
    x = Reshape((-1, config.beat_len, 1))(inputs)
    x = Reshape((int(config.signal_len / config.beat_len), config.beat_len))(x)
    x = Bidirectional(LSTM(config.lstm_units, return_sequences=True))(x)
    x = Dropout(config.dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(config.dense_size)(x)

    outputs = Dense(config.classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

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
    model._name = "Baseline"

    return model
