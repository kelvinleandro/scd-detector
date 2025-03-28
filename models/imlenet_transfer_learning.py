"""

Model version presented at https://ieeexplore.ieee.org/document/9658706
Implemented initially at https://github.com/likith012/IMLE-Net/blob/main/models/IMLENet.py

The version developed here generalizes the configuration, allowing different inputs for
lstm_units and num_filters.
Besides, it was included other options of attention layers.

"""

from typing import Tuple

from tensorflow.keras.models import Model, load_model
from models import imlenet
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.initializers import glorot_normal
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
from .reshape_layer import ReshapeLayer
from metrics import GeometricMean, Specificity, Recall


def build_transfer_model(model_conf, preprocess_conf) -> tf.keras.Model:
    if model_conf.att_layer == "Additive":
        loaded_model = load_model(
            model_conf.path_model,
            custom_objects={"AdditiveAttention": imlenet.AdditiveAttention},
            compile=False,
        )
    elif model_conf.att_layer == "DotProduct":
        loaded_model = load_model(
            model_conf.path_model,
            custom_objects={"DotProductAttention": imlenet.DotProductAttention},
            compile=False,
        )
    elif model_conf.att_layer == "MultiHead":
        loaded_model = load_model(
            model_conf.path_model,
            custom_objects={"MultiHeadAttention": imlenet.MultiHeadAttention},
            compile=False,
        )
    else:
        loaded_model = load_model(model_conf.path_model, compile=False)

    if not model_conf.fine_tuning:
        print("Freezing weights")
        for layer in loaded_model.layers:
            loaded_model.trainable = False
    else:
        for layer in loaded_model.layers:
            loaded_model.trainable = True
            if model_conf.retrain:
                if hasattr(layer, "kernel_initializer"):
                    print("Kernel init")
                    layer.kernel_initializer = glorot_normal()

    base_model_layers = loaded_model.layers[: model_conf.level_to_cut]
    new_base_model = Model(
        inputs=loaded_model.input, outputs=base_model_layers[-1].output
    )
    if not preprocess_conf.beat_segmentation:
        shape_of_data = int(preprocess_conf.seconds * preprocess_conf.fs)
        shape_ten_seconds = int(model_conf.transfer_seconds * preprocess_conf.fs)
    else:
        shape_of_data = int(
            preprocess_conf.seconds
            * int(preprocess_conf.fs * preprocess_conf.beat_percentage * 2)
        )
        shape_ten_seconds = int(model_conf.transfer_seconds * preprocess_conf.fs)
    print(shape_of_data, shape_ten_seconds, base_model_layers[-1].output.shape[-1])
    inputs = Input(shape=(1, shape_of_data, 1), batch_size=None)
    # x = K.reshape(inputs, (-1, 1, shape_ten_seconds, 1))
    x = ReshapeLayer((-1, 1, shape_ten_seconds, 1))(inputs)
    x = new_base_model(x)
    x = Flatten()(x)
    # x = K.reshape(x, (-1, int(shape_of_data / shape_ten_seconds), x.shape[-1]))
    x = ReshapeLayer((-1, int(shape_of_data / shape_ten_seconds), x.shape[-1]))(x)
    if model_conf.has_lstm:
        x = Bidirectional(
            LSTM(model_conf.lstm_units, return_sequences=True),
            merge_mode=model_conf.lstm_mergemode,
        )(x)
    x = Flatten()(x)
    for each_layer in model_conf.dense_layers:
        x = Dense(each_layer, activation="relu")(x)
        # x = Dense(each_layer)(x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        if model_conf.has_dropout:
            x = Dropout(model_conf.dropout_rate)(x)
    outputs = Dense(model_conf.classes, activation=model_conf.last_activation)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if model_conf.optimizer == "adam":
        optimizer_in_use = tf.keras.optimizers.Adam(
            learning_rate=model_conf.learning_rate, weight_decay=1e-6
        )
    elif model_conf.optimizer == "sgd":
        optimizer_in_use = tf.keras.optimizers.SGD(
            learning_rate=model_conf.learning_rate,
            weight_decay=1e-6,
            momentum=0.9,
            nesterov=True,
        )
    else:
        raise ValueError(f"Optimizer {model_conf.optimizer} not supported!")
        sys.exit(1)

    model.compile(
        optimizer=optimizer_in_use,
        loss=model_conf.loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(multi_label=True, name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.F1Score(average="macro", threshold=0.5, name="f1_score"),
            GeometricMean(),
            Specificity(),
            Recall(name="recall_argmax"),
        ],
    )
    model._name = "IMLE-Net_Transfer_Learning"
    return model
