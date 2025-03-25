import json
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model

import hashlib
import os

from models import imlenet, baseline, imlenet_transfer_learning
import wandb


def freeze_layers(model):
    for i in model.layers:
        i.trainable = True
        if isinstance(i, Model):
            freeze_layers(i)
    return model


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def initialize_wandb(model_conf, preprocess_conf, log_wandb_conf, hash_pids=None):

    model_conf.__dict__.update(preprocess_conf.__dict__)
    model_conf.__dict__.update(hash_pids)
    conf_str = str(model_conf.__dict__)
    hash_object = hashlib.sha256(conf_str.encode())
    hash_value = hash_object.hexdigest()

    fname = os.path.sep.join([f"weights-{hash_value}"])

    print(model_conf.__dict__)

    create_directory(log_wandb_conf.project)
    wandb.init(
        project=log_wandb_conf.project,
        entity=log_wandb_conf.entity,
        config=model_conf.__dict__,
        name=hash_value,
    )

    return fname


class BestWeightsCallback(Callback):
    def __init__(self, metric):
        super(BestWeightsCallback, self).__init__()
        self.metric = metric
        self.best_weights = None
        self.best_val_loss = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        current_val = logs.get(self.metric)
        if current_val is None:
            print(f"Warning: Metric '{self.metric}' not found in logs.")
            return

        if epoch == 0 or current_val > self.best_val_loss:
            print(f"Saving model weights... {self.metric}={current_val}")
            self.best_val_loss = current_val
            self.best_weights = self.model.get_weights()


class WandbWeightsCallback(Callback):
    def __init__(self):
        super(WandbWeightsCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if "loss" in logs:
            wandb.log({"loss": logs["loss"]}, step=epoch)
        if "accuracy" in logs:
            wandb.log({"accuracy": logs["accuracy"]}, step=epoch)
        if "auc" in logs:
            wandb.log({"auc": logs["auc"]}, step=epoch)
        if "recall" in logs:
            wandb.log({"recall": logs["recall"]}, step=epoch)
        if "f1_score" in logs:
            wandb.log({"f1_score": logs["f1_score"]}, step=epoch)
        if "val_loss" in logs:
            wandb.log({"val_loss": logs["val_loss"]}, step=epoch)
        if "val_accuracy" in logs:
            wandb.log({"val_accuracy": logs["val_accuracy"]}, step=epoch)
        if "val_auc" in logs:
            wandb.log({"val_auc": logs["val_auc"]}, step=epoch)
        if "val_recall" in logs:
            wandb.log({"val_recall": logs["val_recall"]}, step=epoch)
        if "val_f1_score" in logs:
            wandb.log({"val_f1_score": logs["val_f1_score"]}, step=epoch)
        if "val_specificity" in logs:
            wandb.log({"val_specificity": logs["val_specificity"]}, step=epoch)
        if "val_geometric_mean" in logs:
            wandb.log({"val_geometric_mean": logs["val_geometric_mean"]}, step=epoch)


def train_baseline_model(
    train_gen,
    val_gen,
    model_conf=None,
    preprocess_conf=None,
    log_wandb_conf=None,
    hash_pids=None,
):

    keras.utils.set_random_seed(model_conf.seed)

    metric = model_conf.metric

    # Early Stopping
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        min_delta=0.0001,
        patience=model_conf.patience,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    fname = initialize_wandb(
        model_conf, preprocess_conf, log_wandb_conf, hash_pids=hash_pids
    )

    if os.path.exists(f"{log_wandb_conf.project}/{fname}_full_model.keras"):
        print("This configuration was executed already!")
        return fname, None, wandb

    model = baseline.build_model(model_conf)

    best_weights_callback = BestWeightsCallback(metric="val_f1_score")
    wand_callback = WandbWeightsCallback()

    callbacks = [stop_early, best_weights_callback, wand_callback]

    print(model.summary())
    history = model.fit(
        train_gen,
        epochs=100,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # freeze model

    model.set_weights(best_weights_callback.best_weights)

    model_freezed = freeze_layers(model)

    # Save the best weights to an HDF5 file
    model_freezed.save(f"{log_wandb_conf.project}/{fname}_full_model.keras")

    return fname, history, wandb


def train_model(
    train_gen,
    val_gen,
    imlenet_conf=None,
    preprocess_conf=None,
    log_wandb_conf=None,
    transfer_learning_conf=None,
    transfer_learning=False,
    hash_pids=None,
):
    keras.utils.set_random_seed(imlenet_conf.seed)

    metric = imlenet_conf.metric

    # Early Stopping
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        min_delta=0.0001,
        patience=imlenet_conf.patience,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    if transfer_learning:
        fname = initialize_wandb(
            transfer_learning_conf, preprocess_conf, log_wandb_conf, hash_pids=hash_pids
        )
    else:
        fname = initialize_wandb(
            imlenet_conf, preprocess_conf, log_wandb_conf, hash_pids=hash_pids
        )

    if os.path.exists(f"{log_wandb_conf.project}/{fname}_full_model.keras"):
        print("This configuration was executed already!")
        return fname, None, wandb

    if imlenet_conf.att_layer == "Additive":
        att_layer = imlenet.AdditiveAttention
    elif imlenet_conf.att_layer == "DotProduct":
        att_layer = imlenet.DotProductAttention
    elif imlenet_conf.att_layer == "MultiHead":
        att_layer = imlenet.MultiHeadAttention
    else:
        raise ValueError(f"Attention layer {imlenet_conf.att_layer} not supported!")
        sys.exit(1)

    if transfer_learning:
        model = imlenet_transfer_learning.build_transfer_model(
            transfer_learning_conf, preprocess_conf
        )
    else:
        model = imlenet.build_imle_net(imlenet_conf, att_layer)

    best_weights_callback = BestWeightsCallback(metric)
    wand_callback = WandbWeightsCallback()
    reduce_lr = (
        keras.callbacks.ReduceLROnPlateau(
            monitor=metric, mode="max", factor=0.1, patience=7, min_lr=1e-07
        ),
    )
    callbacks = [stop_early, best_weights_callback, wand_callback, reduce_lr]
    # "backup" models at each epoch
    callbacks += [
        ModelCheckpoint("./backup_model_last.keras", monitor=metric, mode="max"),
        ModelCheckpoint(
            "./backup_model_best.keras", monitor=metric, mode="max", save_best_only=True
        ),
    ]

    print(model.summary())
    history = model.fit(
        train_gen,
        epochs=100,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # freeze model

    model.set_weights(best_weights_callback.best_weights)

    model_freezed = freeze_layers(model)

    # Save the best weights to an HDF5 file
    model_freezed.save(f"{log_wandb_conf.project}/{fname}_full_model.keras")

    return fname, history, wandb
