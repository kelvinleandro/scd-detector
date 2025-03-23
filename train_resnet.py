import pickle
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from models.resnet import get_model
from utils import sample_patient_data
from collections import Counter

HAS_LIMIT = True
NUM_LIMIT = 20

weights_path = "resnet_weights"
os.makedirs(weights_path, exist_ok=True)

data = {
    "x_train": None,
    "x_val": None,
    "y_train": None,
    "y_val": None,
    "pid_train": None,
}

data_path = "data/preprocessed_1class_standard"

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

print("LOADING DATA...")
for key in data.keys():
    with open(os.path.join(data_path, f"{key}.pkl"), "rb") as f:
        data[key] = pickle.load(f)

print(f"Train: {Counter(data['y_train'])}")
print(f"Validation: {Counter(data['y_val'])}")

if HAS_LIMIT:
    print(f"Old training len: {len(data['y_train'])}")
    tmp_x, tmp_y, tmp_pids = sample_patient_data(
        data["x_train"],
        data["y_train"],
        data["pid_train"],
        n=NUM_LIMIT,
        random_seed=42,
    )
    data["x_train"] = tmp_x
    data["y_train"] = tmp_y
    data["pid_train"] = tmp_pids
    print(f"New training len: {len(data['y_train'])}")
    print(f"Train: {Counter(data['y_train'])}")

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(data["y_train"]), y=data["y_train"]
)
class_weights = {0: class_weights[0], 1: class_weights[1]}

if len(data["y_val"].shape) == 1:
    data["y_train"] = data["y_train"].reshape(-1, 1)
    data["y_val"] = data["y_val"].reshape(-1, 1)
print("DATA READ!")

print("CREATING MODEL...")
model = get_model(n_classes=1, input_shape=(128 * 60, 1), dropout_keep_prob=0.5)
print(model.summary())

# Optimization settings
print("COMPILING MODEL...")
loss = "binary_crossentropy"
lr = 0.0001
batch_size = 32
opt = Adam(lr, weight_decay=1e-03)
model.compile(
    loss=loss,
    optimizer=opt,
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.F1Score(),
    ],
)

callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss", mode="min", factor=0.1, patience=7, min_lr=1e-07
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        min_delta=0.00001,
    ),
]
callbacks += [
    ModelCheckpoint(os.path.join(weights_path, "backup_model_last.keras")),
    ModelCheckpoint(
        os.path.join(weights_path, "backup_model_best.keras"), save_best_only=True
    ),
]

print("START TRAINING...")
history = model.fit(
    data["x_train"],
    data["y_train"],
    batch_size=128,
    epochs=70,
    callbacks=callbacks,
    validation_data=(data["x_val"], data["y_val"]),
    verbose=1,
    class_weights=class_weights,
)

# Save final result
model.save(os.path.join(weights_path, "final_model.keras"))
