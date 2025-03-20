import pickle
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from models.resnet import get_model

weights_path = "resnet_weights"
os.makedirs(weights_path, exist_ok=True)

data = {
    "x_train": None,
    "x_val": None,
    "x_test": None,
    "y_train": None,
    "y_val": None,
    "y_test": None,
    "pid_train": None,
    "pid_val": None,
    "pid_test": None,
}

data_path = "data/music_preprocessed_10s_hotencodeFalse_standard"

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

print("LOADING DATA...")
for key in data.keys():
    with open(os.path.join(data_path, f"{key}.pkl"), "rb") as f:
        data[key] = pickle.load(f)
print("DATA READ!")

print("CREATING MODEL...")
model = get_model(1)

# Optimization settings
print("COMPILING MODEL...")
loss = "binary_crossentropy"
lr = 0.001
batch_size = 64
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
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=7, min_lr=1e-07),
    EarlyStopping(
        patience=10,  # Patience should be larger than the one in ReduceLROnPlateau
        min_delta=0.00001,
    ),
]
callbacks += [
    ModelCheckpoint(os.path.join(weights_path, "backup_model_last.keras")),
    ModelCheckpoint(
        os.path.join(weights_path, "backup_model_best.keras"), save_best_only=True
    ),
]

print(model.summary())

print("START TRAINING...")
history = model.fit(
    data["x_train"],
    data["y_train"],
    batch_size=64,
    epochs=70,
    callbacks=callbacks,
    validation_data=(data["x_val"], data["y_val"]),
    verbose=1,
)
# Save final result
model.save(os.path.join(weights_path, "final_model.keras"))
