import pickle
import os
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tqdm import tqdm

weights_path = "resnet_weights"

data = {
    "x_test": None,
    "y_test": None,
    "pid_test": None,
}

data_path = "data/music_preprocessed_10s_hotencodeFalse_standard"

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

print("LOADING DATA...")
for key in data.keys():
    with open(os.path.join(data_path, f"{key}.pkl"), "rb") as f:
        data[key] = pickle.load(f)
print("DATA READ!")

print("LOADING WEIGHTS...")
model = tf.keras.models.load_model(f"{weights_path}/final_model.keras")
print(model.summary())

n_classes = data["y_test"].shape[-1]

all_pred_prob = []
for idx, input_x in tqdm(
    enumerate(data["x_test"]), total=len(data["x_test"]), desc="validating"
):
    if idx >= len(data["x_test"]):
        print("Forcing stop: Exceeded dataset length")
        break
    pred = model(input_x)
    all_pred_prob.append(pred)

print("After predicting")
# all_pred_prob = np.concatenate(all_pred_prob)
if n_classes > 1:
    all_pred = np.argmax(all_pred_prob, axis=1)
else:
    all_pred = all_pred_prob

final_pred = []
final_gt = []

print("Generating classification report...")
for i_pid in np.unique(data["pid_test"]):
    tmp_pred = (all_pred[data["pid_test"] == i_pid] > 0.5).astype(int).ravel()

    if n_classes > 1:
        tmp_gt = np.argmax(
            data["y_test"][data["pid_test"] == i_pid],
            axis=1,
        )
    else:
        tmp_gt = data["y_test"].flatten()[data["pid_test"] == i_pid]

    final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
    final_gt.append(Counter(tmp_gt).most_common(1)[0][0])

tmp_report = classification_report(final_gt, final_pred, output_dict=True)
print("Classification report counter:")
print(tmp_report)

print("Confusion matrix:")
print(confusion_matrix(final_gt, final_pred))
