import main_preprocessing
from configs import log_wandb_config, preprocess_config, baseline_config
from data.datagen import DataGen

import train
import test

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

run_transfer_learning = True

preprocess_conf = preprocess_config.Config()

baseline_conf = baseline_config.Config()

log_wandb_conf = log_wandb_config.Config()

data_split_result = main_preprocessing.run_preprocessing_steps(preprocess_conf)
baseline_conf.signal_len = data_split_result["x_train"].shape[1]

print(data_split_result["x_train"].shape, data_split_result["x_val"].shape)
print(data_split_result["y_train"].shape, data_split_result["y_val"].shape)

train_gen = DataGen(
    data_split_result["x_train"],
    data_split_result["y_train"],
    batch_size=baseline_conf.batch_size,
)
val_gen = DataGen(
    data_split_result["x_val"],
    data_split_result["y_val"],
    batch_size=baseline_conf.batch_size,
)
test_gen = DataGen(
    data_split_result["x_test"],
    data_split_result["y_test"],
    batch_size=baseline_conf.batch_size,
)

model_path, history, wandb = train.train_baseline_model(
    train_gen,
    val_gen,
    model_conf=baseline_conf,
    preprocess_conf=preprocess_conf,
    log_wandb_conf=log_wandb_conf,
    hash_pids={},
)

print("Test data:")
print(data_split_result["x_test"].shape, data_split_result["y_test"].shape)

test.test_baseline_model(
    model_path,
    test_gen,
    data_split_result,
    model_conf=baseline_conf,
    log_wandb=wandb,
    log_wandb_conf=log_wandb_conf,
)
