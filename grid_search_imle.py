from itertools import product

import main_preprocessing
from configs import (
    imlenet_config,
    log_wandb_config,
    preprocess_config,
    transfer_learning_config,
)
from data.datagen import DataGen
import train
import test
from utils import sample_patient_data

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

run_transfer_learning = True

preprocess_conf = preprocess_config.Config()

imlenet_conf = imlenet_config.Config()

log_wandb_conf = log_wandb_config.Config()

transfer_learning_conf = transfer_learning_config.Config()
transfer_learning_conf.batch_size = imlenet_conf.batch_size
imlenet_conf.decision_threshold = transfer_learning_conf.decision_threshold

data_split_result = main_preprocessing.run_preprocessing_steps(
    preprocess_conf,
    compare=False,
    data_path="preprocessed_60s_onehot_beatSeg_standard",
)
# tmp_x, tmp_y, tmp_pids = sample_patient_data(
#     data_split_result["x_train"],
#     data_split_result["y_train"],
#     data_split_result["pid_train"],
#     n=50,
#     random_seed=42,
# )
# data_split_result["y_train"] = data_split_result["y_train"].reshape(-1, 1)
# data_split_result["y_val"] = data_split_result["y_val"].reshape(-1, 1)
# data_split_result["y_test"] = data_split_result["y_test"].reshape(-1, 1)

# limit_opts = [(False, None)]
# beat_len_opts = [16, 32, 64]
# beat_len_opts = [64, 128]
# blocks_list_opts = [[2, 2], [2, 2, 2], [3, 4, 3], [2, 2, 2, 2]]
# blocks_list_opts = [[1], [2], [2, 2]]
# start_filters_opts = [16, 32]
# bs_opts = [256, 512]
# dropout_values = [0.5, 0.8]
# lr_opts = [0.01, 0.001, 0.0001]
# kernel_size_opts = [8, 16]

lstm_opts = [128, 256, 2048]
dense_opts = [[64], [128], [256], [512]]

# combinations = list(
#     product(beat_len_opts, blocks_list_opts, start_filters_opts, dropout_values, lr_opts)
# )
combinations = list(product(lstm_opts, dense_opts))

print(f"Total combinations: {len(combinations)}")
# for idx, (beat_len, num_blocks_list, start_filters, dropout, lr) in enumerate(
#     combinations
# ):
for idx, (lstm_units, dense_layers) in enumerate(combinations):
    print(f"Starting combination {idx+1}/{len(combinations)}")
    try:
        preprocess_conf = preprocess_config.Config()
        imlenet_conf = imlenet_config.Config()
        transfer_learning_conf = transfer_learning_config.Config()
        transfer_learning_conf.batch_size = imlenet_conf.batch_size
        imlenet_conf.decision_threshold = transfer_learning_conf.decision_threshold

        transfer_learning_conf.lstm_units = lstm_units
        transfer_learning_conf.dense_layers = dense_layers

        train_gen = DataGen(
            data_split_result["x_train"],
            data_split_result["y_train"],
            batch_size=imlenet_conf.batch_size,
        )
        val_gen = DataGen(
            data_split_result["x_val"],
            data_split_result["y_val"],
            batch_size=imlenet_conf.batch_size,
        )
        test_gen = DataGen(
            data_split_result["x_test"],
            data_split_result["y_test"],
            batch_size=imlenet_conf.batch_size,
        )

        hash_pids = {}
        model_path, history, wandb = train.train_model(
            train_gen,
            val_gen,
            imlenet_conf=imlenet_conf,
            preprocess_conf=preprocess_conf,
            log_wandb_conf=log_wandb_conf,
            transfer_learning_conf=transfer_learning_conf,
            transfer_learning=run_transfer_learning,
            hash_pids=hash_pids,
        )

        test.test_imlenet_model(
            model_path,
            test_gen,
            data_split_result,
            imle_conf=imlenet_conf,
            log_wandb=wandb,
            log_wandb_conf=log_wandb_conf,
            transfer_learning_conf=transfer_learning_conf,
            transfer_learning=run_transfer_learning,
        )
    except Exception as e:
        print(f"Combination {idx+1}/{len(combinations)} failed.")
        print(e)
