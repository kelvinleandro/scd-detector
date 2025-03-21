import numpy as np
from numpy.random import default_rng

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import main_preprocessing
from configs import (
    imlenet_config,
    log_wandb_config,
    preprocess_config,
    transfer_learning_config,
)
from data.datagen import DataGen
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import hashlib
from datetime import datetime

import train
import test

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# exit(1)
run_transfer_learning = False

preprocess_conf = preprocess_config.Config()

imlenet_conf = imlenet_config.Config()

log_wandb_conf = log_wandb_config.Config()

transfer_learning_conf = transfer_learning_config.Config()
transfer_learning_conf.batch_size = imlenet_conf.batch_size
imlenet_conf.decision_threshold = transfer_learning_conf.decision_threshold

data_split_result = main_preprocessing.run_preprocessing_steps(
    preprocess_conf, compare=False
)
imlenet_conf.signal_len = data_split_result["x_train"].shape[1]

print("SHAPEEEEEEEEEEEEEEEEE")
print(data_split_result["y_test"].shape, data_split_result["pid_test"].shape)

if preprocess_conf.kfolds:
    datax_total = np.append(
        data_split_result["x_train"], data_split_result["x_val"], axis=0
    )
    datax_total = np.append(datax_total, data_split_result["x_test"], axis=0)

    datay_total = np.append(
        data_split_result["y_train"], data_split_result["y_val"], axis=0
    )
    datay_total = np.append(datay_total, data_split_result["y_test"], axis=0)

    pid_total = np.append(
        data_split_result["pid_train"], data_split_result["pid_val"] + 1000, axis=0
    )
    pid_total = np.append(pid_total, data_split_result["pid_test"] + 10000, axis=0)

    kf = StratifiedKFold(
        n_splits=transfer_learning_conf.test_factor_kfold,
        random_state=preprocess_conf.seed,
        shuffle=True,
    )
    rng = default_rng(seed=preprocess_conf.seed)

    y_total_unique = []
    for each_id in np.unique(pid_total):
        y_total_unique.append(
            np.argmax(np.mean(datay_total[pid_total == each_id], axis=0))
        )
    y_total_unique = np.array(y_total_unique)
    print(np.unique(pid_total).astype(int))
    preprocess_conf.execution = datetime.now()
    # preprocess_conf.execution = datetime(2024,6,1,21,48,53,335076)
    for i, (train_index_idx, test_index_idx) in enumerate(
        kf.split(np.unique(pid_total).astype(int), y_total_unique)
    ):

        train_index = np.unique(pid_total)[train_index_idx]
        test_index = np.unique(pid_total)[test_index_idx]

        print("Running fold " + str(i))
        data_kfold = data_split_result.copy()
        val_index = np.unique(train_index)[
            rng.choice(
                len(train_index),
                size=int(len(train_index) / transfer_learning_conf.val_factor_kfold),
                replace=False,
            )
        ]

        data_kfold["pid_test"] = pid_total[np.isin(pid_total, test_index)]
        data_kfold["pid_train"] = pid_total[
            (~np.isin(pid_total, test_index)) & (~np.isin(pid_total, val_index))
        ]
        data_kfold["pid_val"] = pid_total[np.isin(pid_total, val_index)]

        print(
            len(np.unique(data_kfold["pid_train"])),
            len(np.unique(data_kfold["pid_val"])),
            len(np.unique(data_kfold["pid_test"])),
        )

        data_kfold["x_test"] = datax_total[~np.isin(pid_total, train_index)]
        data_kfold["x_train"] = datax_total[
            (~np.isin(pid_total, test_index)) & (~np.isin(pid_total, val_index))
        ]
        data_kfold["x_val"] = datax_total[np.isin(pid_total, val_index)]

        data_kfold["y_test"] = datay_total[~np.isin(pid_total, train_index)]
        data_kfold["y_train"] = datay_total[
            (~np.isin(pid_total, test_index)) & (~np.isin(pid_total, val_index))
        ]
        data_kfold["y_val"] = datay_total[np.isin(pid_total, val_index)]

        pid_trainval = np.append(data_kfold["pid_train"], data_kfold["pid_val"], axis=0)
        datay_trainval = np.append(data_kfold["y_train"], data_kfold["y_val"], axis=0)
        y_train_val_unique = []
        for each_id in np.unique(pid_trainval):
            y_train_val_unique.append(
                np.argmax(np.mean(datay_trainval[pid_trainval == each_id], axis=0))
            )
        y_total_unique = np.array(y_total_unique)
        new_train_index, new_val_index, _, _ = train_test_split(
            np.unique(pid_trainval).astype(int),
            y_train_val_unique,
            test_size=(1 / transfer_learning_conf.val_factor_kfold),
            random_state=preprocess_conf.seed,
        )

        data_kfold["pid_train"] = pid_total[np.isin(pid_total, new_train_index)]
        data_kfold["pid_val"] = pid_total[np.isin(pid_total, new_val_index)]

        data_kfold["x_train"] = datax_total[np.isin(pid_total, new_train_index)]
        data_kfold["x_val"] = datax_total[np.isin(pid_total, new_val_index)]

        data_kfold["y_train"] = datay_total[np.isin(pid_total, new_train_index)]
        data_kfold["y_val"] = datay_total[np.isin(pid_total, new_val_index)]

        pid_train_hash = hashlib.sha256(
            str(np.sort(np.unique(data_kfold["pid_train"]).astype(int))).encode()
        ).hexdigest()
        pid_val_hash = hashlib.sha256(
            str(np.sort(np.unique(data_kfold["pid_val"]).astype(int))).encode()
        ).hexdigest()
        pid_test_hash = hashlib.sha256(
            str(np.sort(np.unique(data_kfold["pid_test"]).astype(int))).encode()
        ).hexdigest()
        print(
            np.unique(data_kfold["pid_train"]).astype(int),
            np.unique(data_kfold["pid_val"]).astype(int),
            np.unique(data_kfold["pid_test"]).astype(int),
        )
        hash_pids = {
            "pid_train": pid_train_hash,
            "pid_val": pid_val_hash,
            "pid_test": pid_test_hash,
        }
        # print(hash_pids)

        train_gen = DataGen(
            data_kfold["x_train"],
            data_kfold["y_train"],
            batch_size=imlenet_conf.batch_size,
        )
        val_gen = DataGen(
            data_kfold["x_val"], data_kfold["y_val"], batch_size=imlenet_conf.batch_size
        )
        test_gen = DataGen(
            data_kfold["x_test"],
            data_kfold["y_test"],
            batch_size=imlenet_conf.batch_size,
        )

        preprocess_conf.fold = i

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
            data_kfold,
            imle_conf=imlenet_conf,
            log_wandb=wandb,
            log_wandb_conf=log_wandb_conf,
            transfer_learning_conf=transfer_learning_conf,
            transfer_learning=run_transfer_learning,
        )

else:
    # np.random.seed(42)
    # np.random.shuffle(data_split_result["x_train"])
    # np.random.shuffle(data_split_result["y_train"])
    import tensorflow as tf

    datax_total = np.append(
        data_split_result["x_train"], data_split_result["x_val"], axis=0
    )
    datay_total = np.append(
        data_split_result["y_train"], data_split_result["y_val"], axis=0
    )
    pid_total = np.append(
        data_split_result["pid_train"], data_split_result["pid_val"] + 1000, axis=0
    )
    preprocess_conf.execution = datetime.now()

    kf = StratifiedKFold(
        n_splits=transfer_learning_conf.test_factor_kfold,
        random_state=preprocess_conf.seed,
        shuffle=True,
    )
    rng = default_rng(seed=preprocess_conf.seed)

    y_total_unique = []
    for each_id in np.unique(pid_total):
        y_total_unique.append(
            np.argmax(np.mean(datay_total[pid_total == each_id], axis=0))
        )
    y_total_unique = np.array(y_total_unique)
    print(np.unique(pid_total).astype(int))

    for i, (train_index_idx, val_index_idx) in enumerate(
        kf.split(np.unique(pid_total).astype(int), y_total_unique)
    ):

        train_index = np.unique(pid_total)[train_index_idx]
        val_index = np.unique(pid_total)[val_index_idx]

        print("Running fold " + str(i))
        data_kfold = data_split_result.copy()
        preprocess_conf.fold = i
        # preprocess_conf.execution= datetime(2024, 6, 8, 0, 20, 33, 727945)
        data_kfold["pid_train"] = pid_total[np.isin(pid_total, train_index)]
        data_kfold["pid_val"] = pid_total[np.isin(pid_total, val_index)]

        data_kfold["x_train"] = datax_total[np.isin(pid_total, train_index)]
        data_kfold["x_val"] = datax_total[np.isin(pid_total, val_index)]

        data_kfold["y_train"] = datay_total[np.isin(pid_total, train_index)]
        data_kfold["y_val"] = datay_total[np.isin(pid_total, val_index)]

        pid_train_hash = hashlib.sha256(
            str(np.sort(np.unique(data_kfold["pid_train"]).astype(int))).encode()
        ).hexdigest()
        pid_val_hash = hashlib.sha256(
            str(np.sort(np.unique(data_kfold["pid_val"]).astype(int))).encode()
        ).hexdigest()
        pid_test_hash = hashlib.sha256(
            str(np.sort(np.unique(data_kfold["pid_test"]).astype(int))).encode()
        ).hexdigest()
        print(np.unique(data_kfold["pid_test"]).astype(int))
        hash_pids = {
            "pid_train": pid_train_hash,
            "pid_val": pid_val_hash,
            "pid_test": pid_test_hash,
        }

        train_gen = DataGen(
            data_kfold["x_train"],
            data_kfold["y_train"],
            batch_size=imlenet_conf.batch_size,
        )
        val_gen = DataGen(
            data_kfold["x_val"], data_kfold["y_val"], batch_size=imlenet_conf.batch_size
        )
        test_gen = DataGen(
            data_kfold["x_test"],
            data_kfold["y_test"],
            batch_size=imlenet_conf.batch_size,
        )

        model_path, history, wandb = train.train_model(
            train_gen,
            val_gen,
            imlenet_conf=imlenet_conf,
            preprocess_conf=preprocess_conf,
            log_wandb_conf=log_wandb_conf,
            transfer_learning_conf=transfer_learning_conf,
            transfer_learning=run_transfer_learning,
            hash_pids={},
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
