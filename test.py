from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
from collections import Counter
from models import imlenet


def test_baseline_model(
    model_path,
    test_gen,
    data_split_result,
    model_conf=None,
    log_wandb=None,
    log_wandb_conf=None,
):

    print("Test Data Evaluating...")
    print(model_path)

    n_classes = model_conf.classes

    loaded_model = load_model(
        f"{log_wandb_conf.project}/{model_path}_full_model.keras", compile=False
    )

    print("Passed load model")

    all_pred_prob = []
    prog_iter_test = tqdm(test_gen, desc="validating", leave=False, disable=True)
    print(f"Before predicting. Len = {len(test_gen)}")
    for batch_idx, batch in enumerate(prog_iter_test):
        if batch_idx >= len(test_gen):
            print("Forcing stop: Exceeded dataset length")
            break
        print(f"Processing batch {batch_idx+1} / {len(test_gen)}")
        input_x, input_y = batch[0], batch[1]
        pred = loaded_model(input_x)
        all_pred_prob.append(pred)
    print("After predicting")
    all_pred_prob = np.concatenate(all_pred_prob)
    if n_classes > 1:
        all_pred = np.argmax(all_pred_prob, axis=1)
    else:
        all_pred = all_pred_prob

    final_pred = []
    final_gt = []
    final_pred_sum = []
    final_pred_avg = []

    for i_pid in np.unique(data_split_result["pid_test"]):
        # print(data_split_result["pid_test"], data_split_result["pid_test"].shape)

        # pelo menos quando n_classes > 1, n tem diferenca
        tmp_pred = (
            (all_pred[data_split_result["pid_test"] == i_pid] > 0.9).astype(int).ravel()
        )

        if n_classes > 1:
            # one hot p/ binario dos valores verdadeiros
            tmp_gt = np.argmax(
                data_split_result["y_test"][data_split_result["pid_test"] == i_pid],
                axis=1,
            )
            # soma das probabilidades de cada classe e pega o maior
            final_pred_sum.append(
                np.argmax(
                    np.sum(
                        all_pred_prob[data_split_result["pid_test"] == i_pid], axis=0
                    )
                )
            )
            # media das probabilidades de cada classe e pega o maior
            final_pred_avg.append(
                np.argmax(
                    np.mean(
                        all_pred_prob[data_split_result["pid_test"] == i_pid], axis=0
                    )
                )
            )
        else:
            tmp_gt = data_split_result["y_test"][data_split_result["pid_test"] == i_pid]

        final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
        final_gt.append(Counter(tmp_gt).most_common(1)[0][0])

    tmp_report = classification_report(final_gt, final_pred, output_dict=True)
    print(tmp_report)

    log_wandb.log(
        {
            "classification_report_counter": tmp_report,
            "confusion_matrix_counter": confusion_matrix(final_gt, final_pred).tolist(),
        }
    )
    if n_classes > 1:
        tmp_report = classification_report(final_gt, final_pred_sum, output_dict=True)
        log_wandb.log(
            {
                "classification_report_sum": tmp_report,
                "confusion_matrix_sum": confusion_matrix(
                    final_gt, final_pred_sum
                ).tolist(),
            }
        )

        tmp_report = classification_report(final_gt, final_pred_avg, output_dict=True)
        log_wandb.log(
            {
                "classification_report_avg": tmp_report,
                "confusion_matrix_avg": confusion_matrix(
                    final_gt, final_pred_avg
                ).tolist(),
            }
        )

    log_wandb.finish()


def test_imlenet_model(
    model_path,
    test_gen,
    data_split_result,
    imle_conf=None,
    log_wandb=None,
    log_wandb_conf=None,
    transfer_learning_conf=None,
    transfer_learning=False,
):
    print("Test Data Evaluating...")
    print(model_path)

    if transfer_learning:
        n_classes = transfer_learning_conf.classes
    else:
        n_classes = imle_conf.classes
    if imle_conf.att_layer == "Additive":
        loaded_model = load_model(
            f"{log_wandb_conf.project}/{model_path}_full_model.keras",
            custom_objects={"AdditiveAttention": imlenet.AdditiveAttention},
            compile=False,
        )
    elif imle_conf.att_layer == "DotProduct":
        loaded_model = load_model(
            f"{log_wandb_conf.project}/{model_path}_full_model.keras",
            custom_objects={"DotProductAttention": imlenet.DotProductAttention},
            compile=False,
        )
    elif imle_conf.att_layer == "MultiHead":
        loaded_model = load_model(
            f"{log_wandb_conf.project}/{model_path}_full_model.keras",
            custom_objects={"MultiHeadAttention": imlenet.MultiHeadAttention},
            compile=False,
        )
    else:
        loaded_model = load_model(
            f"{log_wandb_conf.project}/{model_path}_full_model.keras", compile=False
        )

    all_pred_prob = []
    prog_iter_test = tqdm(test_gen, desc="validating", leave=False, disable=True)
    for batch_idx, batch in enumerate(prog_iter_test):
        if batch_idx >= len(test_gen):
            print("Forcing stop: Exceeded dataset length")
            break
        input_x, input_y = batch[0], batch[1]
        pred = loaded_model(input_x)
        all_pred_prob.append(pred)
    all_pred_prob = np.concatenate(all_pred_prob)
    if n_classes > 1:
        all_pred = np.argmax(all_pred_prob, axis=1)
    else:
        all_pred = all_pred_prob

    final_pred = []
    final_gt = []
    final_pred_sum = []
    final_pred_avg = []

    for i_pid in np.unique(data_split_result["pid_test"]):
        # print(
        #     data_split_result["pid_test"],
        #     data_split_result["pid_test"].shape,
        #     i_pid,
        #     all_pred.shape,
        # )
        tmp_pred = (
            (
                all_pred[data_split_result["pid_test"] == i_pid]
                > transfer_learning_conf.decision_threshold
            )
            .astype(int)
            .ravel()
        )

        if n_classes > 1:
            tmp_gt = np.argmax(
                data_split_result["y_test"][data_split_result["pid_test"] == i_pid],
                axis=1,
            )
            final_pred_sum.append(
                np.argmax(
                    np.sum(
                        all_pred_prob[data_split_result["pid_test"] == i_pid], axis=0
                    )
                )
            )
            final_pred_avg.append(
                np.argmax(
                    np.mean(
                        all_pred_prob[data_split_result["pid_test"] == i_pid], axis=0
                    )
                )
            )
        else:
            tmp_gt = data_split_result["y_test"][data_split_result["pid_test"] == i_pid]

        final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
        final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
    # print(final_gt)
    # print(final_pred)
    tmp_report = classification_report(final_gt, final_pred, output_dict=True)

    log_wandb.log(
        {
            "classification_report_counter": tmp_report,
            "confusion_matrix_counter": confusion_matrix(final_gt, final_pred).tolist(),
        }
    )
    if n_classes > 1:
        tmp_report = classification_report(final_gt, final_pred_sum, output_dict=True)
        log_wandb.log(
            {
                "classification_report_sum": tmp_report,
                "confusion_matrix_sum": confusion_matrix(
                    final_gt, final_pred_sum
                ).tolist(),
            }
        )

        tmp_report = classification_report(final_gt, final_pred_avg, output_dict=True)
        log_wandb.log(
            {
                "classification_report_avg": tmp_report,
                "confusion_matrix_avg": confusion_matrix(
                    final_gt, final_pred_avg
                ).tolist(),
            }
        )

    log_wandb.finish()
