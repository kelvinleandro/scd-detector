import os
from typing import Tuple

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


def low_pass_filter(signal_array, cutoff_frequency=50, order=4, fs=128):
    b, a = butter(order, cutoff_frequency / (0.5 * fs), btype="low")
    filtered_ecg_low = filtfilt(b, a, signal_array).reshape(1, -1)

    return filtered_ecg_low


def high_pass_filter(signal_array, cutoff_frequency=0.5, order=4, fs=128):
    b, a = butter(order, cutoff_frequency / (0.5 * fs), btype="high")
    filtered_ecg_high = filtfilt(b, a, signal_array).reshape(1, -1)

    return filtered_ecg_high


def powerline_filter(signal_array, fs=128, powerline=50):
    if fs >= 100:
        b = np.ones(int(fs / powerline))
    else:
        b = np.ones(2)
    a = [len(b)]
    y = filtfilt(b, a, signal_array, method="pad")
    return y


def emg_noise_filter(signal_array, fs=128):
    lowcut = 0.5  # Lower cutoff frequency (Hz)
    highcut = 50.0  # Upper cutoff frequency (Hz)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")

    # Apply the bandpass filter to remove noise
    ecg_filtered = filtfilt(b, a, signal_array)

    return ecg_filtered


def median_filter_local(signal_array, fs=128):
    window_size = int(fs * 0.125)

    return median_filter(signal_array, size=7)


def slide_and_cut(
    X,
    Y,
    window_size,
    stride,
    output_pid=False,
    percentage=None,
    filters=[],
    current_config=None,
    peak_method=None,
):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    # window_size = window_size*10
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]

        i_stride = stride

        if not output_pid:
            if tmp_Y == 0:
                i_stride = stride // current_config.stride_numerator_factor
            elif tmp_Y == 1:
                i_stride = stride // (current_config.stride_numerator_factor * 2)

        count_beats_already = 0
        for j in range(0, len(tmp_ts), i_stride):

            if current_config.has_limit:
                if count_beats_already > 50:
                    break
            count_beats_already = count_beats_already + 1
            tmp_array = tmp_ts[j : j + window_size].reshape(1, -1)
            if (np.count_nonzero(tmp_array) / tmp_array.size) < 0.9:
                continue

            if tmp_array.shape[1] != window_size:
                continue

            array_filtered = tmp_array.copy()
            if "lowpass" in filters:
                array_filtered = low_pass_filter(array_filtered)

            if "highpass" in filters:
                array_filtered = high_pass_filter(array_filtered)

            if "powerline" in filters:
                array_filtered = powerline_filter(array_filtered)

            if "emg" in filters:
                array_filtered = emg_noise_filter(array_filtered)

            if "median" in filters:
                array_filtered = median_filter_local(array_filtered)

            if tmp_array.shape[1] == window_size:
                out_X.append(array_filtered)
                out_Y.append(tmp_Y)
                out_pid.append(i)

    return np.concatenate(out_X), np.array(out_Y), np.array(out_pid)


def slide_cut_withpeaks(
    X,
    Y,
    window_size,
    stride,
    output_pid=False,
    peak_method="pantompkins1985",
    percentage=0.4,
    frequency=128,
    filters=[],
    current_config=None,
):
    pid_values = []
    out_X = []
    out_Y = []

    n_sample = X.shape[0]

    for each_sample in range(n_sample):

        tmp_ts = X[each_sample]
        tmp_Y = Y[each_sample]

        i_stride = stride

        if not output_pid:
            if tmp_Y == 0:
                i_stride = stride // current_config.stride_numerator_factor
            elif tmp_Y == 1:
                i_stride = stride // (current_config.stride_numerator_factor * 2)

        array_filtered = tmp_ts.copy()

        if "lowpass" in filters:
            array_filtered = low_pass_filter(array_filtered)

        if "highpass" in filters:
            array_filtered = high_pass_filter(array_filtered)

        if "powerline" in filters:
            array_filtered = powerline_filter(array_filtered)

        if "emg" in filters:
            array_filtered = emg_noise_filter(array_filtered)

        if "median" in filters:
            array_filtered = median_filter_local(array_filtered)

        X_peaks = nk.ecg_peaks(
            array_filtered.reshape(-1),
            sampling_rate=frequency,
            method=current_config.peak_method,
        )[1]["ECG_R_Peaks"]

        indexes_peaks = X_peaks
        count_of_peaks = len(indexes_peaks)
        # print(each_sample, count_of_peaks)
        count_beats_already = 0
        for each_stride_counts in range(0, count_of_peaks, i_stride):

            if current_config.has_limit:
                if count_beats_already > 50:
                    break
            count_beats_already = count_beats_already + 1
            peaks_to_use = indexes_peaks[
                each_stride_counts : each_stride_counts + stride
            ]
            sequence_of_peaks = []
            for each_peak in peaks_to_use:
                if round(2 * percentage * frequency) % 2 == 1:
                    this_beat = np.array(
                        array_filtered[
                            each_peak
                            - round(percentage * frequency) : each_peak
                            + round(percentage * frequency)
                            - 1
                        ]
                    )
                else:
                    this_beat = np.array(
                        array_filtered[
                            each_peak
                            - round(percentage * frequency) : each_peak
                            + round(percentage * frequency)
                        ]
                    )

                if this_beat.shape[0] == round(2 * percentage * frequency):

                    sequence_of_peaks.append(this_beat)

            if len(sequence_of_peaks) != window_size:
                continue

            out_X.append(np.concatenate(sequence_of_peaks).reshape(-1).reshape(1, -1))
            pid_values.append(each_sample)
            out_Y.append(tmp_Y)

    return np.concatenate(out_X), np.array(out_Y), np.array(pid_values)


def read_data_only_split(
    window_size=3000, stride=500, data=None, label=None, current_config=None
):

    all_data_itens = data
    print(all_data_itens.shape)
    all_label = label

    # split train val test
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_data_itens,
        all_label,
        test_size=current_config.test_size,
        random_state=current_config.seed,
        stratify=all_label,
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, test_size=0.5, random_state=current_config.seed, stratify=Y_test
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def read_data(window_size=3000, stride=500, data=None, label=None, current_config=None):
    # read data
    all_data_itens = data
    print(all_data_itens.shape)
    all_label = label

    # split train val test
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_data_itens,
        all_label,
        test_size=current_config.test_size,
        random_state=current_config.seed,
        stratify=all_label,
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, test_size=0.5, random_state=current_config.seed, stratify=Y_test
    )

    if current_config.beat_segmentation:
        slide_method = slide_cut_withpeaks
    else:
        slide_method = slide_and_cut
    # slide and cut
    print("before: ")
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train, pid_train = slide_method(
        X_train,
        Y_train,
        window_size=window_size,
        stride=stride,
        percentage=current_config.beat_percentage,
        filters=current_config.filters,
        current_config=current_config,
    )
    X_val, Y_val, pid_val = slide_method(
        X_val,
        Y_val,
        window_size=window_size,
        stride=stride,
        percentage=current_config.beat_percentage,
        output_pid=True,
        filters=current_config.filters,
        current_config=current_config,
    )
    X_test, Y_test, pid_test = slide_method(
        X_test,
        Y_test,
        window_size=window_size,
        stride=stride,
        percentage=current_config.beat_percentage,
        output_pid=True,
        filters=current_config.filters,
        current_config=current_config,
    )
    print("after: ")
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]
    pid_train = pid_train.ravel().reshape(-1, 1)[shuffle_pid].ravel()

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        Y_train,
        Y_val,
        Y_test,
        pid_train,
        pid_val,
        pid_test,
    )


def normalize(x: np.array = None, type_norm: str = "standard"):
    x = x.reshape(x.shape[0], x.shape[2])

    if type_norm == "minmax":
        x = (x - np.min(x, axis=1).reshape(-1, 1)) / (
            np.max(x, axis=1).reshape(-1, 1) - np.min(x, axis=1).reshape(-1, 1)
        )

    elif type_norm == "minmax_minus1":
        x = (x - np.min(x, axis=1).reshape(-1, 1)) / (
            np.max(x, axis=1).reshape(-1, 1) - np.min(x, axis=1).reshape(-1, 1)
        ) * 2 - 1

    elif type_norm == "standard":
        x = (x - np.mean(x, axis=1).reshape(-1, 1)) / (
            np.std(x, axis=1).reshape(-1, 1) + 1e-10
        )

    else:
        x = x

    return x


def preprocess_chagas(
    path: str = "data/data_128hz.pkl",
    class_threshold: float = 0.5,
    current_config: object = None,
) -> Tuple[np.array]:
    print("Loading pickle...")

    with open(path, "rb") as fin:
        res = pickle.load(fin)

    # res["data"] = res["data"].astype(np.float32)
    if current_config.label_type == "death_label":
        ef_label = res["death_label"].astype(np.int16)
    else:
        ef_label = (res["ejection_fraction"] < current_config.class_threshold).astype(
            np.int16
        )

    if current_config.beat_segmentation:
        time = current_config.seconds
    else:
        time = current_config.seconds * current_config.fs
    print("Reading data and spliting...")
    x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test = (
        read_data(
            window_size=time,
            stride=time,
            data=res["data"],
            label=ef_label,
            current_config=current_config,
        )
    )

    x_train = normalize(x_train, type_norm=current_config.type_norm)
    x_val = normalize(x_val, type_norm=current_config.type_norm)
    x_test = normalize(x_test, type_norm=current_config.type_norm)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    if current_config.hot_encode:
        multi_bin = MultiLabelBinarizer()

        y_train = multi_bin.fit_transform(y_train.reshape(-1, 1))
        y_val = multi_bin.transform(y_val.reshape(-1, 1))
        y_test = multi_bin.transform(y_test.reshape(-1, 1))

    return x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test


######################################


def save_unprocessed(
    pkl_path: str = "data/data_128hz.pkl",
    output_path: str = "unprocessed",
    current_config: object = None,
):
    with open(pkl_path, "rb") as fin:
        res = pickle.load(fin)

    os.makedirs(f"data/{output_path}", exist_ok=True)

    if current_config.label_type == "death_label":
        ef_label = res["death_label"].astype(np.int16)
    else:
        ef_label = (res["ejection_fraction"] < current_config.class_threshold).astype(
            np.int16
        )

    X_train, X_val, X_test, Y_train, Y_val, Y_test = read_data_only_split(
        data=res["data"], label=ef_label, current_config=current_config
    )

    data_split_result = {
        "x_train": X_train,
        "x_val": X_val,
        "x_test": X_test,
        "y_train": Y_train,
        "y_val": Y_val,
        "y_test": Y_test,
    }

    for each_key in data_split_result:
        with open(f"data/{output_path}/{each_key}.pkl", "wb") as fout:
            pickle.dump(data_split_result[each_key].astype(object), fout)


def incremental_read_data(
    window_size=3000, stride=500, unprocessed_path=None, current_config=None
):
    if current_config.beat_segmentation:
        slide_method = slide_cut_withpeaks
    else:
        slide_method = slide_and_cut

    with open(os.path.join(unprocessed_path, "x_train.pkl"), "rb") as fin:
        X_train_og = pickle.load(fin)
    with open(os.path.join(unprocessed_path, "y_train.pkl"), "rb") as fin:
        Y_train_og = pickle.load(fin)

    print("before (train): ")
    print(Counter(Y_train_og))
    X_train, Y_train, pid_train = slide_method(
        X_train_og,
        Y_train_og,
        window_size=window_size,
        stride=stride,
        percentage=current_config.beat_percentage,
        filters=current_config.filters,
        current_config=current_config,
        output_pid=False,
    )
    del X_train_og, Y_train_og
    print("after (train): ")
    print(Counter(Y_train))

    with open(os.path.join(unprocessed_path, "x_val.pkl"), "rb") as fin:
        X_val_og = pickle.load(fin)
    with open(os.path.join(unprocessed_path, "y_val.pkl"), "rb") as fin:
        Y_val_og = pickle.load(fin)

    print("before (val): ")
    print(Counter(Y_val_og))
    X_val, Y_val, pid_val = slide_method(
        X_val_og,
        Y_val_og,
        window_size=window_size,
        stride=stride,
        percentage=current_config.beat_percentage,
        output_pid=False,
        filters=current_config.filters,
        current_config=current_config,
    )
    del X_val_og, Y_val_og
    print("after (val): ")
    print(Counter(Y_val))

    with open(os.path.join(unprocessed_path, "x_test.pkl"), "rb") as fin:
        X_test_og = pickle.load(fin)
    with open(os.path.join(unprocessed_path, "y_test.pkl"), "rb") as fin:
        Y_test_og = pickle.load(fin)

    print("before (test): ")
    print(Counter(Y_test_og))
    X_test, Y_test, pid_test = slide_method(
        X_test_og,
        Y_test_og,
        window_size=window_size,
        stride=stride,
        percentage=current_config.beat_percentage,
        output_pid=False,
        filters=current_config.filters,
        current_config=current_config,
    )
    del X_test_og, Y_test_og
    print("after (test): ")
    print(Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]
    pid_train = pid_train.ravel().reshape(-1, 1)[shuffle_pid].ravel()

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        Y_train,
        Y_val,
        Y_test,
        pid_train,
        pid_val,
        pid_test,
    )


def incremental_preprocess_chagas(
    unprocessed_path: str = "data/unprocessed",
    class_threshold: float = 0.5,
    current_config: object = None,
):
    if current_config.beat_segmentation:
        time = current_config.seconds
    else:
        time = current_config.seconds * current_config.fs
    print("Reading data and spliting...")
    x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test = (
        incremental_read_data(
            window_size=time,
            stride=time,
            unprocessed_path=unprocessed_path,
            current_config=current_config,
        )
    )

    print("normalizing...")
    x_train = normalize(x_train, type_norm=current_config.type_norm)
    x_val = normalize(x_val, type_norm=current_config.type_norm)
    x_test = normalize(x_test, type_norm=current_config.type_norm)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    if current_config.hot_encode:
        multi_bin = MultiLabelBinarizer()

        y_train = multi_bin.fit_transform(y_train.reshape(-1, 1))
        y_val = multi_bin.transform(y_val.reshape(-1, 1))
        y_test = multi_bin.transform(y_test.reshape(-1, 1))

    return x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test
