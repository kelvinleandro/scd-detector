import pickle
import numpy as np


def save_preprocessed_pkls(config=None, dict_to_save={}):
    for each_key in dict_to_save:
        with open(f"data/preprocessed/{each_key}.pkl", "wb") as fout:
            pickle.dump(dict_to_save[each_key].astype(np.float32), fout)

    with open(f"data/preprocessed/last_config.pkl", "wb") as fout:
        pickle.dump(config, fout)


def load_preprocessed_pkls(data_split_result):
    for each_key in data_split_result:
        pkl_file = open(f"data/preprocessed/{each_key}.pkl", "rb")
        pickle_loaded = pickle.load(pkl_file)
        data_split_result[each_key] = pickle_loaded.astype(np.float32)

    return data_split_result


def compare_current_config_with_last_used(
    current_config=None, last_config_pkl="data/preprocessed/last_config.pkl"
):
    try:
        config_file = open(last_config_pkl, "rb")
        last_config = pickle.load(config_file)
    except Exception:
        print("Não foi possível ler arquivo de configuração.")
        return None

    return True if current_config.__dict__ == last_config.__dict__ else False
