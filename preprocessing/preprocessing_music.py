from .preprocessing_chagas import (
    save_unprocessed,
    incremental_preprocess_chagas,
)
from .status_manegement import save_preprocessed_pkls
from configs.preprocess_config import Config

if __name__ == "__main__":
    conf = Config()
    # save unprocessed data (just splitted into train-val-test)
    # save_unprocessed("data/music_128hz_176.pkl", "music_unprocessed", conf)

    # pre processing data
    x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test = (
        incremental_preprocess_chagas(
            unprocessed_path="data/music_unprocessed", current_config=conf
        )
    )
    print("Deu certo.")
    data_split_result = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pid_train": pid_train,
        "pid_val": pid_val,
        "pid_test": pid_test,
    }
    save_preprocessed_pkls(
        config=conf,
        dict_to_save=data_split_result,
        output_folder="music_preprocessed_10s_standard",
    )
    print("deu certo de vdd, salvou nos arquivos.")
