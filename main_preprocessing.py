from preprocessing import preprocessing_chagas, status_manegement
from configs import preprocess_config


def run_preprocessing_steps(current_config, compare=True):
    print(current_config.__dict__)

    if compare:
        current_config_already_run = (
            status_manegement.compare_current_config_with_last_used(
                current_config=current_config
            )
        )
    else:
        current_config_already_run = True

    if not current_config_already_run:

        # x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test = (
        #     preprocessing_chagas.preprocess_chagas(
        #         path="data/data_128hz.pkl", current_config=current_config
        #     )
        # )
        x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test = (
            preprocessing_chagas.incremental_preprocess_chagas(
                current_config=current_config
            )
        )
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
        status_manegement.save_preprocessed_pkls(
            config=current_config, dict_to_save=data_split_result
        )
    else:

        data_split_result = {
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

        data_split_result = status_manegement.load_preprocessed_pkls(data_split_result)

    return data_split_result


if __name__ == "__main__":
    conf = preprocess_config.Config()
    x_train, x_val, x_test, y_train, y_val, y_test, pid_train, pid_val, pid_test = (
        preprocessing_chagas.incremental_preprocess_chagas(current_config=conf)
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
    status_manegement.save_preprocessed_pkls(
        config=conf, dict_to_save=data_split_result
    )
    print("deu certo de vdd, salvou nos arquivos.")
