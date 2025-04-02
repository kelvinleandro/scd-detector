"""Configs for preprocess ECG signals."""


class Config:
    """A class used for Preprocess Config."""

    def __init__(self) -> None:
        """
        Parameters
        ----------
        seconds: int
            The number of seconds to slide the signal
        fs: int
            The frequency of signal
        filters: List[str]
            The list of filters applied to signals.


        """

        self.server = "chagas1"
        self.execution = ""
        self.seed = 42
        self.val_factor_kfold = 5
        self.test_factor_kfold = 5
        self.att_layer = "Additive"
        self.dense_layers = [64]
        self.has_lstm = True
        self.has_dropout = True
        self.fine_tuning = False
        self.retrain = False
        self.dropout_rate = 0.5
        self.level_to_cut = -2
        self.lstm_units = 1024
        self.transfer_seconds = 10
        self.lstm_mergemode = "concat"  # "ave" default "concat"
        self.path_model = "scd-detector/weights-6de941130f41c259619f65cec66ac5ae9231ff21fcef12440247e9a4b946f643_full_model.keras"
        # self.path_model = "scd-detector/weights-9b14b3044192ec72660f12ac9b4f82fd037bb9fa9eee27984c2f07520cac6b6a_full_model.keras"
        # self.path_model = "scd-detector/weights-63383cf6db911fadc8ddb783dc67319902b17865fe6bd4970e4ad5795f8a3222_full_model.keras"
        # self.path_model = "scd-detector/weights-e76cae1b1d0bbe4264c7feaa046339debf46cf427df1bb01ecd3a34b3c2730b5_full_model.keras"
        # self.path_model = "scd-detector/weights-40f9499252e90b5a3339fec339cbd41b25f7711cde09f140fc7a980b3044e1bc_full_model.keras"
        # self.path_model = "scd-detector/weights-40f07c781cc98e3f36121ced140b89b6637bf2cb0c917d28683f0dbc0c7fa92c_full_model.keras"

        self.decision_threshold = 0.5
        self.learning_rate = 0.0001
        self.last_activation = "softmax"
        self.loss = "categorical_crossentropy"
        self.optimizer = "adam"
        self.classes = 2
