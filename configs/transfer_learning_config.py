
"""Configs for preprocess ECG signals.
"""


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
        self.lstm_mergemode = "concat" #"ave" default "concat"
        #self.path_model = "cinc2017_128hz/weights-6bf54788419a521e857daf0f3fdb13b6df13a0d06799c43aba24d3cabb281557_full_model.h5"
        #self.path_model = "cinc2017_rawsignal/weights-4c65646aba5b79ba75b33940ca32a7fea9513de38897b32d94cab447f0ed8204_full_model.h5"
        self.path_model = "pretraining_shareedb_10s_3classes/weights-00c62c223d3b662359efb8505c8766675068d2714f2375a3f32b0430caac1e88_full_model.h5"
        self.decision_threshold = 0.5
        self.learning_rate = 0.0001
        self.last_activation = "softmax"
        self.optimizer = "adam"
        self.classes = 2
