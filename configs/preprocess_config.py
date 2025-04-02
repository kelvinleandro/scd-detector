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

        self.seed = 42
        self.seconds = 60
        self.fs = 128
        self.label_type = "death_label"  # "ejection_fraction"
        self.class_threshold = 0.50
        self.hot_encode = True
        self.test_size = 0.3
        self.stride_numerator_factor = 1
        self.type_norm = "standard"
        self.filters = [
            "powerline",
            "emg",
        ]  # ["lowpass", "highpass", "powerline", "emg"]
        self.peak_method = "neurokit"
        self.has_limit = True
        self.num_limit = 50  # int | None
        self.beat_segmentation = True
        self.beat_percentage = 0.5
        self.kfolds = False
        self.fold = None
