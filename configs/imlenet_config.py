"""Configs for building the IMLE-Net model."""


class Config:
    """A class used for IMLE-Net configs."""

    def __init__(self) -> None:
        """
        Parameters
        ----------
        signal_len: int
            The length of the input ECG signal (Time in secs * Sampling rate).
        input_channels: int
            The number of input channels of an ECG signal.
        beat_len: int
            The length of the segmented ECG beat (Time in secs * Sampling rate).
        kernel_size: int
            The kernel size of the 1D-convolution kernel.
        num_blocks_list: List[int]
            The number of residual blocks in the model.
        lstm_units: int
            The number of units in the LSTM layer.
        start_filters: int
            The number of filters at the start of the 1D-convolution layer.
        classes: int
        The number of classes in the output layer.

        """

        self.signal_len = 60 * 128  # 640
        self.seed = 42
        self.batch_size = 64
        self.metric = "val_f1_score"
        self.att_layer = "Additive"
        self.batch_position = "before"
        self.input_channels = 1
        self.beat_len = 128
        self.kernel_size = 16
        self.learning_rate = 0.001
        self.dropout_rate = 0.8
        self.num_blocks_list = [2, 2, 2]
        self.start_filters = 16
        self.lstm_units = self.start_filters * (2 ** (len(self.num_blocks_list) - 2))
        self.classes = 2
        self.patience = 10
