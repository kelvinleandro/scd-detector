"""Configs for building the IMLE-Net model.
"""


class Config:
    """A class used for IMLE-Net configs."""

    def __init__(self) -> None:

        self.signal_len = 128 * 60  # 12800
        self.beat_len = 1280
        self.seed = 42
        self.batch_size = 64  # 4
        self.metric = "val_auc"
        self.input_channels = 1
        self.learning_rate = 0.001
        self.dropout_rate = 0.8
        self.lstm_units = 64  # 256
        self.dense_size = 128  # 512
        self.classes = 2
        self.patience = 20
