"""Configs for wandb.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """A class used for Wandb configs."""

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

        self.api_key_wandb = os.getenv("WANDB_API_KEY")
        self.project = os.getenv("WANDB_PROJECT")
        self.entity = os.getenv("WANDB_ENTITY")
