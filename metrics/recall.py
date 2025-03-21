import tensorflow as tf
from tensorflow.keras.metrics import Metric


class Recall(Metric):
    def __init__(self, name="recall", **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # If one-hot encoded, convert to class indices
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
        else:
            y_pred = tf.cast(
                y_pred > 0.5, tf.float32
            )  # Convert probabilities to binary values

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute TP and FN
        tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
        fn = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), tf.float32))

        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0)
        self.fn.assign(0)
