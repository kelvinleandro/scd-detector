import tensorflow as tf
from tensorflow.keras.metrics import Metric


class Specificity(Metric):
    def __init__(self, name="specificity", **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

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

        # Compute TN and FP
        tn = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 0), tf.float32))
        fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), tf.float32))

        self.tn.assign_add(tn)
        self.fp.assign_add(fp)

    def result(self):
        return self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tn.assign(0)
        self.fp.assign(0)
