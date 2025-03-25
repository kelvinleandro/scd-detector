import tensorflow as tf
from tensorflow.keras.metrics import Metric


class GeometricMean(Metric):
    def __init__(self, name="geometric_mean", **kwargs):
        super(GeometricMean, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot to categorical labels if needed
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
        else:
            y_pred = tf.cast(
                y_pred > 0.5, tf.float32
            )  # Convert probabilities to binary values

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute TP, FP, FN, TN
        tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
        fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), tf.float32))
        fn = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), tf.float32))
        tn = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 0), tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.tn.assign_add(tn)

    def result(self):
        # Compute Recall and Specificity
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        specificity = self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())

        # Compute Geometric Mean
        g_mean = tf.sqrt(recall * specificity)
        return g_mean

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
        self.tn.assign(0)
