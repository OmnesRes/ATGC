import numpy as np
import tensorflow as tf


class ActivationFunctions:
    @staticmethod
    def anlu(x, alpha):
        return (x + ((alpha + (x ** 2)) ** (1 / 2))) / 2

    @staticmethod
    def aisru(x, lower_asymptote, upper_asymptote, lower_alpha, upper_alpha):
        x_2 = x ** 2
        lower_sqrt = (lower_alpha + x_2) ** (1 / 2)
        upper_sqrt = (upper_alpha + x_2) ** (1 / 2)
        return lower_asymptote + ((upper_asymptote - lower_asymptote) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))

    @staticmethod
    def adaptive_bell(x, alpha, beta):
        return 1 / (1 + (((x ** 2) / alpha) ** beta))


class Embed(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, trainable=False, triangular=False):
        super(Embed, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.trainable = trainable
        self.triangular = triangular
        self.embedding_matrix = None
        self.embedding_matrix_padded = None

    def build(self, input_shape):
        if self.triangular:
            self.embedding_matrix = self.add_weight(shape=[self.embedding_dimension, self.embedding_dimension], initializer=tf.constant_initializer(value=np.tri(self.embedding_dimension)), trainable=self.trainable, dtype=tf.float32)
        else:
            self.embedding_matrix = self.add_weight(shape=[self.embedding_dimension, self.embedding_dimension], initializer=tf.keras.initializers.identity(), trainable=self.trainable, dtype=tf.float32)
        self.embedding_matrix_padded = tf.concat([tf.zeros([1, self.embedding_dimension]), self.embedding_matrix], axis=0)

    def call(self, inputs, **kwargs):
        return tf.gather(self.embedding_matrix_padded, inputs, axis=0)


class AISRU(tf.keras.layers.Layer):
    def __init__(self, trainable=True, lower_asymptote=0., upper_asymptote=1., alpha_init=1.):
        super(AISRU, self).__init__()
        self.trainable = trainable
        self.lower_asymptote = lower_asymptote
        self.upper_asymptote = upper_asymptote
        self.alpha_init = alpha_init
        self.lower_alpha = None
        self.upper_alpha = None

    def build(self, input_shape):
        self.lower_alpha = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)
        self.upper_alpha = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)

    def call(self, inputs, **kwargs):
        return ActivationFunctions.aisru(inputs, lower_asymptote=self.lower_asymptote, upper_asymptote=self.upper_asymptote, lower_alpha=tf.exp(self.lower_alpha), upper_alpha=tf.exp(self.upper_alpha))


class ANLU(tf.keras.layers.Layer):
    def __init__(self, trainable=True, alpha_init=0.):
        super(ANLU, self).__init__()
        self.trainable = trainable
        self.alpha_init = alpha_init
        self.alpha = None

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)

    def call(self, inputs, **kwargs):
        return ActivationFunctions.anlu(inputs, alpha=tf.exp(self.alpha))


class StrandWeight(tf.keras.layers.Layer):
    def __init__(self, n_features, trainable=True, strand_init=0.):
        super(StrandWeight, self).__init__()
        self.n_features = n_features
        self.trainable = trainable
        self.strand_init = strand_init
        self.strand_weight = None

    def build(self, input_shape):
        self.strand_weight = self.add_weight(shape=[self.n_features, ], initializer=tf.keras.initializers.constant(self.strand_init), dtype=tf.float32, trainable=self.trainable)

    def call(self, inputs, **kwargs):
        return (ActivationFunctions.aisru(self.strand_weight, lower_asymptote=0., upper_asymptote=1., lower_alpha=1., upper_alpha=1.)[tf.newaxis, tf.newaxis, ...] * (inputs[..., tf.newaxis] - 1)) + 1



