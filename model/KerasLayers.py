import tensorflow as tf
import numpy as np


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



#
#
# class ActivationFunctions:
#
#     @staticmethod
#     def aisru(x, lower_asymptote, upper_asymptote, lower_alpha, upper_alpha):
#         x_2 = x ** 2
#         lower_sqrt = (lower_alpha + x_2) ** (1 / 2)
#         upper_sqrt = (upper_alpha + x_2) ** (1 / 2)
#         return lower_asymptote + ((upper_asymptote - lower_asymptote) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))
#



class Activations:

    class ASU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, lower_asymptote=0., upper_asymptote=1., alpha_init=1., bias_init=None):
            super(Activations.ASU, self).__init__()
            self.trainable = trainable
            self.lower_asymptote = lower_asymptote
            self.upper_asymptote = upper_asymptote
            self.alpha_init = alpha_init
            self.lower_alpha = None
            self.upper_alpha = None
            self.bias_init = bias_init
            self.bias = None

        @staticmethod
        def activation_function(x, lower_asymptote, upper_asymptote, lower_alpha, upper_alpha):
            x_2 = x ** 2
            lower_sqrt = (lower_alpha + x_2) ** (1 / 2)
            upper_sqrt = (upper_alpha + x_2) ** (1 / 2)
            return lower_asymptote + ((upper_asymptote - lower_asymptote) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))

        def build(self, input_shape):
            self.lower_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable)
            self.upper_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable)
            if self.bias_init is not None:
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)


        def call(self, inputs, **kwargs):
            return self.activation_function(inputs + self.bias if self.bias is not None else inputs,
                                            lower_asymptote=self.lower_asymptote, upper_asymptote=self.upper_asymptote,
                                            lower_alpha=tf.exp(self.lower_alpha), upper_alpha=tf.exp(self.upper_alpha))

    class ARU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, alpha_init=0., bias_init=None):
            super(Activations.ARU, self).__init__()
            self.trainable = trainable
            self.alpha_init = alpha_init
            self.alpha = None
            self.bias_init = bias_init
            self.bias = None

        @staticmethod
        def activation_function(x, alpha):
            return (x + ((alpha + (x ** 2)) ** (1 / 2))) / 2

        def build(self, input_shape):
            self.alpha = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)
            if self.bias_init is not None:
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.bias_init), dtype=tf.float32, trainable=True)

        def call(self, inputs, **kwargs):
            return self.activation_function(inputs + self.bias if self.bias is not None else inputs, alpha=tf.exp(self.alpha))



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
        return (Activations.ASU.activation_function(self.strand_weight, lower_asymptote=0., upper_asymptote=1., lower_alpha=1., upper_alpha=1.)[tf.newaxis, tf.newaxis, ...] * (inputs[..., tf.newaxis] - 1)) + 1


class Dense:

    class Gate(tf.keras.layers.Layer):
        def __init__(self, units, activation, bias_kwargs):
            super(Dense.Gate, self).__init__()
            self.units, self.activation, self.bias_kwargs = units, activation, bias_kwargs

        def build(self, input_shape):
            self.bias = self.add_weight(shape=(self.units, ), dtype=tf.float32, **self.bias_kwargs)

        def call(self, inputs, **kwargs):
            return self.activation(inputs + self.bias)


class Ragged:

    class MapFlatValues(tf.keras.layers.Layer):
        def __init__(self, op):
            super(Ragged.MapFlatValues, self).__init__()
            self._supports_ragged_inputs = True
            self.op = op

        def call(self, inputs, **kwargs):
            return tf.ragged.map_flat_values(self.op, inputs)

    class Dense(tf.keras.layers.Layer):
        def __init__(self, units, activation=None):
            super(Ragged.Dense, self).__init__()
            self._supports_ragged_inputs = True
            self.units, self.activation = units, activation
            self.ragged_layer, self.tensor_layer, self.activation_layer = None, None, None

        def build(self, inputs):
            self.ragged_layer = tf.keras.layers.Dense(units=self.units, activation=None, use_bias=False)
            self.tensor_layer = tf.keras.layers.Dense(units=self.units, activation=None, use_bias=False)
            self.activation_layer = Dense.Gate(units=self.units, activation=self.activation, bias_kwargs=dict(initializer=tf.keras.initializers.constant(0), trainable=True))

        def call(self, inputs, **kwargs):
            ragged_dot = tf.ragged.map_flat_values(self.ragged_layer, inputs[0]) + tf.expand_dims(self.tensor_layer(inputs[1]), inputs[0].ragged_rank)
            return tf.ragged.map_flat_values(self.activation_layer, ragged_dot)

    class Attention(tf.keras.layers.Layer):
        def __init__(self, pooling='sum', regularization=.2, layers=[16, ]):
            super(Ragged.Attention, self).__init__()
            self.pooling = pooling
            self._supports_ragged_inputs = True
            self.layers = layers
            self.regularization = regularization
            self.attention_layers = []
            for i in layers:
                self.attention_layers.append(tf.keras.layers.Dense(units=i, activation='relu'))
            self.attention_layers.append(tf.keras.layers.Dense(units=1, activation=Activations.ASU(), activity_regularizer=tf.keras.regularizers.l1(regularization)))

        def call(self, inputs, **kwargs):
            if self.pooling == 'dynamic':
                attention_weights = [inputs[1]]
                for i in self.attention_layers:
                    attention_weights.append(tf.ragged.map_flat_values(i, attention_weights[-1]))
                attention_weights = attention_weights[-1]
                pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                 [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                  tf.ragged.map_flat_values(tf.expand_dims, inputs[0], axis=1)]), axis=1)

            else:
                attention_weights = [inputs]
                for i in self.attention_layers:
                    attention_weights.append(tf.ragged.map_flat_values(i, attention_weights[-1]))
                attention_weights = attention_weights[-1]
                if self.pooling == 'mean':
                    pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                     [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                      tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)]), axis=1)
                    pooled = pooled / tf.expand_dims(tf.reduce_sum(attention_weights, axis=1), axis=-1)
                else:
                    pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                        [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                         tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)]), axis=1)

            return pooled, attention_weights




class Losses:
    class CrossEntropy(tf.keras.losses.Loss):
        def __init__(self, name='CE', from_logits=True):
            super(Losses.CrossEntropy, self).__init__(name=name)
            self.from_logits = from_logits

        def call(self, y_true, y_pred, loss_clip=0.):
            return tf.maximum(tf.keras.losses.CategoricalCrossentropy(reduction='none', from_logits=self.from_logits)(y_true, y_pred) - loss_clip, 0.)

        def __call__(self, y_true, y_pred, sample_weight=None):
            # get sample loss
            losses = self.call(y_true, y_pred)
            # return correct true weighted average if provided sample_weight
            if sample_weight is not None:
                return tf.reduce_sum(losses * sample_weight[:, 0], axis=0) / tf.reduce_sum(sample_weight)
            else:
                return tf.reduce_mean(losses, axis=0)



    class QuantileLoss(tf.keras.losses.Loss):
        def __init__(self, name='quantile_loss', alpha=0.1, weight=0.5):
            super(Losses.QuantileLoss, self).__init__(name=name)
            self.quantiles = tf.constant(((alpha / 2), 0.5, 1 - (alpha / 2)))
            self.quantiles_weight = tf.constant([weight / 2, 1 - weight, weight / 2])

        def call(self, y_true, y_pred):
            # per sample losses across the quantiles
            residual = y_true - y_pred
            return residual * (self.quantiles[tf.newaxis, :] - tf.cast(tf.less(residual, 0.), tf.float32))

        def __call__(self, y_true, y_pred, sample_weight=None):
            # get sample loss
            losses = self.call(y_true, y_pred)
            # return correct true weighted average if provided sample_weight
            if sample_weight is not None:
                return tf.reduce_sum(tf.reduce_sum(losses * sample_weight[:, 0], axis=0) / tf.reduce_sum(sample_weight) * self.quantiles_weight)
            else:
                return tf.reduce_sum(tf.reduce_mean(losses, axis=0) * self.quantiles_weight)

    class CoxPH(tf.keras.losses.Loss):
        def __init__(self, name='coxph', cancers=1):
            super(Losses.CoxPH, self).__init__(name=name)
            self.cancers = cancers

        def call(self, y_true, y_pred):
            total_losses = []
            for cancer in range(self.cancers):
                mask = tf.equal(y_true[:, -1], cancer)
                cancer_y_true = y_true[mask]
                cancer_y_pred = y_pred[mask]
                time_d = tf.cast(cancer_y_true[:, 0][tf.newaxis, :] <= cancer_y_true[:, 0][:, tf.newaxis], tf.float32)
                loss = (tf.math.log(tf.tensordot(time_d, tf.math.exp(cancer_y_pred[:, 0][:, tf.newaxis]), [0, 0])[:, 0]) - cancer_y_pred[:, 0]) * cancer_y_true[:, 1]
                total_losses.append(loss)
            return tf.concat(total_losses, axis=-1)

        def __call__(self, y_true, y_pred, sample_weight=None):
            ##sample weights out of order for multiple cancers, need to reweight based on events. Don't use weighting for now.
            losses = self.call(y_true, y_pred)
            if sample_weight is not None:
                return tf.reduce_sum(losses * sample_weight[:, 0]) / tf.reduce_sum(sample_weight)
            else:
                return tf.reduce_mean(losses)


class Metrics:
    class CrossEntropy(tf.keras.metrics.Metric):
        def __init__(self, name='CE', from_logits=True):
            super(Metrics.CrossEntropy, self).__init__(name=name)
            self.from_logits = from_logits
            self.CE = self.add_weight(name='CE', initializer=tf.keras.initializers.constant(0.))

        def update_state(self, y_true, y_pred, sample_weight=None):
            losses = tf.keras.losses.CategoricalCrossentropy(reduction='none', from_logits=self.from_logits)(y_true, y_pred)
            if sample_weight is not None:
                self.CE.assign(tf.reduce_sum(losses * sample_weight[:, 0]) / tf.reduce_sum(sample_weight))
            else:
                self.CE.assign(tf.reduce_mean(losses))

        def result(self):
            return self.CE

        def reset_states(self):
            self.CE.assign(0)

    class Accuracy(tf.keras.metrics.Metric):
        def __init__(self, name='accuracy'):
            super(Metrics.Accuracy, self).__init__(name=name)
            self.accuracy = self.add_weight(name='accuracy', initializer=tf.keras.initializers.constant(0.))

        def update_state(self, y_true, y_pred, sample_weight=None):
            accuracy = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), dtype=tf.float32)
            if sample_weight is not None:
                self.accuracy.assign(tf.reduce_sum(accuracy * sample_weight[:, 0]) / tf.reduce_sum(sample_weight))
            else:
                self.accuracy.assign(tf.reduce_mean(accuracy))

        def result(self):
            return self.accuracy

        def reset_states(self):
            self.accuracy.assign(0)
