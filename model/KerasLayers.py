import tensorflow as tf



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


class Convolutions:

    class DenselyConnectedConv2D(tf.keras.layers.Layer):
        def __init__(self, k, filters, kernel_size):
                super(Convolutions.DenselyConnectedConv2D, self).__init__()
                self.k = k
                self.convs = [tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, activation=tf.keras.activations.relu, padding='same') for _ in range(k)]

        def call(self, input, **kwargs):
            previous_outputs = list()
            for op in self.convs:
                previous_outputs.append(op(tf.concat([input] + previous_outputs, axis=-1)))

            return previous_outputs[-1]


class Ragged:

    class MapFlatValues(tf.keras.layers.Layer):
        def __init__(self, op):
            super(Ragged.MapFlatValues, self).__init__()
            self._supports_ragged_inputs = True
            self.op = op

        def call(self, inputs, **kwargs):
            return tf.ragged.map_flat_values(self.op, inputs)

    class Attention(tf.keras.layers.Layer):
        def __init__(self, pooling='sum', regularization=.2):
            super(Ragged.Attention, self).__init__()
            self.pooling = pooling
            self._supports_ragged_inputs = True
            self.attention_layer_1 = tf.keras.layers.Dense(units=16, activation='relu')
            self.attention_layer_2 = tf.keras.layers.Dense(units=1, activation=Activations.ASU(), activity_regularizer=tf.keras.regularizers.l1(regularization))

        def call(self, inputs, **kwargs):
            attention_weights = tf.ragged.map_flat_values(self.attention_layer_1, inputs)
            attention_weights = tf.ragged.map_flat_values(self.attention_layer_2, attention_weights)

            if self.pooling == 'max':
                pooled = tf.reduce_max(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                    [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                     tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)]), axis=1)
            elif self.pooling == 'mean':
                pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                 [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                  tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)]), axis=1)
                pooled = pooled / tf.expand_dims(tf.reduce_sum(attention_weights, axis=1), axis=-1)

            elif self.pooling == 'logsumexp':
                pooled = tf.math.log(tf.reduce_sum(tf.math.exp(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                    [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                     tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)])), axis=1))

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
                return tf.reduce_sum(tf.reduce_sum(losses * sample_weight, axis=0) / tf.reduce_sum(sample_weight))
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
                return tf.reduce_sum(tf.reduce_sum(losses * sample_weight, axis=0) / tf.reduce_sum(sample_weight) * self.quantiles_weight)
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
            ##sample weights out of order, will have to mask them
            losses = self.call(y_true, y_pred)
            if sample_weight is not None:
                return tf.reduce_sum(losses * sample_weight) / tf.reduce_sum(sample_weight)
            else:
                return tf.reduce_mean(losses)