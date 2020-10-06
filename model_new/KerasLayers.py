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
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)

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
        def __init__(self, attention_heads=1):
            super(Ragged.Attention, self).__init__()
            self._supports_ragged_inputs = True
            self.attention_layer = tf.keras.layers.Dense(units=attention_heads, activation=Activations.ASU())
            self.attention_attention_layer = tf.keras.layers.Dense(units=1, activation=Activations.ASU())

        def call(self, inputs, **kwargs):
            attention_weights = tf.ragged.map_flat_values(self.attention_layer, inputs)
            attention_attention_weights = tf.ragged.map_flat_values(self.attention_attention_layer, attention_weights)
            weighted_attention_sums = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                    [tf.ragged.map_flat_values(tf.expand_dims, attention_attention_weights, axis=2),
                                                                     tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=1)]), axis=1)


            return weighted_attention_sums, attention_weights, attention_attention_weights
