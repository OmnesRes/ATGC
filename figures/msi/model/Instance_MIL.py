import tensorflow as tf
from model.KerasLayers import Activations, Ragged, Embed, StrandWeight

class InstanceModels:

    class VariantPositionBin:
        def __init__(self, chromosome_embedding_dimension, position_embedding_dimension, default_activation=tf.keras.activations.relu):
            self.chromosome_embedding_dimension = chromosome_embedding_dimension
            self.position_embedding_dimension = position_embedding_dimension
            self.default_activation = default_activation
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            position_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            position_bin = tf.keras.layers.Input(shape=(), dtype=tf.int32)
            chromosome_input = tf.keras.layers.Input(shape=(), dtype=tf.int32)
            chromosome_emb = Embed(embedding_dimension=self.chromosome_embedding_dimension, trainable=False)
            position_emb = Embed(embedding_dimension=self.position_embedding_dimension, trainable=False, triangular=False)
            pos_loc = tf.keras.layers.Dense(units=64, activation=Activations.ASU())(position_input)
            pos_loc = tf.keras.layers.Dense(units=32, activation=Activations.ARU())(pos_loc)
            pos_loc = tf.concat([position_emb(position_bin), pos_loc], axis=-1)
            pos_loc = tf.keras.layers.Dense(units=96, activation=Activations.ARU())(pos_loc)
            fused = tf.concat([chromosome_emb(chromosome_input), pos_loc], axis=-1)
            latent = tf.keras.layers.Dense(units=128, activation=Activations.ARU())(fused)
            self.model = tf.keras.Model(inputs=[position_input, position_bin, chromosome_input], outputs=[latent])


    class VariantSequence:
        def __init__(self, sequence_length, sequence_embedding_dimension, n_strands, convolution_params, fusion_dimension=64, default_activation=tf.keras.activations.relu, use_frame=False):
            self.sequence_length = sequence_length
            self.sequence_embedding_dimension = sequence_embedding_dimension
            self.convolution_params = convolution_params
            self.default_activation = default_activation
            self.n_strands = n_strands
            self.use_frame = use_frame
            self.fusion_dimension = fusion_dimension
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            five_p = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            three_p = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            ref = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            alt = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            strand = tf.keras.layers.Input(shape=(self.n_strands,), dtype=tf.float32)

            # layers of convolution for sequence feature extraction based on conv_params
            features = [[]] * 4
            convolutions = [[]] * 4
            nucleotide_emb = Embed(embedding_dimension=4, trainable=False)
            for index, feature in enumerate([five_p, three_p, ref, alt]):
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.convolution_params[index], kernel_size=[1, self.sequence_length], activation=Activations.ARU())
                # apply conv to forward and reverse
                features[index] = tf.stack([convolutions[index](nucleotide_emb(feature)[:, tf.newaxis, :, i, :]) for i in range(self.n_strands)], axis=3)
                # pool over any remaining positions
                features[index] = tf.reduce_max(features[index], axis=[1, 2])

            fused = tf.concat(features, axis=2)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation, kernel_regularizer=tf.keras.regularizers.l2())(fused)
            fused = tf.reduce_max(StrandWeight(trainable=True, n_features=fused.shape[2])(strand) * fused, axis=1)

            if self.use_frame:
                cds = tf.keras.layers.Input(shape=(3,), dtype=tf.float32)
                frame = tf.concat([strand, cds], axis=-1)
                frame = tf.keras.layers.Dense(units=6, activation=self.default_activation)(frame)
                fused = tf.concat([fused, frame], axis=-1)
                self.model = tf.keras.Model(inputs=[five_p, three_p, ref, alt, strand, cds], outputs=[fused])
            else:
                self.model = tf.keras.Model(inputs=[five_p, three_p, ref, alt, strand], outputs=[fused])

    class PassThrough:
        def __init__(self, shape=None):
            self.shape = shape
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(shape=self.shape, dtype=tf.float32)
            self.model = tf.keras.Model(inputs=[input], outputs=[input])


class SampleModels:
    class PassThrough:
        def __init__(self, shape=None):
            self.shape = shape
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            self.model = tf.keras.Model(inputs=[input], outputs=[input])

    class HLA:
        def __init__(self, filters=8, latent_dim=4, fusion_dimension=64, default_activation=tf.keras.activations.relu):
            self.default_activation = default_activation
            self.fusion_dimension = fusion_dimension
            self.filters = filters
            self.latent_dim = latent_dim
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            hla_A = tf.keras.layers.Input(shape=(2, self.latent_dim), dtype=tf.float32)
            hla_B = tf.keras.layers.Input(shape=(2, self.latent_dim), dtype=tf.float32)
            hla_C = tf.keras.layers.Input(shape=(2, self.latent_dim), dtype=tf.float32)

            # layers of convolution for sequence feature extraction based on conv_params
            features = [[]] * 3
            convolutions = [[]] * 3
            for index, feature in enumerate([hla_A, hla_B, hla_C]):
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=[1, 1], activation=Activations.ARU())
                # apply conv to each allele
                features[index] = convolutions[index](feature[:, tf.newaxis, :, :])
                # pool over both alleles
                features[index] = tf.reduce_max(features[index], axis=[1, 2])

            fused = tf.concat(features, axis=-1)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation, kernel_regularizer=tf.keras.regularizers.l2())(fused)

            self.model = tf.keras.Model(inputs=[hla_A, hla_B, hla_C], outputs=[fused])


class RaggedModels:

    class MIL:
        def __init__(self, instance_encoders=[], sample_encoders=[], output_dim=1, attention_heads=1, output_type='classification', pooling='sum', instance_activation=None, instance_layers=[], custom_layers=[16, 32, 16, 1]):
            self.instance_encoders, self.sample_encoders, self.output_dim, self.attention_heads, self.output_type, self.pooling, self.instance_activation, self.instance_layers, self.custom_layers = instance_encoders, sample_encoders, output_dim, attention_heads, output_type, pooling, instance_activation, instance_layers, custom_layers
            self.model, self.attention_model = None, None
            self.build()

        def build(self):
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]

            if self.instance_encoders != []:
                ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
                # flatten encoders if needed
                ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]

                # based on the design of the input and graph instances can be fused prior to bag aggregation
                ragged_fused = [tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)]

                for i in self.instance_layers:
                    ragged_fused.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_fused[-1]))

                ragged_hidden = Ragged.MapFlatValues(tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu))(ragged_fused[-1])
                ragged_hidden = Ragged.MapFlatValues(tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu))(ragged_hidden)

                if self.output_type == 'regression':
                    instance_predictions = Ragged.MapFlatValues(tf.keras.layers.Dense(units=self.output_dim,
                                                                                      activation='softplus',
                                                                                      use_bias=True))(ragged_hidden)
                else:
                    instance_predictions = Ragged.MapFlatValues(tf.keras.layers.Dense(units=self.output_dim,
                                                                                      activation=self.instance_activation,
                                                                                      use_bias=True))(ragged_hidden)
                ##MIL pooling
                if self.pooling == 'mean':
                    pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=instance_predictions.ragged_rank))(instance_predictions)
                else:
                    pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=instance_predictions.ragged_rank))(instance_predictions)


            ##sample level model encodings (not applicable to instance MIL)
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]

            if self.output_type == 'quantiles':
                ##quantiles with instance model needs to be done before aggregation
                pass
            elif self.output_type == 'regression':
                ##assumes log transformed output
                output_tensor = tf.math.log(pooling + 1)
            elif self.output_type == 'custom':
                output = [pooling]
                for i in self.custom_layers:
                    output.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(output[-1]))
                output_tensor = output[-1]

            else:
                output_tensor = pooling

            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[output_tensor])
            self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[instance_predictions])