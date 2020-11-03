import tensorflow as tf
from model.KerasLayers import Activations, Ragged, Convolutions
from model.CustomLayers import ANLU, AISRU, Embed, StrandWeight


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
            pos_loc = tf.keras.layers.Dense(units=64, activation=AISRU())(position_input)
            pos_loc = tf.keras.layers.Dense(units=32, activation=ANLU())(pos_loc)
            pos_loc = tf.concat([position_emb(position_bin), pos_loc], axis=-1)
            pos_loc = tf.keras.layers.Dense(units=96, activation=ANLU())(pos_loc)
            fused = tf.concat([chromosome_emb(chromosome_input), pos_loc], axis=-1)
            latent = tf.keras.layers.Dense(units=128, activation=ANLU())(fused)
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
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.convolution_params[index], kernel_size=[1, self.sequence_length], activation=ANLU())
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
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=[1, 1], activation=ANLU())
                # apply conv to each allele
                features[index] = convolutions[index](feature[:, tf.newaxis, :, :])
                # pool over both alleles
                features[index] = tf.reduce_max(features[index], axis=[1, 2])

            fused = tf.concat(features, axis=-1)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation, kernel_regularizer=tf.keras.regularizers.l2())(fused)

            self.model = tf.keras.Model(inputs=[hla_A, hla_B, hla_C], outputs=[fused])



class RaggedModels:

    class MIL:
        def __init__(self, instance_encoders=[], sample_encoders=[], instance_layers=[], sample_layers=[], pooled_layers=[], output_dim=1, output_type='classification', mode='attention', pooling='sum', regularization=.2):
            self.instance_encoders, self.sample_encoders, self.instance_layers, self.sample_layers, self.pooled_layers, self.output_dim, self.output_type, self.mode, self.pooling, self.regularization, = instance_encoders, sample_encoders, instance_layers, sample_layers, pooled_layers, output_dim, output_type, mode, pooling, regularization
            self.model, self.attention_model = None, None
            self.build()

        def build(self):
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]

            if self.instance_encoders != []:
                ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
                # flatten encoders if needed
                ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]

                # based on the design of the input and graph instances can be fused prior to bag aggregation
                ragged_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)

                ragged_hidden = [ragged_fused]
                for i in self.instance_layers:
                    ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_hidden[-1]))

                if self.mode == 'attention':
                    pooling, ragged_attention_weights = Ragged.Attention(pooling=self.pooling, regularization=self.regularization)(ragged_hidden[-1])
                    pooled_hidden = [pooling[:, 0, :]]
                else:
                    if self.pooling == 'mean':
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    else:
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    pooled_hidden = [pooling]

                for i in self.pooled_layers:
                    pooled_hidden = tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(pooled_hidden[-1])

            ##sample level model encodings
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]
            if self.sample_encoders != []:
                sample_encodings = [encoder(sample_input) for sample_input, encoder in zip(sample_inputs, self.sample_encoders)]
                sample_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(sample_encodings)
                if self.instance_encoders != []:
                    fused = tf.concat([pooled_hidden[-1], sample_fused], axis=-1)
                else:
                    fused = sample_fused
            else:
                fused = pooled_hidden[-1]

            fused = tf.keras.layers.Dense(units=32, activation='relu')(fused)
            fused = tf.keras.layers.Dense(units=16, activation='relu')(fused)

            if self.output_type == 'quantiles':
                output_layers = (4, 1)
                point_estimate, lower_bound, upper_bound = list(), list(), list()
                for i in range(len(output_layers)):
                    point_estimate.append(tf.keras.layers.Dense(units=output_layers[i], activation=None if i == (len(output_layers) - 1) else tf.keras.activations.softplus)(fused if i == 0 else point_estimate[-1]))

                for l in [lower_bound, upper_bound]:
                    for i in range(len(output_layers)):
                        l.append(tf.keras.layers.Dense(units=output_layers[i], activation=tf.keras.activations.softplus)(fused if i == 0 else l[-1]))

                output_tensor = tf.keras.activations.softplus(tf.concat([point_estimate[-1] - lower_bound[-1], point_estimate[-1], point_estimate[-1] + upper_bound[-1]], axis=1))

            elif self.output_type == 'survival':
                output_layers = (4, 1)
                pred = list()
                for i in range(len(output_layers)):
                    pred.append(tf.keras.layers.Dense(units=output_layers[i], activation=None if i == (len(output_layers) - 1) else tf.keras.activations.relu)(fused if i == 0 else pred[-1]))

                output_tensor = pred[-1]

            elif self.output_type == 'regression':
                ##assumes log transformed output
                pred = tf.keras.layers.Dense(units=self.output_dim, activation='softplus')(fused)
                output_tensor = tf.math.log(pred + 1)

            else:
                output_tensor = tf.keras.layers.Dense(units=self.output_dim, activation=None)(fused)

            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[output_tensor])
            if self.mode == 'attention':
                self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_attention_weights])



    class losses:
        class CrossEntropy(tf.keras.losses.Loss):
            def __init__(self, name='CE', from_logits=True):
                super(RaggedModels.losses.CrossEntropy, self).__init__(name=name)
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
                super(RaggedModels.losses.QuantileLoss, self).__init__(name=name)
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
                super(RaggedModels.losses.CoxPH, self).__init__(name=name)
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
