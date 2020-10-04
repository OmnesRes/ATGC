import tensorflow as tf
from model.CustomLayers import ANLU, AISRU, Embed, StrandWeight
import re

class InputFeatures:
    @staticmethod
    def generate_input_tensors(feature, batch_singleton=False):
        # input_tensors is a dict of tensors matching naming and dimensionality of input data dict
        input_tensors = dict()
        for key in feature.inputs.keys():
            if batch_singleton:
                input_tensors[key] = tf.keras.Input(shape=(None,) + feature.inputs[key].shape[1:], batch_size=1, dtype=tf.float32, name=key + '_mil')
            else:
                input_tensors[key] = tf.keras.Input(shape=feature.inputs[key].shape[1:], dtype=tf.float32, name=key)

        return input_tensors

    class VariantSequence:
        def __init__(self, sequence_length, sequence_embedding_dimension, n_strands, convolution_params, data, fusion_dimension=64, name='variant_sequence', default_activation=tf.keras.activations.relu, use_frame=False):
            self.name = name
            self.sequence_length = sequence_length
            self.sequence_embedding_dimension = sequence_embedding_dimension
            self.convolution_params = convolution_params
            self.inputs = data
            self.default_activation = default_activation
            self.n_strands = n_strands
            self.use_frame = use_frame
            self.fusion_dimension = fusion_dimension

        def encode(self, input_tensors):
            # layers of convolution for sequence feature extraction based on conv_params
            features = [[]] * 4
            convolutions = [[]] * 4
            nucleotide_emb = Embed(embedding_dimension=4, trainable=False)
            for index, feature in enumerate(['5p', '3p', 'ref', 'alt']):
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.convolution_params[index], kernel_size=[1, self.sequence_length], activation=ANLU())
                # apply conv to forward and reverse
                features[index] = tf.stack([convolutions[index](nucleotide_emb(tf.cast(input_tensors[feature], dtype=tf.int32))[:, tf.newaxis, :, i, :]) for i in range(self.n_strands)], axis=3)
                # pool over any remaining positions
                features[index] = tf.reduce_max(features[index], axis=[1, 2])

            fused = tf.concat(features, axis=2)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation, kernel_regularizer=tf.keras.regularizers.l2())(fused)
            fused = tf.reduce_max(StrandWeight(trainable=True, n_features=fused.shape[2])(input_tensors['strand']) * fused, axis=1)

            if self.use_frame:
                frame = tf.concat([input_tensors['strand'], input_tensors['cds']], axis=-1)
                frame = tf.keras.layers.Dense(units=6, activation=self.default_activation)(frame)
                return tf.concat([fused, frame], axis=-1)
            else:
                return fused

        def decode(self, latent_tensor):
            pass

        def reconstruction_loss(self, y_true, y_pred):
            pass

    class VariantPositionBin:
        def __init__(self, chromosome_embedding_dimension, position_embedding_dimension, data, name='variant_position', default_activation=tf.keras.activations.relu):
            self.name = name
            self.chromosome_embedding_dimension = chromosome_embedding_dimension
            self.position_embedding_dimension = position_embedding_dimension
            self.inputs = data
            self.default_activation = default_activation

        def encode(self, input_tensor):
            chromosome_emb = Embed(embedding_dimension=self.chromosome_embedding_dimension, trainable=False)
            position_emb = Embed(embedding_dimension=self.position_embedding_dimension , trainable=False, triangular=False)
            pos_loc = tf.keras.layers.Dense(units=64, activation=AISRU())(input_tensor['position_loc'])
            pos_loc = tf.keras.layers.Dense(units=32, activation=ANLU())(pos_loc)
            pos_bin_loc = tf.concat([position_emb(tf.cast(input_tensor['position_bin'], dtype=tf.int32)), pos_loc], axis=1)
            pos_bin_loc = tf.keras.layers.Dense(units=96, activation=ANLU())(pos_bin_loc)
            fused = tf.concat([chromosome_emb(tf.cast(input_tensor['chromosome'], tf.int32)), pos_bin_loc], axis=1)
            fused = tf.keras.layers.Dense(units=128, activation=ANLU())(fused)
            return fused

        def decode(self, latent_tensor):
            pass

        def reconstruction_loss(self, y_true, y_pred):
            pass


    class OnesLike:
        def __init__(self, data, name='ones_like'):
            self.name = name
            self.inputs = data

        def encode(self, input_tensor):
            pos = input_tensor['position']
            return tf.ones_like(pos)

        def decode(self, latent_tensor):
            pass

        def reconstruction_loss(self, y_true, y_pred):
            pass


class ATGC:
    def __init__(self, features, latent_dimension=64, sample_dimension=64, fusion_dimension=64, sample_features=(), aggregation_dimension=64):
        self.features = features
        self.latent_dimension = latent_dimension
        self.fusion_dimension = fusion_dimension
        self.sample_dimension = sample_dimension
        self.sample_features = sample_features
        self.aggregation_dimension = aggregation_dimension
        self.encoder_model = None
        self.decoder_model = None
        self.vae_model = None
        self.mil_model = None

    @staticmethod
    def reparameterize(z_mean, z_log_var):
        # reparameterize for z_latent
        return tf.random.normal(shape=tf.shape(z_mean), mean=0, stddev=1) * tf.exp(z_log_var * .5) + z_mean

    def build_instance_encoder_model(self, return_latent=True):
        self.return_latent = return_latent
        # get encoder inputs
        input_tensors = [InputFeatures.generate_input_tensors(f, batch_singleton=False) for f in self.features]

        # get list of encoded tensors for concat
        concat = [f.encode(i) for f, i, in zip(self.features, input_tensors)]
        # fusion layer
        fused = tf.keras.layers.Dense(units=self.latent_dimension, activation=ANLU(trainable=False))(tf.concat(concat, axis=-1))
        # encoder output is some latent representation - this can be used by AE directly but will require an additional layer for VAE
        latent = tf.keras.layers.Dense(units=self.latent_dimension, activation=None, name='encoder_latent')(fused)

        # generate encoder model
        if return_latent:
            self.encoder_model = tf.keras.Model(inputs=[v for f in input_tensors for k, v in f.items()], outputs=[latent], name='encoder')
        else:
            self.encoder_model = tf.keras.Model(inputs=[v for f in input_tensors for k, v in f.items()], outputs=[concat], name='encoder')

    def build_sample_encoder_model(self):
        if self.sample_features == ():
            self.sample_encoder_model = tf.keras.Model(inputs=[], outputs=[], name='sample_encoder')
        else:
            # get encoder inputs
            input_tensors = [InputFeatures.generate_input_tensors(f, batch_singleton=False) for f in self.sample_features]
            # get list of encoded tensors for concat
            concat = [f.encode(i) for f, i, in zip(self.sample_features, input_tensors)]
            # fusion layer
            fused = tf.keras.layers.Dense(units=self.sample_dimension, activation=ANLU(trainable=False))(tf.concat(concat, axis=-1))
            # generate encoder model
            self.sample_encoder_model = tf.keras.Model(inputs=[v for f in input_tensors for k, v in f.items()], outputs=[fused], name='sample_encoder')


    def build_instance_decoder_model(self):
        # set decoder input to z latent
        decoder_latent = tf.keras.Input(shape=[self.latent_dimension, ], dtype=tf.float32, name='decoder_latent')

        # get reconstructions from each feature
        reconstructions = [f.decode(decoder_latent) for f in self.features]

        # generate decoder model
        self.decoder_model = tf.keras.Model(inputs=[decoder_latent], outputs=[v for i in reconstructions for (k, v) in i.items()], name='decoder')
        for i in range(len(self.decoder_model.output)):
            m = re.search(r'/?([^/:]+_recon)[/:]', self.decoder_model.output[i].name)
            if m:
                self.decoder_model.output_names[i] = m.group(1)

    def build_vae(self, kl_weight=0.1):
        # build encoder and decoder models if not already built
        if self.encoder_model is None:
            self.build_instance_encoder_model()
        if self.decoder_model is None:
            self.build_instance_decoder_model()

        # z_mean, the output of the encoder model (could have used Model.outputs but explicit to layer due to z_log_var derivation below)
        z_mean = self.encoder_model.layers[-1].output
        # z_log_var, the input to the layer that makes z_mean is extracted and used to make an equally sized layer for z_log_var
        z_log_var = tf.keras.layers.Dense(units=z_mean.shape[-1], activation=None)(self.encoder_model.layers[-1].input)

        # reparameterize to z_latent
        z_latent = self.reparameterize(z_mean, z_log_var)
        # reconstructed input
        reconstructions = self.decoder_model(z_latent)

        # kl divergence regularization per data point
        kl_losses = tf.math.multiply(0.5, tf.reduce_mean(tf.math.square(z_mean) + tf.math.exp(z_log_var) - z_log_var - 1, axis=-1), name='kl_div_losses')

        # generate vae model
        self.vae_model = tf.keras.Model(inputs=self.encoder_model.inputs, outputs=(reconstructions if type(reconstructions) == list else [reconstructions]) + [kl_losses], name='vae_model')
        # rename to match input names and added kl term
        for i in range(len(self.decoder_model.output_names)):
            self.vae_model.output_names[i] = self.decoder_model.output_names[i]
        self.vae_model.output_names[-1] = 'kl_div_losses'

    def mil_top(self, input_tensor, hidden_units, logits_units, default_activation=tf.keras.activations.relu):
        if hidden_units is not ():
            hidden = [tf.keras.layers.Dense(units=hidden_units[0], activation=default_activation)(input_tensor)]
            for i in range(1, len(hidden_units)):
                hidden.append(tf.keras.layers.Dense(units=hidden_units[i], activation=default_activation)(hidden[-1]))
        else:
            hidden = [input_tensor]
        hidden.append(tf.keras.layers.Dense(units=logits_units, activation=None)(hidden[-1]))
        return hidden[-1]

    def build_mil_model(self, mil_hidden=(64, 32), output_dim=1, output_extra=0, output_type='classification_probability', aggregation='recursion'):
        # singleton batch trick for MIL in keras with assignment of input tensor name back to features object to be used by generator

        instance_sample_idx = tf.keras.Input(shape=[None, ], batch_size=1, dtype=tf.int32, name='instance_sample_idx_mil')
        instance_features = [v for f in self.features for k, v in InputFeatures.generate_input_tensors(f, batch_singleton=True).items()]
        sample_idx = tf.keras.Input(shape=[None, ], batch_size=1, dtype=tf.int32, name='sample_idx_mil')
        samples_n = tf.keras.Input(shape=[1, ], batch_size=1, dtype=tf.int32, name='n_samples_mil')

        # get latent for instance encoder model, removing leading singleton
        if self.return_latent:
            instance_latents = [tf.keras.activations.relu(self.encoder_model([f[0] for f in instance_features]))]
        else:
            concat = self.encoder_model([f[0] for f in instance_features])[0]
            if len(self.features) == 1:
                instance_latents = [concat]
            else:
                instance_latents = concat
                ##if you want features fused before aggregation
                # instance_latents = concat + [tf.keras.layers.Dense(units=self.latent_dimension, activation=ANLU(trainable=False))(tf.concat(concat, axis=-1))]
        sample_aggregations = []
        attentions = []

        for instance_latent in instance_latents:
            if aggregation == 'recursion':
                instance_attention_w = tf.keras.layers.Dense(units=1, activation=AISRU(trainable=True), bias_initializer=tf.keras.initializers.constant(-1.))(instance_latent)
                attentions += [instance_attention_w]
                sample_aggregation = tf.gather(tf.math.unsorted_segment_sum(instance_latent * instance_attention_w, instance_sample_idx[0], samples_n[0, 0]), sample_idx[0])
            elif aggregation == 'none':
                sample_aggregation = instance_latent
            else:
                sample_aggregation = tf.gather(tf.math.unsorted_segment_sum(instance_latent, instance_sample_idx[0], samples_n[0, 0]), sample_idx[0])
            sample_aggregations.append(sample_aggregation)

        fused_aggregations = []
        for aggregation in sample_aggregations:
            fused_aggregations.append(tf.keras.layers.Dense(self.aggregation_dimension, activation='relu')(aggregation))

        fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation='relu')(tf.concat(fused_aggregations, axis=-1))

        ##sample level information
        sample_features = [v for f in self.sample_features for k, v in InputFeatures.generate_input_tensors(f, batch_singleton=True).items()]
        if self.sample_features != ():
            sample_fused = self.sample_encoder_model([f[0] for f in sample_features])
            fused = tf.concat([fused, sample_fused], axis=-1)

        sample_logits = self.mil_top(fused, hidden_units=mil_hidden, logits_units=output_dim)

        if output_type == 'classification_probability':
            sample_output = ANLU()(sample_logits)
            sample_output /= tf.reduce_sum(sample_output, axis=-1)[..., tf.newaxis]
        elif output_type == 'logits':
            sample_output = sample_logits
        elif output_type == 'quantiles':
            y_pred = tf.keras.layers.Dense(units=1, activation=None)(sample_logits)

            sample_output = tf.concat([tf.math.subtract(y_pred, tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softplus)(sample_logits)), y_pred,
                       tf.math.add(y_pred, tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softplus)(sample_logits))], axis=1)

            sample_output = tf.keras.activations.softplus(sample_output)

        else:
            sample_output = ANLU()(sample_logits)

        # add empty column to match expected weight column in y_true & re-introduce batch singleton
        sample_output = tf.expand_dims(tf.concat([sample_output, tf.zeros([tf.shape(sample_output)[0], output_extra], dtype=tf.float32)], axis=1), axis=0, name='mil_model_output')

        self.mil_model = tf.keras.Model(inputs=[instance_sample_idx] + instance_features + sample_features + [sample_idx, samples_n], outputs=[sample_output])
        self.intermediate_model = tf.keras.Model(inputs=[instance_sample_idx] + instance_features + sample_features + [sample_idx, samples_n], outputs=[attentions])
