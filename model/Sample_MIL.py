import tensorflow as tf
from model.KerasLayers import Activations, Ragged, Embed, StrandWeight, Dropout

class InstanceModels:

    class GeneEmbed:
        def __init__(self, shape=None, dim=None, input_dim=None, regularization=.01):
            self.shape = shape
            self.regularization = regularization
            self.input_dim = input_dim
            self.dim = dim
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.int32)
            output = Embed(input_dimension=self.input_dim, embedding_dimension=self.dim, regularization=self.regularization, trainable=True)(input)
            ##we do a log on the graph so we can't have negative numbers
            output = tf.keras.activations.relu(output)
            self.model = tf.keras.Model(inputs=[input], outputs=[output])

    class VariantPositionBin:
        def __init__(self, bins, fusion_dimension=128, default_activation=tf.keras.activations.relu):
            self.bins = bins
            self.fusion_dimension = fusion_dimension
            self.default_activation = default_activation
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            bin_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.int32) for i in self.bins]
            pos_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            embeds = [Embed(embedding_dimension=i, trainable=False) for i in self.bins]
            bins_fused = tf.concat([emb(i) for i, emb in zip(bin_inputs, embeds)], axis=-1)
            bins_fused = tf.keras.layers.Dense(units=sum(self.bins), activation=Activations.ARU())(bins_fused)
            pos_loc = tf.keras.layers.Dense(units=64, activation=Activations.ASU())(pos_input)
            pos_loc = tf.keras.layers.Dense(units=32, activation=Activations.ARU())(pos_loc)
            fused = tf.concat([bins_fused, pos_loc], axis=-1)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=Activations.ARU())(fused)
            self.model = tf.keras.Model(inputs=bin_inputs + [pos_input], outputs=[fused])


    class VariantBin:
        def __init__(self, bins, layers=[], default_activation=tf.keras.activations.relu, fused_dropout=0, layer_dropout=0):
            self.bins = bins
            self.layers = layers
            self.default_activation = default_activation
            self.layer_dropout = layer_dropout
            self.fused_dropout = fused_dropout
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            bin_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.int32) for i in self.bins]
            embeds = [Embed(embedding_dimension=i, trainable=False) for i in self.bins]
            bins_fused = [tf.concat([emb(i) for i, emb in zip(bin_inputs, embeds)], axis=-1)]
            if self.fused_dropout:
                bins_fused.append(tf.keras.layers.Dropout(self.fused_dropout)(bins_fused[-1]))
            for index, i in enumerate(self.layers):
                bins_fused.append(tf.keras.layers.Dense(units=i, activation=self.default_activation)(bins_fused[-1]))
                bins_fused.append(tf.keras.layers.Dropout(self.layer_dropout)(bins_fused[-1]))
            self.model = tf.keras.Model(inputs=bin_inputs, outputs=[bins_fused[-1]])


    class VariantSequence:
        def __init__(self, sequence_length, sequence_embedding_dimension, n_strands, convolution_params, fusion_dimension=64, default_activation=tf.keras.activations.relu, use_frame=False, regularization=.01):
            self.sequence_length = sequence_length
            self.sequence_embedding_dimension = sequence_embedding_dimension
            self.convolution_params = convolution_params
            self.default_activation = default_activation
            self.n_strands = n_strands
            self.use_frame = use_frame
            self.fusion_dimension = fusion_dimension
            self.regularization=regularization
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
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation, kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(fused)
            fused = tf.reduce_max(StrandWeight(trainable=True, n_features=fused.shape[2])(strand) * fused, axis=1)
            # fused = tf.reduce_mean(StrandWeight(trainable=True, n_features=fused.shape[2])(strand) * fused, axis=1)
            # fused = fused - tf.reduce_mean(fused, axis=-1, keepdims=True)

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
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            self.model = tf.keras.Model(inputs=[input], outputs=[input])

    class Feature:
        def __init__(self, shape=None, input_dropout=.5, layer_dropouts=[.2, .2], layers=[64, 32], regularization=0):
            self.shape = shape
            self.model = None
            self.input_dropout = input_dropout
            self.layer_dropouts = layer_dropouts
            self.layers = layers
            self.regularization = regularization
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            hidden = [tf.keras.layers.Dropout(self.input_dropout)(input)]
            for i, j in zip(self.layers, self.layer_dropouts):
                hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(hidden[-1]))
                hidden.append(tf.keras.layers.Dropout(j)(hidden[-1]))
            self.model = tf.keras.Model(inputs=[input], outputs=[hidden[-1]])

    class Type:
        def __init__(self, shape=None, dim=None):
            self.shape = shape
            self.dim = dim
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.int32)
            # type_emb = Embed(embedding_dimension=self.dim, trainable=False)
            # self.model = tf.keras.Model(inputs=[input], outputs=[type_emb(input)])
            self.model = tf.keras.Model(inputs=[input], outputs=[tf.one_hot(input, self.dim)])

    class Reads:
        def __init__(self, read_layers=[8, 16], fused_layers=[32, 64]):
            self.read_layers = read_layers
            self.fused_layers = fused_layers
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            ref_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            ref = [ref_input]
            for i in self.read_layers:
                ref.append(tf.keras.layers.Dense(units=i, activation='relu')(ref[-1]))
            alt_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            alt = [alt_input]
            for i in self.read_layers:
                alt.append(tf.keras.layers.Dense(units=i, activation='relu')(alt[-1]))
            fused = [tf.concat([ref[-1], alt[-1]], axis=-1)]
            for i in self.fused_layers:
                fused.append(tf.keras.layers.Dense(units=i, activation='relu')(fused[-1]))

            self.model = tf.keras.Model(inputs=[ref_input, alt_input], outputs=[fused[-1]])

class RaggedModels:
    class MIL:
        def __init__(self,
                     instance_encoders=[],
                     sample_encoders=[],
                     instance_layers=[],
                     output_dims=[1],
                     output_names=[],
                     mode='attention',
                     pooling='sum',
                     regularization=.2,
                     fusion='after',
                     mil_hidden=[32, 16],
                     dynamic_hidden=[64, 32],
                     attention_layers=[16],
                     dropout=0,
                     instance_dropout=0,
                     input_dropout=False,
                     heads=1):
            self.instance_encoders,\
            self.sample_encoders,\
            self.instance_layers,\
            self.output_dims,\
            self.output_names,\
            self.mode,\
            self.pooling,\
            self.regularization,\
            self.fusion,\
            self.mil_hidden,\
            self.dynamic_hidden,\
            self.attention_layers,\
            self.dropout,\
            self.instance_dropout,\
            self.input_dropout,\
            self.heads = instance_encoders,\
                         sample_encoders,\
                         instance_layers,\
                         output_dims,\
                         output_names,\
                         mode,\
                         pooling,\
                         regularization,\
                         fusion,\
                         mil_hidden,\
                         dynamic_hidden,\
                         attention_layers,\
                         dropout,\
                         instance_dropout,\
                         input_dropout,\
                         heads

            if self.output_names == []:
                self.output_names = ['output_' + str(index) for index, i in enumerate(self.output_dims)]
            self.model, self.attention_model = None, None
            self.build()

        def build(self):
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]

            ##sample level model encodings
            if self.sample_encoders != []:
                sample_encodings = [encoder(sample_input) for sample_input, encoder in zip(sample_inputs, self.sample_encoders)]
                sample_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(sample_encodings)

            if self.instance_encoders != []:
                ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
                # flatten encoders if needed
                ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]

                # based on the design of the input and graph instances can be fused prior to bag aggregation
                ragged_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)

                if self.instance_dropout:
                    ragged_fused = Ragged.MapFlatValues(tf.keras.layers.Dropout(self.instance_dropout))(ragged_fused)

                if self.sample_encoders != []:
                    if self.fusion == 'before':
                        ragged_hidden = [Ragged.Dense(units=64, activation=tf.keras.activations.relu)((ragged_fused, sample_fused))]
                    else:
                        ragged_hidden = [ragged_fused]
                else:
                    ragged_hidden = [ragged_fused]

                for i in self.instance_layers:
                    ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_hidden[-1]))
                    if self.dropout:
                        ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dropout(self.dropout))(ragged_hidden[-1]))

                if self.mode == 'attention':
                    if self.pooling == 'dynamic':
                        ##only implemented for one attention head
                        pooling_1, ragged_attention_weights_1 = Ragged.Attention(pooling='mean', regularization=self.regularization, layers=self.attention_layers)(ragged_hidden[-1])
                        for index, i in enumerate(self.dynamic_hidden):
                            if index == 0:
                                instance_ragged_fused = [Ragged.Dense(units=i, activation=tf.keras.activations.relu)((ragged_hidden[-1], pooling_1[:, 0, :]))]
                            else:
                                instance_ragged_fused.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(instance_ragged_fused[-1]))
                        pooling_2, ragged_attention_weights = Ragged.Attention(pooling='dynamic', regularization=self.regularization, layers=self.attention_layers)([ragged_hidden[-1], instance_ragged_fused[-1]])
                        pooled_hidden = [pooling_2]
                    else:
                        pooling, ragged_attention_weights = Ragged.Attention(pooling=self.pooling, regularization=self.regularization, layers=self.attention_layers, heads=self.heads)(ragged_hidden[-1])
                        pooled_hidden = [pooling]

                else:
                    if self.pooling == 'mean':
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    else:
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    pooled_hidden = [pooling]

                if self.pooling != 'mean' and self.input_dropout:
                    pooled_hidden = [Dropout(self.input_dropout)(pooled_hidden[-1])]

                if self.pooling != 'mean':
                    pooled_hidden = [tf.math.log(pooled_hidden[-1] + 1)]

                aggregation = pooled_hidden[-1]


                if self.mode == 'none': #assumes you need a newaxis
                    pooled_hidden = [pooled_hidden[-1][:, tf.newaxis, :]]
                    assert self.heads == 1

            if self.sample_encoders != []:
                if self.fusion == 'after':
                    if self.instance_encoders != []:
                        ##need to broadcast
                        fused = [tf.concat([pooled_hidden[-1], tf.broadcast_to(sample_fused[:, tf.newaxis, :], [tf.shape(sample_fused)[0], self.heads, sample_fused.shape[-1]])], axis=-1)]
                    else:
                        fused = [sample_fused[:, tf.newaxis, :]]
                else:
                    fused = [pooled_hidden[-1]]

            else:
                fused = [pooled_hidden[-1]]

            head_networks = [[fused[-1][:, head, :]] for head in range(self.heads)]

            for i in self.mil_hidden:
                for head in range(self.heads):
                    head_networks[head].append(tf.keras.layers.Dense(units=i, activation='relu')(head_networks[head][-1]))
                    if self.dropout:
                        head_networks[head].append(tf.keras.layers.Dropout(self.dropout)(head_networks[head][-1]))

            output_tensors = []
            for output_dim, output_name in zip(self.output_dims, self.output_names):
                if self.heads == 1:
                    head_networks[0].append(tf.keras.layers.Dense(units=output_dim, activation=None)(head_networks[0][-1]))
                    output_tensors.append(head_networks[0][-1])
                else:
                    for head in range(self.heads):
                        head_networks[head].append(tf.keras.layers.Dense(units=1, activation=None)(head_networks[head][-1]))
                    if self.heads == output_dim:
                        output_tensors.append(tf.concat([i[-1] for i in head_networks], axis=-1))
                    else:
                        output_tensors.append(tf.keras.layers.Dense(units=output_dim, activation=None)(tf.concat([i[-1] for i in head_networks], axis=-1)))

            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=output_tensors)
            self.fused_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[fused[-1]])
            if self.instance_encoders != []:
                self.hidden_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_hidden[-1]])
            if self.mode == 'attention':
                self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_attention_weights])
                self.aggregation_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[aggregation])

