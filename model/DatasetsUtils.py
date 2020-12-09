import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold




class Apply:

    class StratifiedMinibatch:
        def __init__(self, batch_size, ds_size):
            self.batch_size, self.ds_size = batch_size, ds_size
            # max number of splits
            self.n_splits = self.ds_size // self.batch_size
            # stratified "mini-batch" via k-fold
            self.batcher = StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_true, y_strat)
                idx, y_true, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                while True:
                    for _, batch_idx in self.batcher.split(y_strat, y_strat):
                        yield tf.gather(idx, batch_idx, axis=0), tf.gather(y_true, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype, ds_input.element_spec[1].dtype),
                                                  output_shapes=((None, ), (None, ds_input.element_spec[1].shape[0])))

    class StratifiedBootstrap:
        def __init__(self, batch_class_sizes=[]):
            self.batch_class_sizes = batch_class_sizes
            self.batch_size = sum(self.batch_class_sizes)
            self.rnd = tf.random.Generator.from_non_deterministic_state()

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_true, y_strat)
                idx, y_true, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                assert (tf.reduce_max(y_strat).numpy() + 1) == len(self.batch_class_sizes)
                class_idx = [tf.where(y_strat == i)[:, 0] for i in range(len(self.batch_class_sizes))]
                while True:
                    batch_idx = list()
                    for j in range(len(self.batch_class_sizes)):
                        batch_idx.append(tf.gather(class_idx[j], self.rnd.uniform(shape=(self.batch_class_sizes[j], ),
                                                                                  maxval=tf.cast(class_idx[j].shape[0] - 1, tf.int64),
                                                                                  dtype=tf.int64)))
                    batch_idx = tf.concat(batch_idx, axis=0)

                    yield tf.gather(idx, batch_idx, axis=0), tf.gather(y_true, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype, ds_input.element_spec[1].dtype),
                                                  output_shapes=((self.batch_size, ), (self.batch_size, ds_input.element_spec[1].shape[0])))







class Map:

    class LoadBatchByIndices:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx, ragged_output):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], self.tf_output_types)
            flat_values.set_shape((None,) + self.inner_shape)

            if ragged_output:
                return self.ragged_constructor(flat_values, *additional_args)
            else:
                return flat_values


    class FromNumpy(LoadBatchByIndices):
        def __init__(self, data, data_type):
            self.data = data
            self.tf_output_types = [data_type, tf.int32]
            self.inner_shape = data[0].shape[1:]
            self.ragged_constructor = tf.RaggedTensor.from_row_lengths

        def loader(self, idx):
            batch = list()
            for i in idx.numpy():
                batch.append(self.data[i])
            return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])




#
#
#
#
#
# class Map:
#     class LoadBatchIndex:
#         def __init__(self, loaders):
#             self.loaders = loaders
#
#         def __call__(self, sample_idx, to_ragged):
#             # flat_values and additional_args together should be the input into the ragged_constructor of the loader
#             data = list()
#             for index, loader in enumerate(self.loaders):
#                 self.loader = loader
#                 flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], self.loader.tf_output_types)
#                 flat_values.set_shape((None, ) + self.loader.inner_shape)
#
#                 if to_ragged[index]:
#                     data.append(self.loader.ragged_constructor(flat_values, *additional_args))
#                 else:
#                     data.append(flat_values)
#             return tuple(data)
#
# class Loaders:
#     class FromNumpy:
#         def __init__(self, data, data_type):
#             self.data = data
#             self.tf_output_types = [data_type, tf.int32]
#             self.inner_shape = data[0].shape[1:]
#             self.ragged_constructor = tf.RaggedTensor.from_row_lengths
#
#         def __call__(self, idx):
#             batch = list()
#             for i in idx.numpy():
#                 batch.append(self.data[i])
#             return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])

