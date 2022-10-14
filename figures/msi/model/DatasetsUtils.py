import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

class Apply:

    class StratifiedMinibatch:
        def __init__(self, batch_size, ds_size, reshuffle_each_iteration=True):
            self.batch_size, self.ds_size, self.reshuffle_each_iteration = batch_size, ds_size, reshuffle_each_iteration
            # max number of splits
            self.n_splits = (self.ds_size // self.batch_size) + 1
            # stratified "mini-batch" via k-fold
            self.batcher = StratifiedKFold(n_splits=self.n_splits, shuffle=self.reshuffle_each_iteration)

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_strat)
                idx, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                while True:
                    for _, batch_idx in self.batcher.split(y_strat, y_strat):
                        yield tf.gather(idx, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype),
                                                  output_shapes=((None, )))

    class StratifiedBootstrap:
        def __init__(self, batch_class_sizes=[]):
            self.batch_class_sizes = batch_class_sizes
            self.batch_size = sum(self.batch_class_sizes)
            self.rnd = tf.random.Generator.from_non_deterministic_state()

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_strat)
                idx, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                assert (tf.reduce_max(y_strat).numpy() + 1) == len(self.batch_class_sizes)
                class_idx = [tf.where(y_strat == i)[:, 0] for i in range(len(self.batch_class_sizes))]
                while True:
                    batch_idx = list()
                    for j in range(len(self.batch_class_sizes)):
                        batch_idx.append(tf.gather(class_idx[j], self.rnd.uniform(shape=(self.batch_class_sizes[j], ),
                                                                                  maxval=tf.cast(class_idx[j].shape[0] - 1, tf.int64),
                                                                                  dtype=tf.int64)))
                    batch_idx = tf.concat(batch_idx, axis=0)

                    yield tf.gather(idx, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype),
                                                  output_shapes=((self.batch_size)))



    class SubSample:
        def __init__(self, batch_size, ds_size):
            self.batch_size = batch_size
            self.ds_size = ds_size
        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of idx
                idx = tf.stack([element for element in ds_input])
                while True:
                    batch_idx = np.random.choice(np.arange(self.ds_size), self.batch_size, replace=False)
                    yield tf.gather(idx, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec.dtype),
                                                  output_shapes=((None, )))



class Map:

    class LoadBatchByIndices:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], self.tf_output_types)
            flat_values.set_shape((None,) + self.inner_shape)

            if self.ragged_constructor:
                return self.ragged_constructor(flat_values, *additional_args)
            else:
                return flat_values

    class FromNumpy(LoadBatchByIndices):
        def __init__(self, data, data_type, dropout=0):
            self.data = data
            if self.data.dtype == np.dtype('O'):
                self.dropout = dropout
                self.inner_shape = data[0].shape[1:]
                self.tf_output_types = [data_type, tf.int32]
                self.ragged_constructor = tf.RaggedTensor.from_row_lengths
            else:
                self.inner_shape = data.shape[1:]
                self.tf_output_types = [data_type]
                self.ragged_constructor = None

        def loader(self, idx):
            if self.data.dtype == np.dtype('O'):
                if self.dropout:
                    batch = list()
                    for i in idx.numpy():
                        batch.append(self.data[i][np.random.random(len(self.data[i])) > self.dropout])
                    return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])
                else:
                    batch = self.data[idx.numpy()]
                    return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])

            else:
                return self.data[idx.numpy()]

    class LoadIndices:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            indices, *additional_args = tf.py_function(self.loader, [sample_idx], [tf.bool])

            return sample_idx, indices

    class FromNumpytoIndices(LoadIndices):
        def __init__(self, data, dropout=.2):
            self.data = data
            self.dropout = dropout

        def loader(self, idx):
            batch = list()
            for i in idx.numpy():
                batch.append(np.random.random(len(self.data[i])) > self.dropout)
            return np.concatenate(batch, axis=0)

    class LoadBatchByDroppedIndices:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx, boolean):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            flat_values, *additional_args = tf.py_function(self.loader, [sample_idx, boolean], self.tf_output_types)
            flat_values.set_shape((None,) + self.inner_shape)

            return self.ragged_constructor(flat_values, *additional_args)

    class FromNumpyandIndices(LoadBatchByDroppedIndices):
        def __init__(self, data, data_type):
            self.data = data
            self.tf_output_types = [data_type, tf.int32]
            self.inner_shape = data[0].shape[1:]
            self.ragged_constructor = tf.RaggedTensor.from_row_lengths

        def loader(self, idx, boolean):
            batch = list()
            boolean = boolean.numpy()
            index = 0
            for i in idx.numpy():
                temp_batch = self.data[i]
                temp_boolean = boolean[index: index + len(temp_batch)]
                batch.append(temp_batch[temp_boolean])
                index += len(temp_batch)
            return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])
