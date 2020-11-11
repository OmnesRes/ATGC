import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold


class Apply:
    class StratifiedMinibatch:
        def __init__(self, batch_size, ds_size, n_repeats=1):
            self.batch_size = batch_size
            self.ds_size = ds_size
            # max number of splits
            self.n_splits = self.ds_size // self.batch_size
            # stratified "mini-batch" via k-fold
            self.batcher = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=n_repeats)

        def __call__(self, ds_input: tf.data.Dataset):
            # expecting ds of (idx, y_true), drop remainder is implicit in this implementation
            idx, y_true, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input.take(self.n_splits * self.batch_size)))))))
            stratification_tensor = y_strat
            ds_batched = [[tf.gather(idx, idx_batch, axis=0), tf.gather(y_true, idx_batch, axis=0)] for _, idx_batch in self.batcher.split(stratification_tensor, stratification_tensor)]

            return tf.data.Dataset.from_tensor_slices(tuple(map(list, zip(*ds_batched))))

    class StratifiedBootstrap:
        def __init__(self, batch_class_sizes=[], n_batches=64):
            self.batch_class_sizes = batch_class_sizes
            self.n_batches = n_batches
            self.rnd = tf.random.Generator.from_non_deterministic_state()

        def __call__(self, ds_input: tf.data.Dataset):
            # expecting ds of (idx, y_true), drop remainder is implicit in this implementation
            idx, y_true, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
            assert y_true.shape[1] == len(self.batch_class_sizes)
            batch_idx = list()
            for i in range(len(self.batch_class_sizes)):
                class_idx = tf.where(y_strat == i)[:, 0]
                batch_rnd = self.rnd.uniform(shape=((self.n_batches * self.batch_class_sizes[i]),), maxval=tf.cast(class_idx.shape[0] - 1, tf.int64), dtype=tf.int64)
                batch_idx.append(tf.split(tf.gather(class_idx, batch_rnd), self.n_batches))

            batch_idx = list(map(lambda x: tf.concat(x, axis=0), list(map(list, zip(*batch_idx)))))

            return tf.data.Dataset.from_tensor_slices((list(map(lambda x: tf.gather(idx, x), batch_idx)), list(map(lambda x: tf.gather(y_true, x), batch_idx))))




class Map:
    class LoadBatchIndex:
        def __init__(self, loaders):
            self.loaders = loaders

        def __call__(self, sample_idx, to_ragged):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            data = list()
            for index, loader in enumerate(self.loaders):
                self.loader = loader
                flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], self.loader.tf_output_types)
                flat_values.set_shape((None, ) + self.loader.inner_shape)

                if to_ragged[index]:
                    data.append(self.loader.ragged_constructor(flat_values, *additional_args))
                else:
                    data.append(flat_values)
            return tuple(data)

class Loaders:
    class FromNumpy:
        def __init__(self, data, data_type):
            self.data = data
            self.tf_output_types = [data_type, tf.int32]
            self.inner_shape = data[0].shape[1:]
            self.ragged_constructor = tf.RaggedTensor.from_row_lengths

        def __call__(self, idx):
            batch = list()
            for i in idx.numpy():
                batch.append(self.data[i])
            return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])

