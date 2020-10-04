import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold

class Losses:
    class Weighted:
        # unifying usage in these functions (which can be considered losses or metrics)
        # add weights in as the last column of y_true & singleton batch dim
        class CrossEntropyfromlogits:
            @staticmethod
            def cross_entropy(y_true, y_pred):
                return tf.reduce_mean(Losses.Weighted.CrossEntropyfromlogits._loss(y_true, y_pred))

            @staticmethod
            def cross_entropy_weighted(y_true, y_pred):
                return Losses.Weighted._weighted_average(Losses.Weighted.CrossEntropyfromlogits._loss(y_true, y_pred), y_true[0, :, -1])

            @staticmethod
            def _loss(y_true, y_pred, loss_clip=0.):
                return tf.maximum(tf.keras.losses.CategoricalCrossentropy(reduction='none', from_logits=True)(y_true[0, :, :-1], y_pred[0, :, :-1]) - loss_clip, 0.)

        class CrossEntropy:
            @staticmethod
            def cross_entropy(y_true, y_pred):
                return tf.reduce_mean(Losses.Weighted.CrossEntropy._loss(y_true, y_pred))

            @staticmethod
            def cross_entropy_weighted(y_true, y_pred):
                return Losses.Weighted._weighted_average(Losses.Weighted.CrossEntropy._loss(y_true, y_pred), y_true[0, :, -1])

            @staticmethod
            def _loss(y_true, y_pred, loss_clip=0.):
                return tf.maximum(tf.keras.losses.CategoricalCrossentropy(reduction='none', from_logits=False)(y_true[0, :, :-1], y_pred[0, :, :-1]) - loss_clip, 0.)


        class Accuracy:
            @staticmethod
            def accuracy(y_true, y_pred):
                return tf.reduce_mean(Losses.Weighted.Accuracy._loss(y_true, y_pred))

            @staticmethod
            def accuracy_weighted(y_true, y_pred):
                return Losses.Weighted._weighted_average(Losses.Weighted.Accuracy._loss(y_true, y_pred), y_true[0, :, -1])

            @staticmethod
            def _loss(y_true, y_pred):
                return tf.cast(tf.equal(tf.argmax(y_true[0, :, :-1], axis=-1), tf.argmax(y_pred[0, :, :-1], axis=-1)), dtype=tf.float32)

        class MeanSquaredError:
            @staticmethod
            def mean_squared_error(y_true, y_pred):
                return tf.reduce_mean(Losses.Weighted.MeanSquaredError._loss(y_true, y_pred))

            @staticmethod
            def mean_squared_error_weighted(y_true, y_pred):
                return Losses.Weighted._weighted_average(Losses.Weighted.MeanSquaredError._loss(y_true, y_pred), y_true[0, :, -1])

            @staticmethod
            def _loss(y_true, y_pred):
                return tf.keras.losses.MeanSquaredError(reduction='none')(y_true[0, :, :-1], y_pred[0, :, :-1])

        class QuantileLoss:
            @staticmethod
            def _check_function(x, tau):
                return x * (tau[tf.newaxis, :] - tf.cast(tf.less(x, 0.), tf.float32))

            @staticmethod
            def quantile_loss(y_true, y_pred):
                return tf.reduce_mean(Losses.Weighted.QuantileLoss._loss(y_true, y_pred))

            @staticmethod
            def quantile_loss_weighted(y_true, y_pred):
                return Losses.Weighted._weighted_average(Losses.Weighted.QuantileLoss._loss(y_true, y_pred), y_true[0, :, -1])

            @staticmethod
            def _loss(y_true, y_pred, alpha=0.05):
                # y_true is a vector
                # y_pred is the n x 3 coming out from the net for regression [low ci, mid-point, high ci]

                # quantiles are made from alpha
                quantiles = tf.constant(((alpha / 2), 0.5, 1 - (alpha / 2)))

                # residuals
                residuals = y_true[0, :, :-1] - y_pred[0, :, :-1]
                # return check function with quantiles - quantile loss
                return tf.reduce_mean(Losses.Weighted.QuantileLoss._check_function(residuals, quantiles), axis=-1)

        @staticmethod
        def _weighted_average(losses, weights):
            return tf.reduce_sum(losses * weights) / tf.reduce_sum(weights)

    class VAE:
        @staticmethod
        def kl_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)


class BatchGenerator:
    def __init__(self, x_instance_sample_idx, x_instance_features, x_sample, y_label, y_weights=None, y_stratification=None, idx_sample=None, sampling_approach='minibatch', batch_size=100, n_repeats=1):
        # x_instance_dict: dictionary of instance level features
        # set sample_idx if not provided based on size of sample data frame
        if idx_sample is None:
            self.idx_sample = np.arange(y_label.shape[0])
        else:
            self.idx_sample = idx_sample

        # set internal variables
        self.n_repeats = n_repeats
        self.batch_size = batch_size
        self.n_splits = np.ceil(len(self.idx_sample) / self.batch_size).astype(int)
        self.sampling_approach = sampling_approach
        self.x_instance_features = x_instance_features
        self.x_sample = x_sample
        self.y_label = y_label
        self.y_weights = y_weights if y_weights is not None else np.ones(y_label.shape[0])
        if x_instance_sample_idx is None:
            self.x_instance_sample_idx = np.arange(self.y_label.shape[0])
            self.x_sample_instance_idx = self.x_instance_sample_idx[:, np.newaxis]
        else:
            self.x_instance_sample_idx = x_instance_sample_idx
            self.x_sample_instance_idx = np.array([np.where(x_instance_sample_idx == i)[0] for i in range(y_label.shape[0])], dtype=object)

        # set no stratification (ie all zeros) if y_stratification is None
        if y_stratification is None:
            self.y_stratification = np.zeros(self.y_label.shape[0])
        else:
            self.y_stratification = y_stratification

        # instantiate "batching" object - RepeatedStratifiedKFold
        if sampling_approach == 'minibatch':
            self.stratified_splitter = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        else:
            self.stratified_splitter = None

    def idx_generator(self):
        # make sample batch index generator
        if self.sampling_approach is None:
            return (self.idx_sample for _ in range(1))
        if self.sampling_approach == 'bootstrap':
            return (np.random.choice(self.idx_sample, self.batch_size, replace=True) for _ in range(self.n_repeats))
        elif self.sampling_approach == 'subsample':
            return (np.random.choice(self.idx_sample, self.batch_size, replace=False) for _ in range(self.n_repeats))
        elif self.sampling_approach == 'minibatch':
            return (self.idx_sample[_[1]] for _ in self.stratified_splitter.split(self.y_stratification[self.idx_sample], self.y_stratification[self.idx_sample]))

    def data_generator(self):
        while True:
            for idx_batch_sample in self.idx_generator():
                # concat maf idx of variants for samples in batch
                idx_batch_instances = np.concatenate(self.x_sample_instance_idx[idx_batch_sample])

                batch = ({k: v for d in [{'instance_sample_idx_mil': self.x_instance_sample_idx[idx_batch_instances][np.newaxis, ...]},  # instance sample_idx
                                         {k + '_mil': v[idx_batch_instances][np.newaxis, ...] for f in self.x_instance_features for k, v in f.inputs.items()},  # instance features
                                         {k + '_mil': v[idx_batch_sample][np.newaxis, ...] for f in self.x_sample for k, v in f.inputs.items()},  # sample features
                                         {'sample_idx_mil': idx_batch_sample[np.newaxis, ...], 'n_samples_mil': np.array(self.y_label.shape[0])[np.newaxis, np.newaxis]}]  # samples idx of the batch & total number of samples
                         for k, v, in d.items()},
                         [np.concatenate([self.y_label[idx_batch_sample], self.y_weights[idx_batch_sample][:, np.newaxis]], axis=1)[np.newaxis, ...]])  # sample level labels with weights as last column

                yield batch
