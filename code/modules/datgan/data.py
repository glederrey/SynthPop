"""Data related functionalities.

This modules contains the tools to preprare the data, from the raw csv files, to the DataFlow
objects will be used to fit our models.
"""
import os
import urllib

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
from tensorpack import DataFlow, RNGDataFlow
from tensorpack.utils import logger

from modules.datgan.utils import get_persistent_homology


def check_metadata(metadata):
    """Check that the given metadata has correct types for all its members.

    Args:
        metadata(dict): Description of the inputs.

    Returns:
        None

    Raises:
        AssertionError: If any of the details is not valid.

    """
    message = 'The given metadata contains unsupported types.'
    assert all([metadata['details'][col]['type'] in ['category', 'continuous'] for col in metadata['details'].keys()]), message


def check_inputs(function):
    """Validate inputs for functions whose first argument is a numpy.ndarray with shape (n,1).

    Args:
        function(callable): Method to validate.

    Returns:
        callable: Will check the inputs before calling :attr:`function`.

    Raises:
        ValueError: If first argument is not a valid :class:`numpy.array` of shape (n, 1).

    """
    def decorated(self, data, *args, **kwargs):
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 1):
            raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

        return function(self, data, *args, **kwargs)

    decorated.__doc__ = function.__doc__
    return decorated


class DATGANDataFlow(RNGDataFlow):
    """Subclass of :class:`tensorpack.RNGDataFlow` prepared to work with :class:`numpy.ndarray`.

    Attributes:
        shuffle(bool): Wheter or not to shuffle the data.
        metadata(dict): Metadata for the given :attr:`data`.
        num_features(int): Number of features in given data.
        data(list): Prepared data from :attr:`filename`.
        distribution(list): DepecrationWarning?

    """

    def __init__(self, data, metadata, shuffle=True):
        """Initialize object.

        Args:
            filename(str): Path to the json file containing the metadata.
            shuffle(bool): Wheter or not to shuffle the data.

        Raises:
            ValueError: If any column_info['type'] is not supported

        """
        self.shuffle = shuffle
        if self.shuffle:
            self.reset_state()

        self.metadata = metadata
        self.num_features = self.metadata['num_features']

        self.data = []
        self.distribution = []

        for col in self.metadata['details'].keys():
            column_info = self.metadata['details'][col]
            if column_info['type'] == 'continuous':
                col_data = data[col]
                n = column_info['n']
                value = col_data[:, :n]
                cluster = col_data[:, n:]

                self.data.append(value)
                self.data.append(cluster)

            elif column_info['type'] == 'category':
                col_data = np.asarray(data[col], dtype='int32')
                self.data.append(col_data)

            else:
                raise ValueError(
                    "column_info['type'] must be either 'category' or 'continuous'."
                    "Instead it was '{}'.".format(column_info['type'])
                )

        self.data = list(zip(*self.data))

    def size(self):
        """Return the number of rows in data.

        Returns:
            int: Number of rows in :attr:`data`.

        """
        return len(self.data)

    def get_data(self):
        """Yield the rows from :attr:`data`.

        Yields:
            tuple: Row of data.

        """
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            yield self.data[k]

    def __iter__(self):
        """Iterate over self.data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.size()


class RandomZData(DataFlow):
    """Random dataflow.

    Args:
        shape(tuple): Shape of the array to return on :meth:`get_data`

    """

    def __init__(self, shape):
        """Initialize object."""
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        """Yield random normal vectors of shape :attr:`shape`."""
        while True:
            yield [np.random.normal(0, 1, size=self.shape)]

    def __iter__(self):
        """Return data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.shape[0]


class MultiModalNumberTransformer:
    r"""Reversible transform for multimodal data.

    To effectively sample values from a multimodal distribution, we cluster values of a
    numerical variable using a `skelarn.mixture.GaussianMixture`_ model (GMM).

    * We train a GMM with :attr:`n` components for each numerical variable :math:`C_i`.
      GMM models a distribution with a weighted sum of :attr:`n` Gaussian distributions.
      The means and standard deviations of the :attr:`n` Gaussian distributions are
      :math:`{\eta}^{(1)}_{i}, ..., {\eta}^{(n)}_{i}` and
      :math:`{\sigma}^{(1)}_{i}, ...,{\sigma}^{(n)}_{i}`.

    * We compute the probability of :math:`c_{i,j}` coming from each of the :attr:`n` Gaussian
      distributions as a vector :math:`{u}^{(1)}_{i,j}, ..., {u}^{(n)}_{i,j}`. u_{i,j} is a
      normalized probability distribution over :attr:`n` Gaussian distributions.

    * We normalize :math:`c_{i,j}` as :math:`v_{i,j} = (c_{i,j}−{\eta}^{(k)}_{i})/2{\sigma}^
      {(k)}_{i}`, where :math:`k = arg max_k {u}^{(k)}_{i,j}`. We then clip :math:`v_{i,j}` to
      [−0.99, 0.99].

    Then we use :math:`u_i` and :math:`v_i` to represent :math:`c_i`. For simplicity,
    we cluster all the numerical features, i.e. both uni-modal and multi-modal features are
    clustered to :attr:`n = 5` Gaussian distributions.

    The simplification is fair because GMM automatically weighs :attr:`n` components.
    For example, if a variable has only one mode and fits some Gaussian distribution, then GMM
    will assign a very low probability to :attr:`n − 1` components and only 1 remaining
    component actually works, which is equivalent to not clustering this feature.

    Args:
        max_clusters(int): Maximum number of Gaussian distributions in Bayesian GMM
        weight_threshold(float): Weight threshold for a Gaussian distribution to be kept.

    Attributes:
        num_modes(int): Number of components in the `skelarn.mixture.GaussianMixture`_ model.

    .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
        sklearn.mixture.GaussianMixture.html

    """

    def __init__(self):
        """Initialize instance."""
        self.max_clusters = 10
        self.std_span = 2
        self.n_bins = 50
        self.thresh = 1e-3

    @check_inputs
    def transform(self, data):
        """Cluster values using a `skelarn.mixture.BayesianGaussianMixture`_ model.

        Args:
            data(numpy.ndarray): Values to cluster in array of shape (n,1).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]: Tuple containg the features,
            probabilities, averages and stds of the given data.

        .. _skelarn.mixture.BayesianGaussianMixture: https://scikit-learn.org/stable/modules/generated/
            sklearn.mixture.BayesianGaussianMixture.html

        """

        # Transform data using the histogram function
        hist = np.histogram(data, bins=self.n_bins, density=True)

        # Use the persistent homology to find the number of peaks in the data
        peaks = get_persistent_homology(hist[0])
        pers = np.array([p.get_persistence(hist[0]) for p in peaks])

        n_peaks = sum(pers > self.thresh) - 1

        logger.info("  Found {} peaks!".format(n_peaks))

        # Decide on the number of modes for the BGM
        if n_peaks < 2:
            n_modes = self.max_clusters
        else:
            n_modes = min(n_peaks, self.max_clusters)

        logger.info("  Encoding with {} components.".format(n_modes))
        while True:
            # Fit the BGM
            model = BayesianGaussianMixture(
                n_components=n_modes,
                max_iter=200,
                n_init=10,
                init_params='kmeans',
                reg_covar=0,
                weight_concentration_prior_type='dirichlet_process',
                mean_precision_prior=.8,
                tol=1e-5,
                weight_concentration_prior=1e-5)
            model.fit(data)

            # Check that BGM is using all the classes!
            pred_ = np.unique(model.predict(data))

            if len(pred_) == n_modes:
                logger.info("  Predictions were done on {:d} components => FINISHED!".format(n_modes))
                valid_component_indicator = np.array([True] * n_modes)
                break
            else:
                n_modes = len(pred_)
                logger.info("  Predictions were done on only {:d} components => Simplification!".format(n_modes))

        means = model.means_.reshape((1, n_modes))
        stds = np.sqrt(model.covariances_).reshape((1, n_modes))

        # Fix this shitty normalization
        normalized_values = ((data - means) / (self.std_span * stds))
        probs = model.predict_proba(data)

        # Clip the values
        normalized_values = np.clip(normalized_values, -.99, .99)

        return normalized_values, probs, model, n_modes

    #@staticmethod
    def inverse_transform(self, data, info):
        """Reverse the clustering of values.

        Args:
            data(numpy.ndarray): Transformed data to restore.
            info(dict): Metadata.

        Returns:
           numpy.ndarray: Values in the original space.

        """

        gmm = info['transform']
        n_modes = info['n']

        normalized_values = data[:, :n_modes]
        probs = data[:, n_modes:]

        selected_component = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            component_prob_t = probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            selected_component[i] = np.random.choice(np.arange(n_modes), p=component_prob_t)

        means = gmm.means_.reshape([-1])
        stds = np.sqrt(gmm.covariances_).reshape([-1])

        mean_t = means[selected_component]
        std_t = stds[selected_component]

        selected_normalized_value = normalized_values[np.arange(len(data)), selected_component]

        return selected_normalized_value * self.std_span * std_t + mean_t


class Preprocessor:
    """Transform back and forth human-readable data into TGAN numerical features.

    Args:
        continous_columns(list): List of columns to be considered continuous
        metadata(dict): Metadata to initialize the object.

    Attributes:
        continous_columns(list): Same as constructor argument.
        metadata(dict): Information about the transformations applied to the data and its format.
        continous_transformer(MultiModalNumberTransformer):
            Transformer for columns in :attr:`continuous_columns`
        categorical_transformer(CategoricalTransformer):
            Transformer for categorical columns.
        columns(list): List of columns labels.

    """

    def __init__(self, continuous_columns=None, metadata=None, columns_order=None):
        """Initialize object, set arguments as attributes, initialize transformers."""
        if continuous_columns is None:
            continuous_columns = []

        self.continuous_columns = continuous_columns

        self.columns_order = columns_order
        self.metadata = metadata
        self.continous_transformer = MultiModalNumberTransformer()
        self.categorical_transformer = LabelEncoder()
        self.columns = None

    def fit_transform(self, data, fitting=True):
        """Transform human-readable data into TGAN numerical features.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        num_cols = data.shape[1]
        self.columns = data.columns

        transformed_data = {}
        details = {}

        for col in self.columns_order:
            if col in self.continuous_columns:
                logger.info("Encoding continuous variable \"{}\"...".format(col))

                column_data = data[col].values.reshape([-1, 1])
                features, probs, model, n_modes = self.continous_transformer.transform(column_data)
                transformed_data[col] = np.concatenate((features, probs), axis=1)

                if fitting:
                    details[col] = {
                        "type": "continuous",
                        "n": n_modes,
                        "transform": model,
                    }
            else:
                logger.info("Encoding categorical variable \"{}\"...".format(col))

                column_data = data[col].astype(str).values
                features = self.categorical_transformer.fit_transform(column_data)
                transformed_data[col] = features.reshape([-1, 1])

                if fitting:
                    mapping = self.categorical_transformer.classes_
                    details[col] = {
                        "type": "category",
                        "mapping": mapping,
                        "n": mapping.shape[0],
                    }

        if fitting:
            metadata = {
                "num_features": num_cols,
                "details": details,
            }
            check_metadata(metadata)
            self.metadata = metadata

        return transformed_data

    def transform(self, data):
        """Transform the given dataframe without generating new metadata.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        return self.fit_transform(data, fitting=False)

    def fit(self, data):
        """Initialize the internal state of the object using :attr:`data`.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        self.fit_transform(data)

    def reverse_transform(self, data):
        """Transform DATGAN numerical features back into human-readable data.

        Args:
            data(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame: Model features

        """
        table = []

        for col in self.columns:
            column_data = data[col]
            column_metadata = self.metadata['details'][col]

            if column_metadata['type'] == 'continuous':
                column = self.continous_transformer.inverse_transform(column_data, column_metadata)

            if column_metadata['type'] == 'category':
                self.categorical_transformer.classes_ = column_metadata['mapping']
                column = self.categorical_transformer.inverse_transform(
                    column_data.ravel().astype(np.int32))

            table.append(column)

        result = pd.DataFrame(dict(enumerate(table)))
        result.columns = self.columns
        return result