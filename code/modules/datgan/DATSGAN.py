#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with the model for DATGAN (Directed Acyclic Tabular GAN), based on TGAN.

This module contains two classes:

- :attr:`GraphBuilder`: That defines the graph and implements a Tensorpack compatible API.
- :attr:`TGANModel`: The public API for the model, that offers a simplified interface for the
  underlying operations with GraphBuilder and trainers in order to fit and sample data.
"""
import json
import os
import pickle
import tarfile
import numpy as np

from tensorpack import BatchData, ModelSaver, PredictConfig, QueueInput, SaverRestore, SimpleDatasetPredictor
from tensorpack.utils import logger

from modules.datgan.data import Preprocessor, RandomZData, DATGANDataFlow
from modules.datgan.trainer import GANTrainerClipping
from modules.datgan.models.DATSGANModel import DATSGANModel

import networkx as nx

# Graphs for tests
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

class DATSGAN:
    """Main model for DATSGAN.

    Args:
        continuous_columns (list[int]): 0-index list of column indices to be considered continuous.
        output (str, optional): Path to store the model and its artifacts. Defaults to
            :attr:`output`.
        gpu (list[str], optional):Comma separated list of GPU(s) to use. Defaults to :attr:`None`.
        max_epoch (int, optional): Number of epochs to use during training. Defaults to :attr:`5`.
        steps_per_epoch (int, optional): Number of steps to run on each epoch. Defaults to
            :attr:`10000`.
        save_checkpoints(bool, optional): Whether or not to store checkpoints of the model after
            each training epoch. Defaults to :attr:`True`
        restore_session(bool, optional): Whether or not continue training from the last checkpoint.
            Defaults to :attr:`True`.
        batch_size (int, optional): Size of the batch to feed the model at each step. Defaults to
            :attr:`200`.
        z_dim (int, optional): Number of dimensions in the noise input for the generator.
            Defaults to :attr:`100`.
        noise (float, optional): Upper bound to the gaussian noise added to categorical columns.
            Defaults to :attr:`0.2`.
        l2norm (float, optional):
            L2 reguralization coefficient when computing losses. Defaults to :attr:`0.00001`.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to
            :attr:`0.001`.
        num_gen_rnn (int, optional): Defaults to :attr:`400`.
        num_gen_feature (int, optional): Number of features of in the generator. Defaults to
            :attr:`100`
        num_dis_layers (int, optional): Defaults to :attr:`2`.
        num_dis_hidden (int, optional): Defaults to :attr:`200`.
        optimizer (str, optional): Name of the optimizer to use during `fit`,possible values are:
            [`GradientDescentOptimizer`, `AdamOptimizer`, `AdadeltaOptimizer`]. Defaults to
            :attr:`AdamOptimizer`.
    """

    def __init__(
        self, continuous_columns, output='output', gpu=None, max_epoch=5, steps_per_epoch=None,
        save_checkpoints=True, restore_session=True, batch_size=200, z_dim=200, noise=0.2,
        l2norm=0.00001, learning_rate=1e-3, num_gen_rnn=100, num_gen_hidden=50,
        num_dis_layers=1, num_dis_hidden=100, noisy_training='WI'
    ):
        """Initialize object."""
        # Output
        self.continuous_columns = continuous_columns
        self.data_dir = os.path.join(output, 'data')
        self.log_dir = os.path.join(output, 'logs')
        self.model_dir = os.path.join(output, 'model')
        self.output = output

        # DAG
        self.dag = None
        self.var_order = None
        self.n_sources = None

        # Training params
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self.save_checkpoints = save_checkpoints
        self.restore_session = restore_session

        # Model params
        self.model = None
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_hidden = num_gen_hidden
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden

        if gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        self.gpu = gpu

        # What changes between DATGAN and DATWGAN
        self.trainer = GANTrainerClipping

        # Parameters for generating results
        self.noisy_training = noisy_training
        if self.noisy_training not in ['WI', 'WO', 'OR']:
            raise ValueError("'noisy_training' must take value 'WI', 'WO', or 'OR'!")

    def get_model(self):
        """Return a new instance of the model."""
        return DATSGANModel(
            metadata=self.metadata,
            dag=self.dag,
            batch_size=self.batch_size,
            z_dim=self.z_dim,
            noise=self.noise,
            l2norm=self.l2norm,
            learning_rate=self.learning_rate,
            num_gen_rnn=self.num_gen_rnn,
            num_gen_hidden=self.num_gen_hidden,
            num_dis_layers=self.num_dis_layers,
            num_dis_hidden=self.num_dis_hidden,
            noisy_training=self.noisy_training,
        )

    def fit(self, data, dag):
        """Fit the model to the given data.

        Args:
            data(pandas.DataFrame): dataset to fit the model.
            dag(networkx.classes.digraph.DiGraph): DAG for the relations between variables

        Returns:
            None

        """
        self.preprocessor = None
        self.restore_path = os.path.join(self.model_dir, 'checkpoint')

        if self.steps_per_epoch is None:
            self.steps_per_epoch = max(len(data) // self.batch_size, 1)

        # Verify that the DAG has the same number of nodes as the number of variables in the data
        # and that it's indeed a DAG.
        self.dag = dag
        self.verify_dag(data)
        self.var_order = self.get_order_variables()

        if os.path.exists(self.data_dir):
            logger.info("Found preprocessed data")

            # Load preprocessed data
            with open(os.path.join(self.data_dir, 'preprocessed_data.pkl'), 'rb') as f:
                transformed_data = pickle.load(f)
            with open(os.path.join(self.data_dir, 'preprocessor.pkl'), 'rb') as f:
                self.preprocessor = pickle.load(f)

            logger.info("Preprocessed data have been loaded!")
        else:
            # Preprocessing steps
            logger.info("Preprocessing the data!")

            self.preprocessor = Preprocessor(continuous_columns=self.continuous_columns, columns_order=self.var_order)
            transformed_data = self.preprocessor.fit_transform(data)

            # Save the preprocessor and the data
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir, exist_ok=True)
            with open(os.path.join(self.data_dir, 'preprocessed_data.pkl'), 'wb') as f:
                pickle.dump(transformed_data, f)
            with open(os.path.join(self.data_dir, 'preprocessor.pkl'), 'wb') as f:
                pickle.dump(self.preprocessor, f)

            logger.info("Preprocessed data have been saved!")

            # Verification for continuous mixture
            self.plot_continuous_mixtures(data, self.data_dir)

        self.metadata = self.preprocessor.metadata
        dataflow = DATGANDataFlow(transformed_data, self.metadata)
        batch_data = BatchData(dataflow, self.batch_size)
        input_queue = QueueInput(batch_data)

        self.model = self.get_model()

        trainer = self.trainer(
            input=input_queue,
            model=self.model
        )

        # Checking if previous training already exists
        session_init = None
        starting_epoch = 1
        if os.path.isfile(self.restore_path) and self.restore_session:
            logger.info("Found an already existing model. Loading it!")

            session_init = SaverRestore(self.restore_path)
            with open(os.path.join(self.log_dir, 'stats.json')) as f:
                starting_epoch = json.load(f)[-1]['epoch_num'] + 1

        action = 'k' if self.restore_session else None
        logger.set_logger_dir(self.log_dir, action=action)

        callbacks = self.get_callbacks()

        trainer.train_with_defaults(
            callbacks=callbacks,
            steps_per_epoch=self.steps_per_epoch,
            max_epoch=self.max_epoch,
            session_init=session_init,
            starting_epoch=starting_epoch
        )

        self.prepare_sampling()

        # Close the log file from the logger of tensorpack


    def get_callbacks(self):

        callbacks = []
        if self.save_checkpoints:
            callbacks.append(ModelSaver(checkpoint_dir=self.model_dir))

        return callbacks

    def prepare_sampling(self):
        """Prepare model to generate samples."""
        if self.model is None:
            self.model = self.get_model()

        predict_config = PredictConfig(
            session_init=SaverRestore(self.restore_path),
            model=self.model,
            input_names=['z'],
            output_names=['output', 'z'],
        )

        self.simple_dataset_predictor = SimpleDatasetPredictor(
            predict_config,
            RandomZData((self.n_sources, self.batch_size, self.z_dim))
        )

    def sample(self, num_samples, argmax_sampling='NO'):
        """Generate samples from model.

        Args:
            num_samples(int)

        Returns:
            None

        Raises:
            ValueError

        """
        logger.info("Loading Preprocessor!")
        # Load preprocessor
        with open(os.path.join(self.data_dir, 'preprocessor.pkl'), 'rb') as f:
            self.preprocessor = pickle.load(f)

        if argmax_sampling not in ['BO', 'NO', 'OD', 'OC']:
            raise ValueError("'argmax_sampling' must take value 'BO', 'NO', 'OC', or 'OD'!")

        self.preprocessor.set_sampling_technique(argmax_sampling)
        self.metadata = self.preprocessor.metadata

        max_iters = (num_samples // self.batch_size)

        results = []
        for idx, o in enumerate(self.simple_dataset_predictor.get_result()):
            results.append(o[0])
            if idx == max_iters:
                break

        results = np.concatenate(results, axis=0)
        # Reduce results to num_samples
        results = results[:num_samples]

        ptr = 0
        features = {}
        # Go through all variables
        for col_id, col in enumerate(self.metadata['details'].keys()):
            # Get info
            col_info = self.metadata['details'][col]
            if col_info['type'] == 'category':
                features[col] = results[:, ptr:ptr + col_info['n']]
                ptr += col_info['n']

            elif col_info['type'] == 'continuous':

                n_modes = col_info['n']

                val = results[:, ptr:ptr + n_modes]
                ptr += n_modes

                pro = results[:, ptr:ptr + n_modes]
                ptr += n_modes

                features[col] = np.concatenate([val, pro], axis=1)

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`continuous`. Instead it was {}.".format(col_id, col_info['type'])
                )

        return self.preprocessor.reverse_transform(features)[:num_samples].copy()

    def tar_folder(self, tar_name):
        """Generate a tar of :self.output:."""
        with tarfile.open(tar_name, 'w:gz') as tar_handle:
            for root, dirs, files in os.walk(self.output):
                for file_ in files:
                    tar_handle.add(os.path.join(root, file_))

            tar_handle.close()

    @classmethod
    def load(cls, path, name):
        """Load a pretrained model from a given path."""
        with tarfile.open(path + name + '.tar.gz', 'r:gz') as tar_handle:
            destination_dir = os.path.dirname(tar_handle.getmembers()[0].name)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_handle)

        with open('{}/{}.pickle'.format(destination_dir, name), 'rb') as f:
            instance = pickle.load(f)

        instance.prepare_sampling()
        return instance

    def save(self, name, force=False):
        """Save the fitted model in the given path."""
        if os.path.exists(self.output) and not force:
            logger.info('The indicated path already exists. Use `force=True` to overwrite.')
            return

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        model = self.model
        dataset_predictor = self.simple_dataset_predictor

        self.model = None
        self.simple_dataset_predictor = None

        with open('{}/{}.pickle'.format(self.output, name), 'wb') as f:
            pickle.dump(self, f)

        self.model = model
        self.simple_dataset_predictor = dataset_predictor

        self.tar_folder(self.output + name + '.tar.gz')

        logger.info('Model saved successfully.')

    def verify_dag(self, data):
        """
        Verify that the given graph is indeed a dag.

        :return:
        """

        # 1. Verify the type
        if type(self.dag) is not nx.classes.digraph.DiGraph:
            raise TypeError("Provided graph is not from the type \"networkx.classes.digraph."
                            "DiGraph\": {}".format(type(self.dag)))

        # 2. Verify that the graph is indeed a DAG
        if not nx.algorithms.dag.is_directed_acyclic_graph(self.dag):

            cycles = nx.algorithms.cycles.find_cycle(self.dag)

            if len(cycles) > 0:
                raise ValueError("Provided graph is not a DAG. Cycles found: {}".format(cycles))
            else:
                raise ValueError("Provided graph is not a DAG.")

        # 3. Verify that the dag has the correct number of nodes
        if len(self.dag.nodes) != len(data.columns):
            raise ValueError("DAG does not have the same number of nodes ({}) as the number of "
                             "variables in the data ({}).".format(len(self.dag.nodes), len(data.columns)))

    def get_in_edges(self):
        # Get the in_edges
        in_edges = {}

        for n in self.dag.nodes:
            in_edges[n] = []
            for edge in self.dag.in_edges:
                if edge[1] == n:
                    in_edges[n].append(edge[0])

        return in_edges

    def get_order_variables(self):
        """
        Get the order of all the variables in the graph

        :return: list of column names
        """
        # Get the in_edges
        in_edges = self.get_in_edges()

        untreated = set(self.dag.nodes)
        treated = []

        # Get all nodes with 0 in degree
        to_treat = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]

        self.n_sources = len(to_treat)

        while len(untreated) > 0:
            # remove the treated nodes
            for n in to_treat:
                untreated.remove(n)
                treated.append(n)

            to_treat = []
            # Find the edges that are coming from the the treated nodes
            for edge in self.dag.in_edges:

                all_treated = True
                for l in in_edges[edge[1]]:
                    if l not in treated:
                        all_treated = False

                if edge[0] in treated and all_treated and edge[1] not in treated and edge[1] not in to_treat:
                    to_treat.append(edge[1])

        return treated

    def plot_continuous_mixtures(self, data, path):

        for col in self.continuous_columns:

            details = self.preprocessor.metadata['details'][col]

            gmm = details['transform']

            tmp = data[col]

            fig = plt.figure(figsize=(10, 7))
            plt.hist(tmp, 50, density=True, histtype='stepfilled', alpha=0.4, color='gray')

            x = np.linspace(np.min(tmp), np.max(tmp), 1000)

            logprob = gmm.score_samples(x.reshape(-1, 1))
            responsibilities = gmm.predict_proba(x.reshape(-1, 1))
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]
            plt.plot(x, pdf, '-k')
            plt.plot(x, pdf_individual, '--k')

            plt.xlabel('$x$')
            plt.ylabel('$p(x)$')
            plt.title("{} - {} mixtures".format(col, details['n']))
            plt.savefig(path + '/{}.png'.format(col), bbox_inches='tight', facecolor='white')
            plt.close(fig)