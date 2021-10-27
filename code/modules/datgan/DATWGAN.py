#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with the model for DATGAN (Directed Acyclic Tabular GAN), based on TGAN.

This module contains two classes:

- :attr:`GraphBuilder`: That defines the graph and implements a Tensorpack compatible API.
- :attr:`TGANModel`: The public API for the model, that offers a simplified interface for the
  underlying operations with GraphBuilder and trainers in order to fit and sample data.
"""
from functools import partial
import os
import pickle
import json

from modules.datgan.trainer import SeparateGANTrainer
from modules.datgan.models.DATWGANModel import DATWGANModel
from modules.datgan.DATSGAN import DATSGAN

from tensorpack import BatchData, ModelSaver, QueueInput, SaverRestore
from tensorpack.utils import logger

from modules.datgan.data import Preprocessor, DATGANDataFlow
from modules.datgan.utils import ClipCallback


class DATWGAN(DATSGAN):
    """Main model for DATWGAN.

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

    def __init__(self, continuous_columns, output='output', gpu=None, max_epoch=5, steps_per_epoch=None,
                 save_checkpoints=True, restore_session=True, batch_size=200, z_dim=200, noise=0.2,
                 l2norm=0.00001, learning_rate=1e-3, num_gen_rnn=100, num_gen_feature=100,
                 num_dis_layers=1, num_dis_hidden=100):

        super().__init__(continuous_columns, output, gpu, max_epoch, steps_per_epoch, save_checkpoints,
                         restore_session, batch_size, z_dim, noise, l2norm, learning_rate, num_gen_rnn,
                         num_gen_feature, num_dis_layers, num_dis_hidden, None)

        # We use a separate trainer for the DATWGAN to train the discirminator more often
        self.trainer = partial(SeparateGANTrainer, g_period=3)

    def get_model(self, training=True):
        """Return a new instance of the model."""
        return DATWGANModel(
            metadata=self.metadata,
            dag=self.dag,
            batch_size=self.batch_size,
            z_dim=self.z_dim,
            noise=self.noise,
            l2norm=self.l2norm,
            learning_rate=self.learning_rate,
            num_gen_rnn=self.num_gen_rnn,
            num_gen_feature=self.num_gen_feature,
            num_dis_layers=self.num_dis_layers,
            num_dis_hidden=self.num_dis_hidden,
            optimizer=self.optimizer,
            training=training
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

        self.model = self.get_model(training=True)

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

        callbacks = []
        if self.save_checkpoints:
            callbacks.append(ModelSaver(checkpoint_dir=self.model_dir))

        callbacks.append(ClipCallback())

        trainer.train_with_defaults(
            callbacks=callbacks,
            steps_per_epoch=self.steps_per_epoch,
            max_epoch=self.max_epoch,
            session_init=session_init,
            starting_epoch=starting_epoch
        )

        self.prepare_sampling()