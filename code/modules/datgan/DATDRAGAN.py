#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with the model for DATGAN (Directed Acyclic Tabular GAN), based on TGAN.

This module contains two classes:

- :attr:`GraphBuilder`: That defines the graph and implements a Tensorpack compatible API.
- :attr:`TGANModel`: The public API for the model, that offers a simplified interface for the
  underlying operations with GraphBuilder and trainers in order to fit and sample data.
"""
from modules.datgan.models.DATDRAGANModel import DATDRAGANModel
from modules.datgan.DATSGAN import DATSGAN

from modules.datgan.trainer import GANTrainer



class DATDRAGAN(DATSGAN):
    """Main model for DATDRAGAN. (https://arxiv.org/abs/1705.07215)

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
                 num_dis_layers=1, num_dis_hidden=100, optimizer='AdamOptimizer', lambda_=10):

        super().__init__(continuous_columns, output, gpu, max_epoch, steps_per_epoch, save_checkpoints,
                         restore_session, batch_size, z_dim, noise, l2norm, learning_rate, num_gen_rnn,
                         num_gen_feature, num_dis_layers, num_dis_hidden, optimizer)

        self.lambda_ = lambda_

        self.trainer = GANTrainer



    def get_model(self, training=True):
        """Return a new instance of the model."""
        return DATDRAGANModel(
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
            lambda_=self.lambda_,
            training=training
        )