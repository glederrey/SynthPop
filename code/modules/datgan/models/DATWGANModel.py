import numpy as np
import tensorflow as tf

from tensorpack import LayerNorm, Dropout, FullyConnected
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary

from modules.datgan.models.DATSGANModel import DATSGANModel

class DATWGANModel(DATSGANModel):
    """
    Main model for DATGAN.

    Args:
        None

    Attributes:

    """

    def __init__(
        self,
        metadata,
        dag,
        batch_size=200,
        z_dim=200,
        noise=0.2,
        l2norm=0.00001,
        learning_rate=0.001,
        num_gen_rnn=100,
        num_gen_hidden=50,
        num_dis_layers=1,
        num_dis_hidden=100,
        noisy_training='WI'
    ):
        super().__init__(metadata, dag, batch_size, z_dim, noise, l2norm, learning_rate,
                         num_gen_rnn, num_gen_hidden, num_dis_layers, num_dis_hidden, noisy_training)
        """Initialize the object, set arguments as attributes."""

    def build_losses(self, vecs_real, vecs_fake):
        """
        WGAN loss

        Args:
            vecs_real (tensorflow.Tensor): discrim logits from real samples.
            vecs_fake (tensorflow.Tensor): discrim logits from fake samples from generator.

        Returns:
            None
        """

        kl = self.kl_loss(vecs_real, vecs_fake)

        # Transform list of tensors into a concatenated tensor
        vecs_real = tf.concat(vecs_real, axis=1)
        vecs_fake = tf.concat(vecs_fake, axis=1)

        with tf.variable_scope('discrim'):
            d_logit_real = self.discriminator(vecs_real)
            d_logit_fake = self.discriminator(vecs_fake)

        with tf.name_scope("GAN_loss"):
            self.d_loss = tf.reduce_mean(d_logit_fake - d_logit_real, name='d_loss')
            self.g_loss = tf.negative(tf.reduce_mean(d_logit_fake), name='g_loss')
            kl = tf.identity(kl, name='kl_div')
            add_moving_summary(self.d_loss, self.g_loss, kl)

            self.g_loss = tf.add(self.g_loss, kl)

    def _get_optimizer(self):
        return tf.train.RMSPropOptimizer(self.learning_rate)
