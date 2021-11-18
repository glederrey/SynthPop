import numpy as np
import tensorflow as tf

from tensorpack import LayerNorm, Dropout, FullyConnected
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary

from modules.datgan.models.DATSGANModel import DATSGANModel


class DATWGANGPModel(DATSGANModel):
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
        lambda_=10,
        noisy_training='WI'
    ):
        super().__init__(metadata, dag, batch_size, z_dim, noise, l2norm, learning_rate,
                         num_gen_rnn, num_gen_hidden, num_dis_layers, num_dis_hidden, noisy_training)
        """Initialize the object, set arguments as attributes."""
        self.lambda_ = lambda_

    @auto_reuse_variable_scope
    def discriminator(self, vecs):
        r"""Build discriminator.

        We use a :math:`l`-layer fully connected neural network as the discriminator.
        We concatenate :math:`v_{1:n_c}`, :math:`u_{1:n_c}` and :math:`d_{1:n_d}` together as the
        input. We compute the internal layers as

        .. math::
            \begin{aligned}

            f^{(D)}_{1} &= \textrm{LeakyReLU}(\textrm{BN}(W^{(D)}_{1}(v_{1:n_c} \oplus u_{1:n_c}
                \oplus d_{1:n_d})

            f^{(D)}_{1} &= \textrm{LeakyReLU}(\textrm{BN}(W^{(D)}_{i}(f^{(D)}_{i−1} \oplus
                \textrm{diversity}(f^{(D)}_{i−1})))), i = 2:l

            \end{aligned}

        where :math:`\oplus` is the concatenation operation. :math:`\textrm{diversity}(·)` is the
        mini-batch discrimination vector [42]. Each dimension of the diversity vector is the total
        distance between one sample and all other samples in the mini-batch using some learned
        distance metric. :math:`\textrm{BN}(·)` is batch normalization, and
        :math:`\textrm{LeakyReLU}(·)` is the leaky reflect linear activation function. We further
        compute the output of discriminator as :math:`W^{(D)}(f^{(D)}_{l} \oplus \textrm{diversity}
        (f^{(D)}_{l}))` which is a scalar.

        Args:
            vecs(list[tensorflow.Tensor]): List of tensors matching the spec of :meth:`inputs`

        Returns:
            tensorpack.FullyConected: a (b, 1) logits

        """
        logits = tf.identity(vecs)
        with tf.variable_scope('discrim'):
            for i in range(self.num_dis_layers):
                with tf.variable_scope('dis_fc{}'.format(i)):
                    if i == 0:
                        logits = FullyConnected(
                            'fc', logits, self.num_dis_hidden, nl=tf.identity,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                        )

                    else:
                        logits = FullyConnected('fc', logits, self.num_dis_hidden, nl=tf.identity)

                    logits = tf.concat([logits, self.batch_diversity(logits)], axis=1)
                    logits = LayerNorm('ln', logits)
                    logits = Dropout(logits)
                    logits = tf.nn.leaky_relu(logits)

            return FullyConnected('dis_fc_top', logits, 1, nl=tf.identity)

    def build_losses(self, vecs_real, vecs_fake):
        r"""
        WGAN-GP loss

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

        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)

        # Compute diff between real and fake data
        vecs_interp = vecs_real + alpha * (vecs_fake - vecs_real)

        with tf.variable_scope('discrim'):
            d_logit_real = self.discriminator(vecs_real)
            d_logit_fake = self.discriminator(vecs_fake)
            d_logit_interp = self.discriminator(vecs_interp)

        with tf.name_scope("GAN_loss"):

            self.d_loss = tf.reduce_mean(d_logit_fake - d_logit_real, name='d_loss')
            self.g_loss = tf.negative(tf.reduce_mean(d_logit_fake), name='g_loss')

            # the gradient penalty loss
            gradients = tf.gradients(d_logit_interp, vecs_interp)[0]
            red_idx = list(range(1, vecs_interp.shape.ndims))
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=red_idx))
            gradients_rms = tf.sqrt(tf.reduce_mean(tf.square(slopes)), name='gradient_rms')
            gradient_penalty = tf.reduce_mean(tf.square(slopes - 1), name='gradient_penalty')

            kl = tf.identity(kl, name='kl_div')
            add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms, kl)

            self.d_loss = tf.add(self.d_loss, self.lambda_ * gradient_penalty)
            self.g_loss = tf.add(self.g_loss, kl)

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)