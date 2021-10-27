import numpy as np
import tensorflow as tf

from tensorpack import LayerNorm, Dropout, FullyConnected
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary

from modules.datgan.models.DATSGANModel import DATSGANModel


class DATDRAGANModel(DATSGANModel):
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
        num_gen_feature=100,
        num_dis_layers=1,
        num_dis_hidden=100,
        optimizer='AdamOptimizer',
        lambda_=10,
        training=True
    ):
        super().__init__(metadata, dag, batch_size, z_dim, noise, l2norm, learning_rate,
                         num_gen_rnn, num_gen_feature, num_dis_layers, num_dis_hidden,
                         optimizer, training)
        """Initialize the object, set arguments as attributes."""
        self.lambda_ = lambda_

    def get_perturbed_batch(self, minibatch):

        noise = tf.random_normal(minibatch.shape, mean=0.0, stddev=0.2)
        return minibatch + noise
        """
        return minibatch + 0.5 * tf.math.multiply(tf.math.reduce_std(minibatch, axis=0),
                                                  np.random.random(minibatch.shape))
        """

    def build_losses(self, vecs_real, vecs_fake):
        r"""
        D and G play two-player minimax game with value function :math:`V(G,D)`.

        .. math::

            min_G max_D V(D, G) = IE_{x \sim p_{data}} [log D(x)] + IE_{z \sim p_{fake}}
                [log (1 - D(G(z)))]

        Args:
            logits_real (tensorflow.Tensor): discrim logits from real samples.
            logits_fake (tensorflow.Tensor): discrim logits from fake samples from generator.
            extra_g(float):
            l2_norm(float): scale to apply L2 regularization.

        Returns:
            None

        """

        kl = self.kl_loss(vecs_real, vecs_fake)

        # Transform list of tensors into a concatenated tensor
        vecs_real = tf.concat(vecs_real, axis=1)
        vecs_fake = tf.concat(vecs_fake, axis=1)

        """
        vecs_real_p = self.get_perturbed_batch(vecs_real)
        diff = vecs_real_p - vecs_real

        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)

        vecs_interp = vecs_real + alpha*diff
        """

        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)

        # Compute diff between real and fake data
        vecs_interp = vecs_real + alpha * (vecs_fake - vecs_real)

        with tf.variable_scope('discrim'):
            logits_real = self.discriminator(vecs_real)
            logits_fake = self.discriminator(vecs_fake)
            logits_interp = self.discriminator(vecs_interp)

        with tf.name_scope("GAN_loss"):
            score_real = tf.sigmoid(logits_real)
            score_fake = tf.sigmoid(logits_fake)
            tf.summary.histogram('score-real', score_real)
            tf.summary.histogram('score-fake', score_fake)

            with tf.name_scope("discrim"):
                d_loss_pos = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=logits_real,
                        labels=tf.ones_like(logits_real)) * 0.7 + tf.random_uniform(
                            tf.shape(logits_real),
                            maxval=0.3
                    ),
                    name='loss_real'
                )

                d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss_fake')

                d_pos_acc = tf.reduce_mean(
                    tf.cast(score_real > 0.5, tf.float32), name='accuracy_real')

                d_neg_acc = tf.reduce_mean(
                    tf.cast(score_fake < 0.5, tf.float32), name='accuracy_fake')

                d_loss = 0.5 * d_loss_pos + 0.5 * d_loss_neg

                # the gradient penalty loss
                gradients = tf.gradients(logits_interp, vecs_interp)[0]
                red_idx = list(range(1, vecs_interp.shape.ndims))
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=red_idx))
                gradients_rms = tf.sqrt(tf.reduce_mean(tf.square(slopes)), name='gradient_rms')
                gradient_penalty = tf.reduce_mean(tf.square(slopes - 1), name='gradient_penalty')

                self.d_loss = tf.add(d_loss, self.lambda_*gradient_penalty, name='loss')

            with tf.name_scope("gen"):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake)))

                g_loss = tf.identity(g_loss, name='loss')
                extra_g = tf.identity(kl, name='klloss')
                self.g_loss = tf.identity(g_loss + extra_g, name='final-g-loss')

            add_moving_summary(
                g_loss, extra_g, self.g_loss, self.d_loss, gradient_penalty, gradients_rms, d_pos_acc, d_neg_acc, decay=0.)