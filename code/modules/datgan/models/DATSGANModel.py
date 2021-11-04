import numpy as np
import tensorflow as tf

from tensorpack import BatchNorm, Dropout, FullyConnected, InputDesc, ModelDescBase
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized
from tensorpack.utils import logger

import networkx as nx


class DATSGANModel(ModelDescBase):
    """
    Main model for DATSGAN.

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
        training=True
    ):
        """Initialize the object, set arguments as attributes."""
        self.metadata = metadata
        self.dag = dag
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_feature = num_gen_feature
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.optimizer = optimizer
        self.training = training

    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        """
        Assign generator and discriminator variables from their scopes.

        Args:
            g_scope(str): Scope for the generator.
            d_scope(str): Scope for the discriminator.

        Raises:
            ValueError: If any of the assignments fails or the collections are empty.

        """
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)

        if not (self.g_vars or self.d_vars):
            raise ValueError('There are no variables defined in some of the given scopes')

    @memoized
    def get_optimizer(self):
        """Return optimizer of base class."""
        return self._get_optimizer()

    def inputs(self):
        """Return metadata about entry data.

        Returns:
            list[tensorpack.InputDesc]

        Raises:
            ValueError: If any of the elements in self.metadata['details'] has an unsupported
                        value in the `type` key.

        """
        inputs = []

        for col in self.metadata['details'].keys():
            col_info = self.metadata['details'][col]
            if col_info['type'] == 'continuous':

                n_modes = col_info['n']

                inputs.append(
                    InputDesc(tf.float32,
                              (self.batch_size, n_modes),
                              'input_{}_value'.format(col))
                )

                inputs.append(
                    InputDesc(tf.float32,
                              (self.batch_size, n_modes),
                              'input_{}_cluster'.format(col)
                              )
                )

            elif col_info['type'] == 'category':
                inputs.append(
                    InputDesc(tf.int32,
                              (self.batch_size, 1),
                              'input_{}'.format(col))
                              )

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`continuous`. Instead it was {}.".format(col, col_info['type'])
                )

        return inputs

    def generator(self, z):
        r"""Build generator graph.

        We generate a numerical variable in 2 steps. We first generate the value scalar
        :math:`v_i`, then generate the cluster vector :math:`u_i`. We generate categorical
        feature in 1 step as a probability distribution over all possible labels.

        The output and hidden state size of LSTM is :math:`n_h`. The input to the LSTM in each
        step :math:`t` is the random variable :math:`z`, the previous hidden vector :math:`f_{t−1}`
        or an embedding vector :math:`f^{\prime}_{t−1}` depending on the type of previous output,
        and the weighted context vector :math:`a_{t−1}`. The random variable :math:`z` has
        :math:`n_z` dimensions.
        Each dimension is sampled from :math:`\mathcal{N}(0, 1)`. The attention-based context
        vector at is a weighted average over all the previous LSTM outputs :math:`h_{1:t}`.
        So :math:`a_t` is a :math:`n_h`-dimensional vector.
        We learn a attention weight vector :math:`α_t \in \mathbb{R}^t` and compute context as

        .. math::
            a_t = \sum_{k=1}^{t} \frac{\textrm{exp}  {\alpha}_{t, j}}
                {\sum_{j} \textrm{exp}  \alpha_{t,j}} h_k.

        We set :math: `a_0` = 0. The output of LSTM is :math:`h_t` and we project the output to
        a hidden vector :math:`f_t = \textrm{tanh}(W_h h_t)`, where :math:`W_h` is a learned
        parameter in the network. The size of :math:`f_t` is :math:`n_f` .
        We further convert the hidden vector to an output variable.

        * If the output is the value part of a continuous variable, we compute the output as
          :math:`v_i = \textrm{tanh}(W_t f_t)`. The hidden vector for :math:`t + 1` step is
          :math:`f_t`.

        * If the output is the cluster part of a continuous variable, we compute the output as
          :math:`u_i = \textrm{softmax}(W_t f_t)`. The feature vector for :math:`t + 1` step is
          :math:`f_t`.

        * If the output is a discrete variable, we compute the output as
          :math:`d_i = \textrm{softmax}(W_t f_t)`. The hidden vector for :math:`t + 1` step is
          :math:`f^{\prime}_{t} = E_i [arg_k \hspace{0.25em} \textrm{max} \hspace{0.25em} d_i ]`,
          where :math:`E \in R^{|D_i|×n_f}` is an embedding matrix for discrete variable
          :math:`D_i`.

        * :math:`f_0` is a special vector :math:`\texttt{<GO>}` and we learn it during the
          training.

        Args:
            z:

        Returns:
            list[tensorflow.Tensor]: Output

        Raises:
            ValueError: If any of the elements in self.metadata['details'] has an unsupported
                        value in the `type` key.

        """

        # Compute the in_edges of the dag
        in_edges = {}

        for n in self.dag.nodes:
            in_edges[n] = []
            for edge in self.dag.in_edges:
                if edge[1] == n:
                    in_edges[n].append(edge[0])

        # Create the NN structure
        with tf.variable_scope('LSTM'):

            # Some variables
            outputs = []
            states = {}

            inputs = []
            attentions = []
            name_to_id = {}

            #zero_input = tf.get_variable(name='zero_input', shape=(1, self.num_gen_feature))

            # Go through all variables
            for col_id, col in enumerate(self.metadata['details'].keys()):

                cell = tf.nn.rnn_cell.LSTMCell(self.num_gen_rnn, name=col)

                ancestors = nx.ancestors(self.dag, col)

                info_ = "\033[91mCreating cell for {} (in-edges: {}; ancestors: {})".format(col, len(in_edges[col]),
                                                                                            len(ancestors))
                logger.info(info_)

                # Get info
                col_info = self.metadata['details'][col]
                name_to_id[col] = col_id

                input = None
                attention = None
                state = None
                ancestor_states = None

                if len(in_edges[col]) <= 1:
                    # Standard procedure as for the TGAN

                    # Get the inputs, attention and state vector in function of the number of in edges
                    if len(in_edges[col]) == 0:
                        input = tf.get_variable(name='zero.input-{}'.format(col), shape=(1, self.num_gen_feature))
                        input = tf.tile(input, [self.batch_size, 1])
                        #input = tf.tile(zero_input, [self.batch_size, 1])
                        attention = tf.zeros(shape=(self.batch_size, self.num_gen_rnn), dtype='float32', name='zero.attention-{}'.format(col))
                        # LSTM state
                        state = cell.zero_state(self.batch_size, dtype='float32')
                    else:
                        id_ = name_to_id[in_edges[col][0]]
                        input = inputs[id_]
                        attention = attentions[id_]
                        # LSTM state
                        state = states[in_edges[col][0]]

                    # Compute the previous states
                    ancestor_states = []
                    for n in self.dag.nodes:
                        if n in ancestors:
                            ancestor_states.append(states[n][-1])
                else:
                    # Compute the previous states
                    ancestors = nx.ancestors(self.dag, col)
                    ancestor_states = []
                    for n in self.dag.nodes:
                        if n in ancestors:
                            ancestor_states.append(states[n][-1])

                    # Go through all in edges to get input, attention and state
                    miLSTM_states = []
                    miLSTM_inputs = []
                    miLSTM_attentions = []
                    for name in in_edges[col]:
                        id_ = name_to_id[name]
                        miLSTM_inputs.append(inputs[id_])
                        miLSTM_attentions.append(attentions[id_])
                        # LSTM state
                        miLSTM_states.append(states[name])

                    # Concatenate the inputs, attention and states
                    with tf.variable_scope("concat-{}".format(col)):
                        # FC for inputs
                        tmp = tf.concat(miLSTM_inputs, axis=1)
                        tmp_fc = FullyConnected('FC_inputs', tmp, self.num_gen_feature, nl=None)
                        input = tmp_fc

                        # FC for attentions
                        tmp = tf.concat(miLSTM_attentions, axis=1)
                        tmp_fc = FullyConnected('FC_attentions', tmp, self.num_gen_rnn, nl=None)
                        attention = tmp_fc

                        # FC for states
                        tmp_states = []
                        # miLSTM_states is a list of list of tuples
                        for j in range(len(miLSTM_states[0])):
                            tmp = []
                            for i in range(len(miLSTM_states)):
                                tmp.append(miLSTM_states[i][j])

                            tmp_fc = FullyConnected('FC_lstm_state_{}'.format(j), tf.concat(tmp, axis=1),
                                                    self.num_gen_rnn, nl=None)

                            tmp_states.append(tmp_fc)

                        state = tf.nn.rnn_cell.LSTMStateTuple(tmp_states[0], tmp_states[1])

                # Concat the input with the random variable z
                # MULTI NOISE
                input = tf.concat([input, z[col_id]], axis=1)

                # ONE NOISE
                # input = tf.concat([input, z], axis=1)

                [new_attention, new_state, new_inputs, new_outputs] = self.create_cell(cell, col, col_info, input,
                                                                                        attention, ancestor_states,
                                                                                        state)

                # Add the input to the list of inputs
                inputs.append(new_inputs)

                # Add the attention to the list of attentions
                attentions.append(new_attention)

                # Add the state to the list of states
                states[col] = new_state

                # Add the list of outputs to the outputs
                for o in new_outputs:
                    outputs.append(o)

        return outputs

    def create_cell(self, cell, col, col_info, inputs, attention, ancestor_states, state):
        """
        Function that create the cells for the generator.
        """

        # Use the LSTM cell
        output, state = cell(tf.concat([inputs, attention], axis=1), state)
        ancestor_states.append(state[1])
        new_states = state
        new_outputs = []
        with tf.variable_scope(col):
            h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)

            # For cont. var, we need to get the probability and the values
            if col_info['type'] == 'continuous':
                w_val = FullyConnected('FC2_val', h, col_info['n'], nl=tf.tanh)
                w_prob = FullyConnected('FC2_prob', h, col_info['n'], nl=tf.nn.softmax)

                # 2 outputs here
                new_outputs.append(w_val)
                new_outputs.append(w_prob)

                w = tf.concat([w_val, w_prob], axis=1)
            # For cat. var, we only need the probability
            elif col_info['type'] == 'category':
                w = FullyConnected('FC2', h, col_info['n'], nl=tf.nn.softmax)
                new_outputs.append(w)

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`continuous`. Instead it was {}.".format(col, col_info['type'])
                )

            new_input = FullyConnected('FC3', w, self.num_gen_feature, nl=tf.identity)
            attw = tf.get_variable("attw", shape=(len(ancestor_states), 1, 1))
            attw = tf.nn.softmax(attw, axis=0, name='softmax-attw')
            new_attention = tf.reduce_sum(tf.stack(ancestor_states, axis=0) * attw, axis=0, name='new-att')

        return new_attention, new_states, new_input, new_outputs

    @staticmethod
    def batch_diversity(l, n_kernel=10, kernel_dim=10):
        r"""Return the minibatch discrimination vector.

        Let :math:`f(x_i) \in \mathbb{R}^A` denote a vector of features for input :math:`x_i`,
        produced by some intermediate layer in the discriminator. We then multiply the vector
        :math:`f(x_i)` by a tensor :math:`T \in \mathbb{R}^{A×B×C}`, which results in a matrix
        :math:`M_i \in \mathbb{R}^{B×C}`. We then compute the :math:`L_1`-distance between the
        rows of the resulting matrix :math:`M_i` across samples :math:`i \in {1, 2, ... , n}`
        and apply a negative exponential:

        .. math::

            cb(x_i, x_j) = exp(−||M_{i,b} − M_{j,b}||_{L_1} ) \in \mathbb{R}.

        The output :math:`o(x_i)` for this *minibatch layer* for a sample :math:`x_i` is then
        defined as the sum of the cb(xi, xj )’s to all other samples:

        .. math::
            :nowrap:

            \begin{aligned}

            &o(x_i)_b = \sum^{n}_{j=1} cb(x_i , x_j) \in \mathbb{R}\\
            &o(x_i) = \Big[ o(x_i)_1, o(x_i)_2, . . . , o(x_i)_B \Big] \in \mathbb{R}^B\\
            &o(X) ∈ R^{n×B}\\

            \end{aligned}

        Note:
            This is extracted from `Improved techniques for training GANs`_ (Section 3.2) by
            Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and
            Xi Chen.

        .. _Improved techniques for training GANs: https://arxiv.org/pdf/1606.03498.pdf

        Args:
            l(tf.Tensor)
            n_kernel(int)
            kernel_dim(int)

        Returns:
            tensorflow.Tensor

        """
        M = FullyConnected('FC_DIVERSITY', l, n_kernel * kernel_dim, nl=tf.identity)
        M = tf.reshape(M, [-1, n_kernel, kernel_dim])
        M1 = tf.reshape(M, [-1, 1, n_kernel, kernel_dim])
        M2 = tf.reshape(M, [1, -1, n_kernel, kernel_dim])
        diff = tf.exp(-tf.reduce_sum(tf.abs(M1 - M2), axis=3))
        return tf.reduce_sum(diff, axis=0)

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
        for i in range(self.num_dis_layers):
            with tf.variable_scope('DISCR_FC_{}'.format(i)):
                if i == 0:
                    logits = FullyConnected(
                        'FC', logits, self.num_dis_hidden, nl=tf.identity,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )

                else:
                    logits = FullyConnected('FC', logits, self.num_dis_hidden, nl=tf.identity)

                logits = tf.concat([logits, self.batch_diversity(logits)], axis=1)
                logits = BatchNorm('BN', logits, center=True, scale=False)
                logits = Dropout(logits)
                logits = tf.nn.leaky_relu(logits)

        return FullyConnected('DISCR_FC_TOP', logits, 1, nl=tf.identity)

    @staticmethod
    def compute_kl(real, pred):
        r"""Compute the Kullback–Leibler divergence, :math:`D_{KL}(\textrm{pred} || \textrm{real})`.

        Args:
            real(tensorflow.Tensor): Real values.
            pred(tensorflow.Tensor): Predicted values.

        Returns:
            float: Computed divergence for the given values.

        """
        return tf.reduce_sum((tf.log(pred + 1e-4) - tf.log(real + 1e-4)) * pred)

    def kl_loss(self, vecs_real, vecs_fake):
        # KL loss
        KL = 0.0
        ptr = 0
        if self.training:
            # Go through all variables
            for col_id, col in enumerate(self.metadata['details'].keys()):
                # Get info
                col_info = self.metadata['details'][col]

                if col_info['type'] == 'continuous':
                    # Skip the value. We only compute the KL on the probability vector
                    ptr += 1

                dist = tf.reduce_sum(vecs_fake[ptr], axis=0)
                dist = dist / tf.reduce_sum(dist)

                real = tf.reduce_sum(vecs_real[ptr], axis=0)
                real = real / tf.reduce_sum(real)
                KL += self.compute_kl(real, dist)
                ptr += 1

        return KL

    def build_graph(self, *inputs):
        """Build the whole graph.

        Args:
            inputs(list[tensorflow.Tensor]):

        Returns:
            None

        """

        # MULTI NOISE
        n_vars = len(self.metadata['details'].keys())
        z = tf.random_normal([n_vars, self.batch_size, self.z_dim], name='z_train')
        z = tf.placeholder_with_default(z, [None, None, self.z_dim], name='z')

        # ONE NOISE
        #z = tf.random_normal([self.batch_size, self.z_dim], name='z_train')
        #z = tf.placeholder_with_default(z, [None, self.z_dim], name='z')

        # Create the output for the model
        with tf.variable_scope('gen'):
            vecs_gen = self.generator(z)

        vecs_output = []
        vecs_fake = []
        vecs_real = []
        ptr = 0
        # Go through all variables
        for col_id, col in enumerate(self.metadata['details'].keys()):
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'category':

                # OUTPUT
                vecs_output.append(vecs_gen[ptr])

                # FAKE
                val = vecs_gen[ptr]

                if self.training:
                    noise = tf.random_uniform(tf.shape(val), minval=0, maxval=self.noise)
                    val = (val + noise) / tf.reduce_sum(val + noise, keepdims=True, axis=1)
                vecs_fake.append(val)

                # REAL
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])

                if self.training:
                    noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=self.noise)
                    one_hot = (one_hot + noise) / tf.reduce_sum(one_hot + noise, keepdims=True, axis=1)
                vecs_real.append(one_hot)

                ptr += 1

            elif col_info['type'] == 'continuous':
                vecs_output.append(vecs_gen[ptr])
                vecs_fake.append(vecs_gen[ptr])
                vecs_real.append(inputs[ptr])
                ptr += 1

                vecs_output.append(vecs_gen[ptr])
                vecs_fake.append(vecs_gen[ptr])
                vecs_real.append(inputs[ptr])
                ptr += 1

        # This weird thing is then used for sampling the generator once it has been trained.
        tf.identity(tf.concat(vecs_output, axis=1), name='output')

        self.build_losses(vecs_real, vecs_fake)
        self.collect_variables()

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

        with tf.variable_scope('discrim'):
            logits_real = self.discriminator(vecs_real)
            logits_fake = self.discriminator(vecs_fake)

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

                d_loss = 0.5 * d_loss_pos + 0.5 * d_loss_neg + \
                    tf.contrib.layers.apply_regularization(
                        tf.contrib.layers.l2_regularizer(self.l2norm),
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim"))

                self.d_loss = tf.identity(d_loss, name='loss')

            with tf.name_scope("gen"):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake))) + \
                    tf.contrib.layers.apply_regularization(
                        tf.contrib.layers.l2_regularizer(self.l2norm),
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen'))

                g_loss = tf.identity(g_loss, name='loss')
                extra_g = tf.identity(kl, name='klloss')
                self.g_loss = tf.identity(g_loss + extra_g, name='final-g-loss')

            add_moving_summary(g_loss, extra_g, self.g_loss, self.d_loss, d_pos_acc, d_neg_acc, decay=0.)


    def _get_optimizer(self):
        if self.optimizer == 'AdamOptimizer':
            return tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)

        elif self.optimizer == 'AdadeltaOptimizer':
            return tf.train.AdadeltaOptimizer(self.learning_rate, 0.95)

        else:
            return tf.train.GradientDescentOptimizer(self.learning_rate)
