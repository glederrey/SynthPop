import numpy as np
import tensorflow as tf

from tensorpack import BatchNorm, Dropout, FullyConnected, InputDesc, ModelDescBase
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized
from tensorpack.utils import logger

import networkx as nx


class GraphBuilder(ModelDescBase):
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
        training=True,
        one_hot=None,
        structure=None
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
        self.one_hot = one_hot
        self.structure = structure

        if structure not in ["TGAN", "simple", "complex"]:
            raise ValueError("Wrong value for variable named 'structure'!")

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

    def build_losses(self, logits_real, logits_fake, extra_g=0, l2_norm=0.00001):
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
                        tf.contrib.layers.l2_regularizer(l2_norm),
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim"))

                self.d_loss = tf.identity(d_loss, name='loss')

            with tf.name_scope("gen"):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake))) + \
                    tf.contrib.layers.apply_regularization(
                        tf.contrib.layers.l2_regularizer(l2_norm),
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen'))

                g_loss = tf.identity(g_loss, name='loss')
                extra_g = tf.identity(extra_g, name='klloss')
                self.g_loss = tf.identity(g_loss + extra_g, name='final-g-loss')

            add_moving_summary(
                g_loss, extra_g, self.g_loss, self.d_loss, d_pos_acc, d_neg_acc, decay=0.)

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
                inputs.append(
                    InputDesc(tf.float32,
                              (self.batch_size, 1),
                              'input_{}_value'.format(col))
                )

                if self.one_hot:
                    inputs.append(
                        InputDesc(tf.int32,
                                  (self.batch_size, 1),
                                  'input_{}_cluster'.format(col)
                                  )
                    )
                else:
                    gaussian_components = col_info['n']

                    inputs.append(
                        InputDesc(tf.float32,
                                  (self.batch_size, gaussian_components),
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
            # LSTM cell
            cell = tf.nn.rnn_cell.LSTMCell(self.num_gen_rnn)

            # Some variables
            ptr = 0
            outputs = []
            states = {}

            inputs = []
            attentions = []
            name_to_id = {}

            input = None
            state = None
            attention = None

            # Go through all variables
            for col_id, col in enumerate(self.metadata['details'].keys()):
                logger.info("\033[91mCreating cell for {} (in-edges: {})".format(col, len(in_edges[col])))

                # Get info
                col_info = self.metadata['details'][col]
                name_to_id[col] = col_id

                if len(in_edges[col]) <= 1:
                    # Standard procedure as for the TGAN

                    # Get the inputs, attention and state vector in function of the number of in edges
                    if len(in_edges[col]) == 0:
                        input = tf.get_variable(name='go_{}'.format(col), shape=(1, self.num_gen_feature))
                        input = tf.tile(input, [self.batch_size, 1])
                        attention = tf.zeros(shape=(self.batch_size, self.num_gen_rnn), dtype='float32')
                        # LSTM state
                        state = cell.zero_state(self.batch_size, dtype='float32')
                    else:
                        id_ = name_to_id[in_edges[col][0]]
                        input = inputs[id_]
                        attention = attentions[id_]
                        # LSTM state
                        state = states[in_edges[col][0]][-1]

                    # Concat the input with the random variable z
                    input = tf.concat([input, z], axis=1)

                    # Compute the previous states
                    ancestors = nx.ancestors(self.dag, col)
                    prev_states = []
                    for n in self.dag.nodes:
                        if n in ancestors:
                            for s in states[n]:
                                prev_states.append(s[1])

                    [tmp_attention, tmp_states, tmp_inputs,
                     tmp_outputs, ptr] = self.create_cell(cell, z, col, col_info, input,
                                                          attention, prev_states, state, ptr)

                    # Add the input to the list of inputs
                    inputs.append(tmp_inputs)

                    # Add the attention to the list of attentions
                    attentions.append(tmp_attention)

                    # Add the state to the list of states
                    states[col] = tmp_states

                    # Add the list of outputs to the outputs
                    for o in tmp_outputs:
                        outputs.append(o)

                else:
                    # Compute the previous states
                    ancestors = nx.ancestors(self.dag, col)
                    prev_states = []
                    for n in self.dag.nodes:
                        if n in ancestors:
                            for s in states[n]:
                                prev_states.append(s[1])

                    # Go through all in edges to get input, attention and state
                    miLSTM_states = []
                    miLSTM_inputs = []
                    miLSTM_attentions = []
                    for name in in_edges[col]:
                        id_ = name_to_id[name]
                        miLSTM_inputs.append(inputs[id_])
                        miLSTM_attentions.append(attentions[id_])
                        # LSTM state
                        miLSTM_states.append(states[name][-1])

                    # Concatenate the inputs, attention and states
                    with tf.variable_scope("%02d" % ptr):
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

                    ptr += 1

                    # Concat the input with the random variable z
                    input = tf.concat([input, z], axis=1)

                    # Final LSTM cell
                    [tmp_attention, tmp_states, tmp_inputs,
                     tmp_outputs, ptr] = self.create_cell(cell, z, col, col_info, input,
                                                          attention, prev_states, state, ptr)

                    # Add the input to the list of inputs
                    inputs.append(tmp_inputs)

                    # Add the attention to the list of attentions
                    attentions.append(tmp_attention)

                    # Add the state to the list of states
                    states[col] = tmp_states

                    # Add the list of outputs to the outputs
                    for o in tmp_outputs:
                        outputs.append(o)

        return outputs

    def create_cell(self, cell, z, col, col_info, inputs, attention, prev_states, state, ptr):
        """
        Function that create the cells for the generator.
        """

        tmp_states = []
        tmp_outputs = []
        tmp_attention = None
        tmp_input = None

        # create the cell(s)
        if col_info['type'] == 'continuous':
            output, state = cell(tf.concat([inputs, attention], axis=1), state)
            prev_states.append(state[1])
            tmp_states.append(state)

            with tf.variable_scope("%02d" % ptr):
                h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)
                w = FullyConnected('FC2', h, 1, nl=tf.tanh)
                tmp_outputs.append(w)
                tmp_input = h
                #tmp_input = FullyConnected('FC3', w, self.num_gen_feature, nl=tf.identity)
                attw = tf.get_variable("attw", shape=(len(prev_states), 1, 1))
                attw = tf.nn.softmax(attw, axis=0)
                tmp_attention = tf.reduce_sum(tf.stack(prev_states, axis=0) * attw, axis=0)

            ptr += 1

            tmp_input = tf.concat([tmp_input, z], axis=1)
            output, state = cell(tf.concat([tmp_input, tmp_attention], axis=1), state)
            prev_states.append(state[1])
            tmp_states.append(state)

            with tf.variable_scope("%02d" % ptr):
                h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)
                w = FullyConnected('FC2', h, col_info['n'], nl=tf.nn.softmax)
                tmp_outputs.append(w)

                if self.one_hot:
                    one_hot = tf.one_hot(tf.argmax(w, axis=1), col_info['n'])
                    # TGAN passes probabilities (w) instead of one_hot here
                    tmp_input = FullyConnected('FC3', one_hot, self.num_gen_feature, nl=tf.identity)
                else:
                    tmp_input = FullyConnected('FC3', w, self.num_gen_feature, nl=tf.identity)

                attw = tf.get_variable("attw", shape=(len(prev_states), 1, 1))
                attw = tf.nn.softmax(attw, axis=0)
                tmp_attention = tf.reduce_sum(tf.stack(prev_states, axis=0) * attw, axis=0)

            ptr += 1

        elif col_info['type'] == 'category':
            output, state = cell(tf.concat([inputs, attention], axis=1), state)
            prev_states.append(state[1])
            tmp_states.append(state)

            with tf.variable_scope("%02d" % ptr):
                h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)
                w = FullyConnected('FC2', h, col_info['n'], nl=tf.nn.softmax)
                tmp_outputs.append(w)

                one_hot = tf.one_hot(tf.argmax(w, axis=1), col_info['n'])
                tmp_input = FullyConnected('FC3', one_hot, self.num_gen_feature, nl=tf.identity)

                attw = tf.get_variable("attw", shape=(len(prev_states), 1, 1))
                attw = tf.nn.softmax(attw, axis=0)
                tmp_attention = tf.reduce_sum(tf.stack(prev_states, axis=0) * attw, axis=0)

            ptr += 1

        else:
            raise ValueError(
                "self.metadata['details'][{}]['type'] must be either `category` or "
                "`continuous`. Instead it was {}.".format(col, col_info['type'])
            )

        return tmp_attention, tmp_states, tmp_input, tmp_outputs, ptr

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
        M = FullyConnected('fc_diversity', l, n_kernel * kernel_dim, nl=tf.identity)
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
        logits = tf.concat(vecs, axis=1)
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
                logits = BatchNorm('bn', logits, center=True, scale=False)
                logits = Dropout(logits)
                logits = tf.nn.leaky_relu(logits)

        return FullyConnected('dis_fc_top', logits, 1, nl=tf.identity)

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

    def build_graph(self, *inputs):
        """Build the whole graph.

        Args:
            inputs(list[tensorflow.Tensor]):

        Returns:
            None

        """
        z = tf.random_normal(
            [self.batch_size, self.z_dim], name='z_train')

        z = tf.placeholder_with_default(z, [None, self.z_dim], name='z')

        with tf.variable_scope('gen'):
            vecs_gen = self.generator(z)

            vecs_denorm = []
            ptr = 0
            # Go through all variables
            for col_id, col in enumerate(self.metadata['details'].keys()):
                # Get info
                col_info = self.metadata['details'][col]

                if col_info['type'] == 'category':
                    t = tf.argmax(vecs_gen[ptr], axis=1)
                    t = tf.cast(tf.reshape(t, [-1, 1]), 'float32')
                    vecs_denorm.append(t)
                    ptr += 1

                elif col_info['type'] == 'continuous':
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1

                    if self.one_hot:
                        t = tf.argmax(vecs_gen[ptr], axis=1)
                        t = tf.cast(tf.reshape(t, [-1, 1]), 'float32')
                        vecs_denorm.append(t)
                    else:
                        vecs_denorm.append(vecs_gen[ptr])

                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`continuous`. Instead it was {}.".format(col_id, col_info['type'])
                    )

            a = tf.identity(tf.concat(vecs_denorm, axis=1), name='gen')

        vecs_pos = []
        ptr = 0
        # Go through all variables
        for col_id, col in enumerate(self.metadata['details'].keys()):
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'category':
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])
                noise_input = one_hot

                if self.training:
                    noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=self.noise)
                    noise_input = (one_hot + noise) / tf.reduce_sum(
                        one_hot + noise, keepdims=True, axis=1)

                vecs_pos.append(noise_input)
                ptr += 1

            elif col_info['type'] == 'continuous':
                # continuous value in the mixture
                vecs_pos.append(inputs[ptr])
                ptr += 1

                if self.one_hot:
                    # one-hot encoding for the mixture
                    one_hot = tf.one_hot(tf.cast(tf.reshape(inputs[ptr], [-1]), tf.int32), col_info['n'])
                    noise_input = one_hot

                    if self.training:
                        noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=self.noise)
                        noise_input = (one_hot + noise) / tf.reduce_sum(
                            one_hot + noise, keepdims=True, axis=1)

                    vecs_pos.append(noise_input)
                else:
                    vecs_pos.append(inputs[ptr])

                ptr += 1

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`continuous`. Instead it was {}.".format(col_id, col_info['type'])
                )

        KL = 0.0
        ptr = 0
        if self.training:
            # Go through all variables
            for col_id, col in enumerate(self.metadata['details'].keys()):
                # Get info
                col_info = self.metadata['details'][col]

                if col_info['type'] == 'category':
                    dist = tf.reduce_sum(vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(dist)

                    real = tf.reduce_sum(vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(real)
                    KL += self.compute_kl(real, dist)
                    ptr += 1

                elif col_info['type'] == 'continuous':
                    ptr += 1
                    dist = tf.reduce_sum(vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(dist)

                    real = tf.reduce_sum(vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(real)
                    KL += self.compute_kl(real, dist)

                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`continuous`. Instead it was {}.".format(col_id, col_info['type'])
                    )

        with tf.variable_scope('discrim'):
            discrim_pos = self.discriminator(vecs_pos)
            discrim_neg = self.discriminator(vecs_gen)

        self.build_losses(discrim_pos, discrim_neg, extra_g=KL, l2_norm=self.l2norm)
        self.collect_variables()

    def _get_optimizer(self):
        if self.optimizer == 'AdamOptimizer':
            return tf.train.AdamOptimizer(self.learning_rate, 0.5)

        elif self.optimizer == 'AdadeltaOptimizer':
            return tf.train.AdadeltaOptimizer(self.learning_rate, 0.95)

        else:
            return tf.train.GradientDescentOptimizer(self.learning_rate)