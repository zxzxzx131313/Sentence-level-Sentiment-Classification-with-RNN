import tensorflow as tf


class BasicRNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):

            new_state = tf.layers.dense(tf.concat([inputs, state], 1), self._num_units, activation=self._activation, use_bias=True)

        return new_h, new_h


class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            # We start with bias of 1.0 to not reset and not update.
            with tf.variable_scope("gates"):
                value = tf.layers.dense(tf.concat([inputs, state], 1), 2 * self._num_units, activation=tf.sigmoid, use_bias=True, bias_initializer=tf.ones_initializer())
                r, z = tf.split(value=value, num_or_size_splits=2, axis=1)
            
            with tf.variable_scope("candidate"):
                c = tf.layers.dense(tf.concat([inputs, r * state], 1), self._num_units, activation=self._activation, use_bias=True)
            
            new_h = z * state + (1 - z) * c
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            # For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            value = tf.layers.dense(tf.concat([inputs, h], 1), 4 * self._num_units, use_bias=True)
            
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=value, num_or_size_splits=4, axis=1)
            batch_size, embed_unit = inputs.shape

            i_t = tf.sigmoid(i)

            o_t = tf.sigmoid(o)

            f_t = tf.sigmoid(f + self._forget_bias)
            
            j_t = self._activation(j)

            new_c = c * f_t + i_t * j_t

            new_h = self._activation(new_c) * o_t

        return new_h, (new_c, new_h)
