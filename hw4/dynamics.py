import tensorflow as tf
import numpy as np
import random

EPSILON = 1e-10

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        scope = 'dynamics_model'
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization
        # number of iterations per aggregation loop (data)
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess

        # input is a state, action concatenation
        input_size = env.observation_space.shape[0] + env.action_space.shape[0]
        output_size = env.observation_space.shape[0]
        self.input_placeholder = tf.placeholder(tf.float32, [None, input_size])

        self.observed_deltas_placeholder = tf.placeholder(tf.float32, [None, output_size])

        self.predicted_deltas = build_mlp(self.input_placeholder,
                                          output_size,
                                          scope,
                                          n_layers=n_layers,
                                          size=size,
                                          activation=activation,
                                          output_activation=output_activation)

        loss = tf.nn.l2_loss(self.predicted_deltas - self.observed_deltas_placeholder)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states
        and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        # TODO NTHOMAS - am I running K iterations here of the update op? yeah I guess... And resample
        # from the dataset each time?

        # we will be sampling from these possible indexes
        indexes_to_sample = range(len(data['states']))
        for i in range(self.iterations):
            indexes = random.sample(indexes_to_sample, self.batch_size)

            states = data['states'].take(indexes)
            normalized_states = (states - self.mean_obs) / (self.std_obs + EPSILON)
            next_states = data['next_states'].take(indexes)

            deltas = next_states - states
            normalized_deltas = (deltas - self.mean_deltas) / (self.std_deltas + EPSILON)

            actions = data['actions'].take(indexes)
            normalized_actions = (actions - self.mean_action) / (self.std_action + EPSILON)

            input_state_actions = np.hstack((normalized_states, normalized_actions))
            feed_dict = {self.input_placeholder: input_state_actions,
                         self.observed_deltas_placeholder: normalized_deltas}
            self.update_op.run(feed_dict=feed_dict)


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions
        and return the (unnormalized) next states as predicted by using the model """
        feed_dict = {self.input_placeholder: np.hstack([states, actions])}
        predicted_deltas = self.sess.run(self.predicted_deltas, feed_dict=feed_dict)
        unnormalized_deltas = predicted_deltas * self.std_deltas + self.mean_deltas
        return states + unnormalized_deltas
