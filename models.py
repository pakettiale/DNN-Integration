import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from losses import regression_loss, jacobian, generator_loss
from logdet import logdet
from custom_functions import tf_integrate, triplegaussian, doublegaussian, custom_tanh, cauchy

class RegressiveDNN():
    def __init__(self, dim):
        self.dim = dim
        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.Dense(64, input_shape=(self.dim,), activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

        self.nn = nn
        self.output = nn.output
        self.input  = nn.input
        self.nn.compile(loss=regression_loss,
                   optimizer="adam")

class GenerativeDNN():
    #ToDO integral values, function bounds
    def __init__(self, dim, prior, activation, scope='gen'):
        [self.output, self.input] = build_net(dim, 3, 32, activation, tf.sigmoid, scope)
        self.scope = scope
        self.dim = dim
        self.prior = prior
        self.target_function = None
        self.optimize = None
        self.target_integral = tf.get_variable('no_train/integral', [], dtype=tf.float32, trainable=False)

    def set_target_function(self, target_function):
        self.target_function = target_function

    def loss(self, KL=True):
        probs = self.prior_prob(self.input)
        if KL:
            self.loss = generator_loss(self.output, self.input, self.target_function(self.output), self.target_integral, probs)
        else:
            self.loss = tf.losses.mean_squared_error(self.target_function(self.input), self.output)
        return self.loss

    def prior_prob(self, input):
        return tf.reduce_prod(self.prior.prob(self.input), (-1,), keepdims=True)

    def variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def abs_grad(self):
        return tf.reshape(tf.exp(logdet(jacobian(self.output, self.input))), (-1, 1))

    def density(self):
        return self.prior_prob(self.input)/self.abs_grad()

    def create_trainer(self, learning_rate=0.0001, beta=0.9, KL=True):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta)
        grads_and_vars = optimizer.compute_gradients(self.loss(KL), var_list=self.variables())
        grads = [x[0] for x in grads_and_vars]
        vars  = [x[1] for x in grads_and_vars]
        grads, grad_norm = tf.clip_by_global_norm(grads, 2)
        self.optimize = optimizer.apply_gradients(zip(grads, vars))

    def integrate_function(self, sample_size, function, skip_gen=False, use_exp=True):
        def exp(input):
            if use_exp:
                return tf.exp(input)
            else:
                return input
        session = tf.get_default_session()
        if not skip_gen:
            data = tf.distributions.Uniform(0., 1.,).sample([sample_size, self.dim]).eval()
            integral = session.run(tf_integrate(exp(function(self.output)), self.density()), {self.input: data})
        else:
            data = self.prior.sample([sample_size, self.dim])
            integral = session.run(tf_integrate(exp(function(data)), 1.0))
        return integral

    def set_target_integral(self, value):
        session = tf.get_default_session()
        session.run(self.target_integral.assign(value))

    def train(self, batches, batch_size, epochs):
        data = self.prior.sample([batches, batch_size, self.dim]).eval()
        session = tf.get_default_session()
        lossT = self.loss
        for e in range(1, epochs+1):
            for idx, batch in enumerate(data):
                _, loss = session.run([self.optimize, lossT], {self.input: batch})
                print(f'\rEpoch {e:-4d}: [{idx+1:-4d}/{batches:d}] Loss: {loss:f}', end='')
            print('')

    def sampler(self, sample_size):
        data = self.prior.sample([sample_size, self.dim]).eval()
        session = tf.get_default_session()
        return session.run(self.output, {self.input: data})

def build_net(dim, depth, nodes, layer_activation, output_activation, scope):
    with tf.variable_scope(scope):
        X = tf.placeholder(tf.float32, [None, dim], 'input')
        input = X
        row_dim = dim
        norm_init      = tf.random_normal_initializer(mean=0.0, stddev=0.1*np.sqrt(2/dim))
        nth_norm_init  = tf.random_normal_initializer(mean=0.0, stddev=0.1*np.sqrt(2/nodes))
        const_init = tf.constant_initializer(0.0)
        for i in range(1, depth+1):
            if i == depth:
                W = tf.get_variable(str(i) + '_W', [row_dim, dim])
                b = tf.get_variable(str(i) + '_b', [dim], initializer=const_init)
                X = output_activation(tf.add(tf.matmul(X, W), b), name='output')
            else:
                W = tf.get_variable(str(i) + '_W', [row_dim, nodes])
                b = tf.get_variable(str(i) + '_b', [nodes], initializer=const_init)
                X = layer_activation(tf.add(tf.matmul(X, W), b))
            row_dim = nodes
            norm_init = nth_norm_init
        output = X
        return [output, input]

