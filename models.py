import tensorflow as tf
import numpy as np
from losses import regression_loss, jacobian

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
    def __init__(self, dim, prior, activation):

        self.dim = dim
        self.activation = activation

        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.Dense(64, input_shape=(self.dim,), activation=self.activation))
        nn.add(tf.keras.layers.Dense(64, activation=self.activation))
        nn.add(tf.keras.layers.Dense(64, activation=self.activation))
        nn.add(tf.keras.layers.Dense(64, activation=self.activation))
        nn.add(tf.keras.layers.Dense(64, activation=self.activation))
        nn.add(tf.keras.layers.Dense(self.dim, activation=tf.keras.activations.sigmoid, name='gen_outputs'))

        self.prior = prior
        self.nn = nn
        self.output = self.nn.output
        self.input = self.nn.input
        self.abs_grad = tf.reshape(tf.abs(tf.linalg.det(jacobian(self.output, self.input))), (-1, 1))
        self.density = tf.reduce_prod(self.prior.prob(self.input), (-1,), keepdims=True)/self.abs_grad

class GenerativeDNNCustom():
    def __init__(self, dim, prior, activation, scope='gen'):
        [self.output, self.input] = build_net(dim, 5, 64, activation, tf.sigmoid, scope)
        self.scope = scope
        self.prior = prior

    def variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def abs_grad(self):
        return tf.reshape(tf.abs(tf.linalg.det(jacobian(self.output, self.input))), (-1, 1))

    def density(self):
        return tf.reduce_prod(self.prior.prob(self.input), (-1,), keepdims=True)/self.abs_grad()

def build_net(dim, depth, nodes, layer_activation, output_activation, scope):
    with tf.variable_scope(scope):
        X = tf.placeholder(tf.float32, [None, dim], 'input')
        input = X
        print("input name: ", X.name)
        row_dim = dim
        for i in range(1, depth+1):
            if i == depth:
                W = tf.get_variable(str(i) + '_W', [row_dim, dim])
                b = tf.get_variable(str(i) + '_b', [dim])
                X = output_activation(tf.add(tf.matmul(X, W), b), name='output')
            else:
                W = tf.get_variable(str(i) + '_W', [row_dim, nodes])
                b = tf.get_variable(str(i) + '_b', [nodes])
                X = layer_activation(tf.add(tf.matmul(X, W), b))
            row_dim = nodes
        output = X
        return [output, input]

def main():
    G = GenerativeDNNCustom(2, tf.distributions.Uniform(), tf.tanh)
    H = RegressiveDNN(2)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(G.output, {G.input: [[1,2], [0.2, 0.2]]}))
    print(sess.run(H.nn(G.output), {G.input: [[1,2], [0.2, 0.2]]}))
    print(sess.run(G.abs_grad(), {G.input: [[1,2], [0.2, 0.2]]}))
    print(sess.run(G.density(), {G.input: [[1,2], [0.2, 0.2]]}))


if __name__ == '__main__':
    main()

