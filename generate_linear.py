
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from losses import regression_loss, jacobian, generator_loss
from logdet import logdet
from custom_functions import tf_integrate, triplegaussian, doublegaussian, custom_tanh, cauchy
from models import *

def main():
    with tf.Session() as sess:
        G = GenerativeDNN(2, tf.distributions.Normal(0., 1.), custom_tanh)

        G.set_target_function(G.prior.cdf)
        G.create_trainer(0.0001, KL=False)
        sess.run(tf.global_variables_initializer())
        G.train(40, 512*4, 30)
        saver = tf.train.Saver(G.variables())
        path = saver.save(sess, 'saved_models/linear')
        print(path)

        xs = G.sampler(1600)
        plt.scatter(xs[:,0], xs[:,1])
        plt.show()
        xs = np.sort(G.prior.sample((160, 1)).eval(), axis=0)
        ys = np.sort(G.prior.sample((160, 1)).eval(), axis=0)
        X, Y = np.meshgrid(xs, ys)
        Z = sess.run(G.output, {G.input: np.array(list(zip(X.flatten(), Y.flatten())))})
        ZX = Z[:,0]
        ZY = Z[:,1]
        Z2 = sess.run(G.density(), {G.input: np.array(list(zip(X.flatten(), Y.flatten())))})
        Z3 = sess.run(G.abs_grad(), {G.input: np.array(list(zip(X.flatten(), Y.flatten())))})
        plt.hist2d(Z[:,0], Z[:,1], bins=40)
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(ZX.reshape((160,160)), ZY.reshape((160,160)), Z2.reshape((160,160)))
        #ax.scatter(ZX, ZY, Z2)
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z3.reshape((160,160)))
        plt.show()

if __name__ == '__main__':
    main()
