import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from losses import regression_loss, jacobian, generator_loss
from logdet import logdet
from custom_functions import tf_integrate, triplegaussian, doublegaussian, custom_tanh, cauchy, integrate
from models import *
import generate_linear


def main():
    test_dim = 4

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    f = doublegaussian
    generate_linear.main(test_dim)
    tf.reset_default_graph()
    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        with sess.as_default():
            G = GenerativeDNN(test_dim, tf.distributions.Normal(0., 1.), custom_tanh)
            #G = GenerativeDNN(2, tf.distributions.Uniform(0., 1.), custom_tanh)
            H = RegressiveDNN(test_dim)
            G.set_target_function(H.nn)
            G.create_trainer(0.0001)

            H_int = G.create_integral(20000, G.target_function)
            f_int = G.create_integral(20000, f, use_exp=False)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(G.variables())
            saver.restore(sess, 'saved_models/linear')
            x = np.zeros((100,))
            for i, _ in enumerate(x):
                x[i] = G.integrate(f_int, 20000)
            print("∫f:", x.mean(), "±", x.std())
            G.set_target_integral(G.integrate(H_int, 20000))
            print("f integral:", G.integrate(f_int, 20000))
            print("H integral:", G.target_integral.eval())
            for _ in range(0,2):
                X = G.sampler(300000)
                Y = f(X.reshape((-1,test_dim))).eval().flatten()
                H.nn.fit(X, Y, 512*4, 15, callbacks=[reduce_lr, early_stopping])
                G.set_target_integral(G.integrate(H_int, 20000))
                print("H integral:", G.target_integral.eval())
                #saver.restore(sess, 'saved_models/linear')
                G.train(40, 512*8, 30)
                G.set_target_integral(G.integrate(H_int, 20000))
                print("H integral:", G.target_integral.eval())
                print("f integral:", G.integrate(f_int, 20000))
            x = np.zeros((100,))
            for i, _ in enumerate(x):
                x[i] = G.integrate(f_int, 20000)
            print("∫f:", x.mean(), "±", x.std())



            ### Plotting

            if G.dim == 1:
                xs = G.prior.sample((4000, 1)).eval()
                plt.plot(sess.run(G.output, {G.input: xs}).flatten(), sess.run(G.density, {G.input: xs}).flatten(), ',b',
                        #xs.flatten(), sess.run(G.output, {G.input: xs}).flatten(), ',k',
                        sess.run(G.output, {G.input: xs}).flatten(), sess.run(f(G.output), {G.input: xs}), ',g')
                plt.show()
            elif G.dim == 2:
                xs = G.sampler(1600)
                plt.scatter(xs[:,0], xs[:,1])
                plt.show()
                xs = np.linspace(0., 1., 160)
                ys = np.linspace(0., 1., 160)
                X, Y = np.meshgrid(xs, ys)
                Z4 = H.nn.predict(np.array(list(zip(X.flatten(), Y.flatten()))))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(X, Y, np.exp(Z4.reshape((160,160))))
                #ax.scatter(ZX, ZY, Z2)
                plt.show()
                xs = np.sort(G.prior.sample((160, 1)).eval(), axis=0)
                ys = np.sort(G.prior.sample((160, 1)).eval(), axis=0)
                X, Y = np.meshgrid(xs, ys)
                Z = sess.run(G.output, {G.input: np.array(list(zip(X.flatten(), Y.flatten())))})
                ZX = Z[:,0]
                ZY = Z[:,1]
                Z2 = sess.run(G.density, {G.input: np.array(list(zip(X.flatten(), Y.flatten())))})
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
