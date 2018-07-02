import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path

from losses import regression_loss, generator_loss, jacobian, integral_loss
from custom_functions import custom_tanh, doublegaussian, triplegaussian, integrate, tf_integrate, cauchy
from logdet import logdet

dist_dim = 1

generative = tf.keras.models.Sequential()
generative.add(tf.keras.layers.Dense(64, input_shape=(dist_dim,), activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid))
generative.add(tf.keras.layers.Dense(dist_dim, activation=tf.keras.activations.sigmoid, name='gen_outputs'))
gen_out = generative.output*2.0 -1.0
gen_in  = generative.input
#normal = tf.distributions.Normal(loc=0.5, scale=1.0)
normal = tf.distributions.Uniform(-1.0, 1.0)
gen_dst = normal.prob(gen_in)/tf.reshape(tf.abs(tf.linalg.det(jacobian(gen_out, gen_in))), (-1,dist_dim))

f = triplegaussian

saver = tf.train.Saver()
plot = True
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    #Prior
    zs = tf.keras.backend.eval(tf.reshape(normal.sample((5120*64)),(-1,dist_dim)))
    probs = tf.keras.backend.eval(normal.prob(zs))

    h_G_z = tf.reshape(f(gen_out),(-1, dist_dim)) #sess.run(reg_out, {reg_in: z})
    integral_f = tf.Variable(sess.run(tf_integrate(f(zs), normal.prob(zs))), trainable=False)
    p_z = normal.prob(gen_in)

    gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f, p_z)
    gen_train = tf.train.AdamOptimizer(0.001).minimize(gen_loss)
    sess.run(tf.global_variables_initializer())

    #This is wrong
    integral_g = tf_integrate(gen_dst, gen_dst)

    for e in range(1,40):

        batches = zip(np.reshape(zs, (-1, 512, dist_dim)), np.reshape(probs, (-1, 512, dist_dim)))

        for z, p in batches:
            _, loss = sess.run([gen_train, gen_loss], {gen_in: z, p_z: p})
        print("epoch: ", e, " -- loss: ", loss)

        print("prior - G sampled integral: ", sess.run((1.0-tf_integrate(f(gen_out), gen_dst)), {gen_in: z}))

        #print("Gen Integral after train: ", sess.run(integral_g, feed_dict={gen_in: batches[0]}))

        x = zs
        x_ = sess.run(normal.cdf(x.flatten()))
        if plot and e%3 == 0:
            plt.plot(
                #x_, sess.run(f(x)), ',k',
                x_, sess.run(gen_dst, {gen_in: x}), ',b',
                x_, sess.run(f(gen_out), {gen_in: x}), ',r',
                x_, sess.run(gen_out*4.0, {gen_in: x}), ',g')
            plt.grid(True)
            plt.axis([0, 1, 0.0, 4.5])
            plt.show()

    n, x, _ = plt.hist(sess.run(gen_out, {gen_in: x}).flatten(), 50, density=0.00311)
    plt.plot(x, sess.run(f(np.reshape(x, (-1,1)))).flatten())
    plt.show()
