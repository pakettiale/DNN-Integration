import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path

from losses import regression_loss, generator_loss, jacobian, integral_loss
from custom_functions import custom_tanh, doublegaussian, triplegaussian, integrate, tf_integrate, cauchy
from logdet import logdet

### Notes:
### Prior distribution Unif(-1, 1) seems to have better performance than Unif(0, 2)
### Integrate_g doesn't work, too large values
###
### ToDo:
### Make gen_loss take p_z as a tf variable => allows generating nontrivial data for training
###    and better possibly better learning rate

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


#f = cauchy
f = triplegaussian
#def f(x):
#    return tf.constant(1.0)*x+0.5

saver = tf.train.Saver()
plot = True
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    #Prior
    zs = tf.keras.backend.eval(tf.reshape(normal.sample((5120*64)),(-1,dist_dim)))
    zs_o = zs
    probs = tf.keras.backend.eval(normal.prob(zs))
    #                                                    vvvvvvv check combined pdf
    gen_zs, gen_zs_pdf = sess.run([gen_out, gen_dst], {gen_in: zs})
    gen_zs = np.reshape(gen_zs, (-1, 512, dist_dim))
    gen_zs_pdf = np.reshape(gen_zs_pdf, (-1, 512, dist_dim))

    h_G_z = tf.reshape(f(gen_out),(-1, dist_dim)) #sess.run(reg_out, {reg_in: z})
    integral_f = tf.Variable(sess.run(tf_integrate(f(zs), normal.prob(zs))), trainable=False)
    p_z = normal.prob(gen_in)
    #This is wrong
    integral_g = tf_integrate(gen_dst, gen_dst)
    gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f, integral_g, p_z)
    int_loss = integral_loss(integral_g)
    int_train = tf.train.AdamOptimizer(0.01).minimize(int_loss)
    gen_train = tf.train.AdamOptimizer(0.001).minimize(gen_loss)
    sess.run(tf.global_variables_initializer())

    for e in range(1,40):

        batches = zip(np.reshape(zs, (-1, 512, dist_dim)), np.reshape(probs, (-1, 512, dist_dim)))
        #print("epoch: ", e)
        #for z in batches:
        #    _, intloss = sess.run([int_train, int_loss], {gen_in: z})
        #print("Gen Integral before train: ", sess.run(integral_g, feed_dict={gen_in: batches[0]}))
        #print("With loss: ", intloss)

        for z, p in batches:
            _, loss = sess.run([gen_train, gen_loss], {gen_in: z, p_z: p})
        print("loss: ", loss)

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
