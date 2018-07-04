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

class RegressiveDNN():
    def __init__(self, dim):
        self.dim = dim
        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.Dense(64, input_shape=(self.dim,), activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
        nn.add(tf.keras.layers.Dense(self.dim, activation=tf.keras.activations.linear))

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
        nn.add(tf.keras.layers.Dense(64, activation=self.activation))
        nn.add(tf.keras.layers.Dense(self.dim, activation=tf.keras.activations.sigmoid, name='gen_outputs'))

        self.prior = prior
        self.nn = nn
        self.output = self.nn.output
        self.input = self.nn.input
        self.density = self.prior.prob(self.input)/tf.reshape(tf.abs(tf.linalg.det(jacobian(self.output, self.input))), (-1, self.dim))

h = RegressiveDNN(1)

normal = tf.distributions.Uniform(-1.0, 1.0)
generative = GenerativeDNN(1, normal, tf.keras.activations.tanh)
gen_out = generative.output
gen_in  = generative.input
gen_dst = generative.density

gen_vars = generative.nn.variables

hG = tf.keras.models.Sequential()
hG.add(generative.nn)
hG.add(h.nn)
hG_in = hG.input
hG_out = hG.output

f = triplegaussian


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)

saver = tf.train.Saver()
plot = True
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    #Prior
    zs = tf.keras.backend.eval(tf.reshape(normal.sample((5120*64)),(-1,dist_dim)))
    xs = np.random.uniform(0.0, 1.0, size=(20000,1))
    correct = tf.keras.backend.eval(f(xs))
    probs = tf.keras.backend.eval(normal.prob(zs))

    reg_file = "reg.hdf5"
    if not os.path.isfile(reg_file):
        h.nn.fit(xs, correct, batch_size=int(5120/2), epochs=8*12, verbose=1
              ,callbacks=[reduce_lr])
        h.nn.save_weights(reg_file)
    else:
        print("Loading regression from file")
        h.nn.load_weights(reg_file)

    plt.plot(xs.flatten(), np.exp(h.nn.predict(xs).flatten()), ',r')
    plt.show()


    h_G_z = tf.reshape(hG_out,(-1, dist_dim)) #sess.run(reg_out, {reg_in: z})
    integral_f = tf.Variable(sess.run(tf_integrate(tf.exp(h.output), 1), {h.input: xs}), trainable=False)
    print(tf.keras.backend.eval(integral_f))
    integral_h_G_z = tf_integrate(h.output, 1.0)
    p_z = normal.prob(gen_in)

    gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f, p_z)
    gen_opt = tf.train.AdamOptimizer(0.001)
    gen_train = gen_opt.minimize(gen_loss, var_list=[gen_vars])
    sess.run(tf.global_variables_initializer())
    h.nn.load_weights(reg_file)

    #This is wrong
    integral_g = tf_integrate(gen_dst, gen_dst)

    for e in range(1,10):

        batches = zip(np.reshape(zs, (-1, 2*512, dist_dim)), np.reshape(probs, (-1, 2*512, dist_dim)))

        for z, p in batches:
            _, loss = sess.run([gen_train, gen_loss], {gen_in: z, p_z: p, hG_in: z})
        print("epoch: ", e, " -- loss: ", loss)

        print("prior - G sampled integral: ", sess.run((1.0-tf_integrate(f(gen_out), gen_dst)), {gen_in: z}))

        #print("Gen Integral after train: ", sess.run(integral_g, feed_dict={gen_in: batches[0]}))

        x = zs
        x_ = sess.run(normal.cdf(x.flatten()))
        if plot and e%3 == 0:
            plt.plot(
                #x_, sess.run(f(x)), ',k',
                x_, sess.run(gen_dst, {gen_in: x}), ',b',
                x_, sess.run(tf.exp(hG_out), {hG_in: x}), ',r',
                x_, sess.run(gen_out*4.0, {gen_in: x}), ',g')
            plt.grid(True)
            plt.axis([0, 1, 0.0, 4.5])
            plt.show()
            plt.plot(xs.flatten(), np.exp(h.nn.predict(xs).flatten()), ',r')
            plt.show()

    n, x, _ = plt.hist(sess.run(gen_out, {gen_in: x}).flatten(), 50, density=0.00311)
    plt.plot(x, sess.run(f(np.reshape(x, (-1,1)))).flatten())
    plt.show()
