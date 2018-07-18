import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path

from losses import regression_loss, generator_loss, jacobian, integral_loss
from custom_functions import custom_tanh, doublegaussian, triplegaussian, integrate, tf_integrate, cauchy
from logdet import logdet
from models import GenerativeDNN, RegressiveDNN

dist_dim = 2

h = RegressiveDNN(dist_dim)

normal = tf.distributions.Uniform(-1.0, 1.0)
generative = GenerativeDNN(dist_dim, normal, custom_tanh)#tf.keras.activations.tanh)#
gen_out = generative.output
gen_in  = generative.input
gen_grad = generative.abs_grad
gen_dst = generative.density

gen_vars = generative.nn.variables

hG = tf.keras.models.Sequential()
hG.add(generative.nn)
hG.add(h.nn)
hG_in = hG.input
hG_out = hG.output

f = doublegaussian


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

saver = tf.train.Saver()
plot = False
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    #Prior
    zs = tf.keras.backend.eval(tf.reshape(normal.sample((5120*64)),(-1,dist_dim)))
    xs = np.random.uniform(0.0, 1.0, size=(5120*4,dist_dim))
    correct = tf.keras.backend.eval(f(xs))
    probs = tf.keras.backend.eval(tf.reduce_prod(normal.prob(zs), (-1,), keepdims=True))
    reg_file = "reg.hdf5"
    if not os.path.isfile(reg_file):
        h.nn.fit(xs, correct, batch_size=int(5120), epochs=30*12, verbose=1
              ,callbacks=[reduce_lr, early_stopping])
        h.nn.save_weights(reg_file)
    else:
        print("Loading regression from file")
        h.nn.load_weights(reg_file)

    diagonals = np.reshape(np.repeat(np.arange(0.0, 1.0, 0.01), dist_dim), (-1,dist_dim))
    plt.plot(np.arange(0.0, 1.0, 0.01), np.exp(h.nn.predict(diagonals)), '.r',
             np.arange(0.0, 1.0, 0.01), tf.keras.backend.eval(f(diagonals)), '.g')
    plt.show()


    h_G_z = hG_out #sess.run(reg_out, {reg_in: z})
    integral_f = tf.Variable(sess.run(tf_integrate(tf.exp(h.output), 1.0), {h.input: xs}), trainable=False)
    print(tf.keras.backend.eval(integral_f))
    integral_h_G_z = tf_integrate(h.output, 1.0)
    p_z = tf.reduce_prod(normal.prob(gen_in), (-1), keepdims=True)

    gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f, p_z)
    gen_opt = tf.train.AdamOptimizer(0.0005, 0.9)
    #gen_opt = tf.train.GradientDescentOptimizer(0.001)
    grads_and_vars = gen_opt.compute_gradients(gen_loss, var_list=[gen_vars])
    grads = [x[0] for x in grads_and_vars]
    vars  = [x[1] for x in grads_and_vars]
    grads, grad_norm = tf.clip_by_global_norm(grads, 2)
    gen_train = gen_opt.apply_gradients(zip(grads, vars))
    sess.run(tf.global_variables_initializer())
    h.nn.load_weights(reg_file)

    for e in range(1,5):

        batches = zip(np.reshape(zs, (-1, 1*512, dist_dim)), np.reshape(probs, (-1, 1*512, dist_dim)))

        for z, p in batches:
            #gradients = sess.run(grads, {gen_in: z, p_z: p, hG_in: z})
            #print(gradients)
            _, loss = sess.run([gen_train, gen_loss], {gen_in: z, hG_in: z})
        print("epoch: ", e, " -- loss: ", loss)

        print("prior - G sampled integral: ", sess.run((1.0-tf_integrate(f(gen_out), gen_dst)), {gen_in: z}))

        x = zs
        x_ = sess.run(normal.cdf(x.flatten()))
        if plot and e%10 == 0:
            plt.plot(
                #x_, sess.run(f(x)), ',k',
                x_, sess.run(gen_dst, {gen_in: x}), ',b',
                x_, sess.run(gen_grad, {gen_in: x}), ',k',
                x_, sess.run(tf.exp(hG_out), {hG_in: x}), ',r',
                x_, sess.run(gen_out*4.0, {gen_in: x}), ',g')
            plt.grid(True)
            plt.axis([0, 1, 0.0, 4.5])
            plt.show()

    for cycle in range(1,10):
        if dist_dim == 2:
            xs = np.arange(-1.0, 1.0, 0.025)
            ys = np.arange(-1.0, 1.0, 0.025)
            X, Y = np.meshgrid(xs, ys)
            Z = np.reshape(sess.run(gen_dst, {gen_in: np.array(list(zip(X.flatten(), Y.flatten()))).reshape([-1,2])}), (80,80))
            plt.contour(X, Y, Z)
            plt.show()
            plt.matshow(Z)
            plt.show()
        n, x, _ = plt.hist(sess.run(gen_out, {gen_in: x}), 50, density=0.00311)
        plt.plot(x, sess.run(f(np.reshape(x, (-1,1)))).flatten())
        plt.show()
        zs = tf.keras.backend.eval(tf.reshape(normal.sample((5120*64)),(-1,dist_dim)))
        probs = tf.keras.backend.eval(normal.prob(zs))
        #xs = np.random.uniform(0.0, 1.0, size=(20000,1))
        xs = sess.run(gen_out, {gen_in: zs})
        correct = tf.keras.backend.eval(f(xs))
        h.nn.fit(xs, correct, batch_size=int(5120/2), epochs=64, verbose=1
              ,callbacks=[reduce_lr, early_stopping])

        diagonals = np.reshape(np.repeat(np.arange(0.0, 1.0, 0.01), dist_dim), (-1,dist_dim))
        plt.plot(np.arange(0.0, 1.0, 0.01), np.exp(h.nn.predict(diagonals)), '.r',
                np.arange(0.0, 1.0, 0.01), tf.keras.backend.eval(f(diagonals)), '.g')
        plt.show()

        sess.run(integral_f.assign(sess.run(tf_integrate(tf.exp(hG_out), gen_dst), {hG_in: zs, gen_in: zs})))
        print("integral_f: ", sess.run(integral_f))
        losses = np.zeros((3,))+100000.0
        for e in range(1,45):

            batches = zip(np.reshape(zs, (-1, 10*512, dist_dim)), np.reshape(probs, (-1, 10*512, dist_dim)))

            for z, p in batches:
                _, loss = sess.run([gen_train, gen_loss], {gen_in: z, hG_in: z})
            print("epoch: ", e, " -- loss: ", loss)

            if loss > np.max(losses):
                break
            else:
                np.roll(losses, 1)
                losses[0] = loss

            #print("prior - G sampled integral: ", sess.run((1.0-tf_integrate(f(gen_out), gen_dst)), {gen_in: z}))

            x = zs
            x_ = sess.run(normal.cdf(x.flatten()))
            if plot and e%3 == 0:
                plt.plot(
                    #x_, sess.run(f(x)), ',k',
                    x_, sess.run(gen_dst, {gen_in: x}), ',b',
                    x_, sess.run(gen_grad, {gen_in: x}), ',k',
                    x_, sess.run(tf.exp(hG_out), {hG_in: x}), ',r',
                    x_, sess.run(gen_out*4.0, {gen_in: x}), ',g')
                plt.grid(True)
                plt.axis([0, 1, 0.0, 4.5])
                plt.show()
