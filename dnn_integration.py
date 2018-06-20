import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
from logdet import logdet
from time import sleep
from tensorflow.python import debug as tf_debug

dist_dim = 1

### Custom Functions
def custom_tanh(x):
    return tf.keras.activations.tanh(x)*0.7+x*0.3

def jacobian(y, x):
    with tf.name_scope("jacob"):
        grads = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(y, axis=1)],
                        axis=2)
        return grads

def integrate(samples, a=0, b=1):
    return np.abs(b-a)/samples.size*np.sum(samples)

def tf_integrate(samples, a=0.0, b=1.0):
    return tf.abs(b-a)/tf.cast(tf.shape(samples)[0],tf.float32)*tf.reduce_sum(samples)

def cauchy(x):
    Gamma = tf.constant(0.1)
    return 1/math.pi*Gamma/((x-0.5)*(x-0.5)+Gamma*Gamma)

def cauchy_cdf(x):
    def f(x):
        return 0.5+Gamma*np.tan(np.arctan(1/(2*Gamma))*math.erf(x/(np.sqrt(2))))
    return list(map(f, x))

def doublegaussian(x):#(batch,dist_dim)
    x = tf.convert_to_tensor(x,dtype=tf.float32)
    a = 0.1/math.sqrt(2)
    n = dist_dim
    print(n)
    c = 0.5*math.pow(1/(a*math.sqrt(math.pi)),n)
    rest = tf.exp(-tf.reduce_sum((x-1/3)*(x-1/3)/(a*a),1))+tf.exp(-tf.reduce_sum((x-2/3)*(x-2/3)/(a*a),1))
    return c*rest

def regression_loss(x_true, x_pred): #x_true <- f(x), x_pred <- h(G(z))
    eps = np.finfo('float32').eps
    fx_max = tf.map_fn(lambda x: tf.maximum(x, eps),x_true)
    return tf.reduce_sum((tf.log(fx_max) - x_pred)**2, [0])

def generator_loss(G_z, z, h_G_z, Int_f): #x_true = exp(h_z), x_pred <- G_z, x_in = z,
    N = tf.cast(tf.shape(z)[0], tf.float32)
    p_z = 1.0
    slogDJ = tf.reshape((logdet(jacobian(G_z, z))),(-1,dist_dim))
    #print(N, p_z, slogDJ)
    D = 1.0/N*tf.reduce_sum(tf.log(p_z) - slogDJ - tf.log(p_z/tf.exp(slogDJ)+tf.exp(h_G_z)/Int_f),(0,))
    return D

### DNN model
generative = tf.keras.models.Sequential()
generative.add(tf.keras.layers.Dense(64, input_shape=(dist_dim,), activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(dist_dim, activation=tf.keras.activations.sigmoid, name='gen_outputs'))
gen_out = generative.output
gen_in  = generative.input

regression = tf.keras.models.Sequential()
regression.add(tf.keras.layers.Dense(64, input_shape=(dist_dim,), activation=tf.keras.activations.elu))
regression.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
regression.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
regression.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
regression.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.elu))
regression.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

reg_out = regression.output
reg_in  = regression.input
regression.compile(loss=regression_loss,
                   optimizer="adam")

#regplotepoch = PlotEpoch()
#fetches = [tf.assign(regplotepoch.var_input, regression.inputs, validate_shape=False),
#           tf.assign(regplotepoch.var_realf, regression.targets, validate_shape=False),
#           tf.assign(regplotepoch.var_predf, regression.outputs, validate_shape=False)]
#regression._function_kwargs = {'fetches': fetches}

### Fitting and testin
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)


def plot_reg(zs, correct, dist_dim):
    if dist_dim == 1:
        data = zs.flatten()
        plt.subplot(211)
        plt.plot(data[0:1000], correct[0:1000], ',g', data, np.exp(regression.predict(zs).flatten()), ',r')
        plt.subplot(212)
        plt.semilogy(data[0:1000], correct[0:1000], ',g', data, np.exp(regression.predict(zs).flatten()), ',r')
        plt.show()

    if dist_dim == 2:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_f[:1000,0], data_f[:1000,1], zs=np.exp(regression.predict(data_f).flatten())[:1000])
        plt.show()

### Test data
target_function = doublegaussian

### Train loop
saver = tf.train.Saver()
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    unif = tf.distributions.Uniform(low=0.0, high=1.0)
    zs = tf.keras.backend.eval(tf.reshape(unif.sample((5120)),(-1,dist_dim)))

    sess.run(tf.global_variables_initializer())
    correct = tf.keras.backend.eval(target_function(zs))
    h_G_z = reg_out #sess.run(reg_out, {reg_in: z})
    integral_f = tf_integrate(tf.exp(reg_out))
    gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f)
    gen_train = tf.train.GradientDescentOptimizer(0.005).minimize(gen_loss)
    #linear_file = "linear.hdf5"
    #if not os.path.isfile(linear_file):
    #regression.fit(zs, zs, batch_size=int(5120/16), epochs=24, verbose=1)
    #regression.save_weights(linear_file)
    #else:
    #regression.load_weights(linear_file)
    #
    #plot_reg(zs, zs, dist_dim)
    #sess.run(gen_train, {gen_in: zs, reg_in: zs})

    reg_file = "reg.hdf5"
    if not os.path.isfile(reg_file):
        regression.fit(zs, correct, batch_size=5120, epochs=12, verbose=1
                   ,callbacks=[reduce_lr])
        regression.save_weights(reg_file)
    else:
        print("!!! Loading regression from file")
        regression.load_weights(reg_file)
    batches = np.reshape(zs, (-1, 5120, dist_dim))
    for e in range(1,10):
        if e != 1:
            newzs = tf.keras.backend.eval(tf.reshape(unif.sample((5120)),(-1,dist_dim)))
            newzs = np.split(newzs, 2)
            zs = np.concatenate((sess.run(gen_out, {gen_in: newzs[0]}), newzs[1]))
            np.random.shuffle(zs)
            correct = tf.keras.backend.eval(target_function(zs))
            regression.fit(zs, correct, batch_size=5120, epochs=8*16, verbose=1
                   ,callbacks=[reduce_lr])
            plot_reg(zs, correct, dist_dim)
            batches = np.reshape(zs,(-1, 5120, dist_dim))

        for i in range(1,100):
            z = batches[0]
            #print(z)
            #h_G_z = sess.run(reg_out, {reg_in: sess.run(gen_out, {gen_in: z})})
            #h_G_z = reg_out #sess.run(reg_out, {reg_in: z})
            #integral_f = integrate(sess.run(tf.exp(reg_out), {reg_in: z}))
            #integral_f = tf_integrate(reg_out)
            #print("h(G(Z)): ", h_G_z)
            print("integral f: ", sess.run(integral_f, {reg_in: z}))
            #gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f)
            #gen_train = tf.train.GradientDescentOptimizer(0.05).minimize(gen_loss)
            _, loss = sess.run([gen_train, gen_loss], {gen_in: z, reg_in: z})
            print(i, ", loss: ", loss)

        print("epoch: ", e)
        plt.plot(np.linspace(0,1,100), sess.run(gen_out, {gen_in: np.reshape(np.linspace(0,1,100),(-1,1))}), '.b')
        plt.show()
