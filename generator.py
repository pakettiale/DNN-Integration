import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path


from losses import regression_loss, generator_loss, jacobian
from logdet import logdet



dist_dim = 1

### Custom Functions
def custom_tanh(x):
    return tf.keras.activations.tanh(x)*0.7+x*0.3

def integrate(samples, a=0, b=1):
    return np.abs(b-a)/samples.size*np.sum(samples)

# Importance sampling integral
def tf_integrate(samples, importance, a=0.0, b=1.0):
    return tf.abs(b-a)/tf.cast(tf.shape(samples)[0],tf.float32)*tf.reduce_sum(samples/importance)

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
    c = 0.5*math.pow(1/(a*math.sqrt(2*math.pi)),n)
    rest = tf.exp(-tf.reduce_sum((x-1/3)*(x-1/3)/(2*a*a),1))+tf.exp(-tf.reduce_sum((x-2/3)*(x-2/3)/(2*a*a),1))
    return c*rest



generative = tf.keras.models.Sequential()
generative.add(tf.keras.layers.Dense(64, input_shape=(dist_dim,), activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh))
generative.add(tf.keras.layers.Dense(dist_dim, activation=tf.keras.activations.sigmoid, name='gen_outputs'))
gen_out = generative.output
gen_in  = generative.input
normal = tf.distributions.Normal(loc=0.5, scale=1.0)
#normal = tf.distributions.Uniform(0.0, 1.0)
gen_dst = normal.prob(gen_in)/tf.reshape(tf.exp(logdet(jacobian(gen_out, gen_in))), (-1,dist_dim))



f = cauchy

saver = tf.train.Saver()
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    unif = tf.distributions.Uniform(low=0.0, high=1.0)
    zs = tf.keras.backend.eval(tf.reshape(normal.sample((5120*32)),(-1,dist_dim)))

    #print((tf_integrate(f(zs, ))))
    #plt.plot(zs, sess.run(f(zs)), ',')
    #plt.show()

    #generative.load_weights("linear.hdf5")

    correct = tf.keras.backend.eval(f(zs))
    h_G_z = tf.reshape(tf.log(f(gen_out)),(-1, dist_dim)) #sess.run(reg_out, {reg_in: z})
    integral_f = tf.Variable(sess.run(tf_integrate(f(zs), normal.prob(zs))), trainable=False)
    p_z = normal.prob(gen_in)
    gen_loss = generator_loss(gen_out, gen_in, h_G_z, integral_f, p_z)
    gen_train = tf.train.AdagradOptimizer(0.005).minimize(gen_loss)
    sess.run(tf.global_variables_initializer())

    batches = np.reshape(zs, (-1, 5120, dist_dim))
    #plt.plot(np.linspace(0,1,100), sess.run(f(np.reshape(np.linspace(0,1,100), (-1,1)))), np.linspace(0,1,100), sess.run(gen_out, {gen_in: np.reshape(np.linspace(0,1,100),(-1,1))}), '.b')
    #plt.show()
    for e in range(1,100):
        for z in batches:
            #z = batches[0]
            print("integral f: ", sess.run(integral_f, {gen_in : z}))
            #integral difference
            _, loss = sess.run([gen_train, gen_loss], {gen_in: z})
            print("loss: ", loss)

        print("epoch: ", e)
        if e % 10 == 1:
            x = batches[0]
            x_ = x.flatten()
            plt.plot(
                x_, sess.run(f(gen_out), {gen_in: x}), ',k',
                x_, sess.run(gen_dst, {gen_in: x}), ',b',
                x_, sess.run(gen_out, {gen_in: x}), ',r')
            plt.grid(True)
            plt.axis([-4, 4, 0.0, 3.5])
            print("prior - G sampled integral: ", sess.run((integral_f-tf_integrate(f(gen_out), gen_dst)), {gen_in: z}))
            plt.show()
