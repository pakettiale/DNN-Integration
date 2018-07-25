import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

### Custom Functions
def custom_tanh(x):
    return tf.tanh(x)*0.7+x*0.3

def integrate(samples, importance=1.0, a=0, b=1):
    dim = samples.shape[1]
    return np.abs(b-a)**dim/samples.shape[0]*np.sum(samples/importance)

# Importance sampling integral
def tf_integrate(samples, importance, a=0.0, b=1.0):
    dim = tf.cast(tf.shape(samples)[1], tf.float32)
    return tf.abs(b-a)**dim/tf.cast(tf.shape(samples)[0],tf.float32)*tf.reduce_sum(samples/importance)

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
    dim = tf.shape(x)[1]
    dim_f = tf.cast(dim, tf.float32)
    c = 0.5*tf.pow(1/(a*math.sqrt(2*math.pi)),dim_f)
    rest = tf.exp(-tf.reduce_sum((x-1/3)*(x-1/3)/(2*a*a),1))+tf.exp(-tf.reduce_sum((x-2/3)*(x-2/3)/(2*a*a),1))
    return tf.reshape(c*rest,(-1,1))

def triplegaussian(x):#(batch,dist_dim)
    x = tf.convert_to_tensor(x,dtype=tf.float32)
    a = 0.1/math.sqrt(2)
    dim = tf.shape(x)[1]
    dim_f = tf.cast(dim, tf.float32)
    c = 2/3*tf.pow(1/(a*math.sqrt(2*math.pi)),dim_f)
    rest = tf.exp(-tf.reduce_sum((x-1/4)**2/(2*a*a),1))+tf.exp(-tf.reduce_sum((x-2/4)**2/(2*a*a),1))+tf.exp(-tf.reduce_sum((x-3/4)**2/(2*a*a),1))
    return tf.reshape(c*rest,(-1,1))

def rosenbrock(x):
    x, y = tf.unstack(x, axis=-1)
    a = 1
    b = 100
    return (a - x)**2 + b*(y-x**2)**2

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
