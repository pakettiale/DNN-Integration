import numpy as np
import tensorflow as tf

from logdet import logdet

def jacobian(y, x):
    with tf.name_scope("jacob"):
        grads = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(y, axis=1)],
                        axis=2)
        return grads

def regression_loss(x_true, x_pred): #x_true <- f(x), x_pred <- h(G(z))
    eps = np.finfo('float32').eps
    fx_max = tf.map_fn(lambda x: tf.maximum(x, eps),x_true)
    return tf.reduce_sum((tf.log(fx_max) - x_pred)**2, [0])


# G_z = generative,       h_G_z = generative_then_regression
# z   = generative input, Int_f = integral of exp(h)?
def generator_loss(G_z, z, h_G_z, Int_f, p_z): #x_true = exp(h_z), x_pred <- G_z, x_in = z,
    dim = tf.shape(z)[1]
    N = tf.cast(tf.shape(z)[0], tf.float32)
    slogDJ = tf.reshape(logdet(jacobian(G_z, z)), (-1, 1))
    #print(N, p_z, slogDJ)
    #D = 1/N*tf.reduce_sum(-tf.log(tf.exp(h_G_z)/Int_f) + tf.log(p_z) - slogDJ, (0,))
    D = tf.log(2.0) +  1.0/N*tf.reduce_sum((tf.log(p_z) - slogDJ - tf.log(p_z/tf.exp(slogDJ)+tf.exp(h_G_z)/Int_f)),(0,)) #
    return D


def integral_loss(integral):
    return (1 - integral)**2
