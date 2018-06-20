import tensorflow as tf
import numpy as np

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_grad.py
# Gradient for logdet
def logdet_grad(op, grad):
    a = op.inputs[0]
    a_adj_inv = tf.matrix_inverse(a, adjoint=True)
    out_shape = tf.concat([tf.shape(a)[:-2], [1, 1]], axis=0)
    return tf.reshape(grad, out_shape) * a_adj_inv
# define logdet by calling numpy.linalg.slogdet
def logdet(a, name = None):
    with tf.name_scope(name, 'LogDet', [a]) as name:
        res = py_func(lambda a: np.linalg.slogdet(a)[1], 
                      [a], 
                      tf.float32, 
                      name=name, 
                      grad=logdet_grad) # set the gradient
        return res
