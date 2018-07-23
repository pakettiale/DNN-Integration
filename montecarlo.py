import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from models import GenerativeDNN
from custom_functions import integrate, tf_integrate, rosenbrock, custom_tanh, triplegaussian

#Function and parameters
f = triplegaussian
dim = 2
low, high = [0.0, 1.0]

#Generative network
G = GenerativeDNN(dim, tf.distributions.Uniform(low, high), custom_tanh)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for sample_size in [1000, 10000, 100000]:
    ## Integrate with naive MC
    batch_size = 1000

    points = np.random.uniform(0.0, 1.0, dim*sample_size).reshape((-1, dim))
    values = sess.run(f(points))
    values_squared = values**2
    naive_integral = integrate(values)
    naive_error = values_squared.mean() - values.mean()**2
    print("Naive integral:", naive_integral, ", +-", naive_error)

    for net in os.listdir("saved_models/"):
        G.nn.load_weights("saved_models/" + net)
        #prior = tf.reshape(G.prior.sample(sample_size), (-1, dim))
        prior = np.random.uniform(-1.0, 1.0, dim*sample_size).reshape((-1, dim))
        #for i, data in enumerate(prior):
            #p, w = sess.run([G.output, G.density], {G.input: data})
            #points = np.concatenate([points, p], axis=0)
            #weights = np.concatenate([weights, w], axis=0)
        points, weights = sess.run([G.output, G.density], {G.input: prior})
        values = sess.run(f(points.reshape(-1, dim)))
        values_squared = values**2
        nn_integral = integrate(values, weights)
        nn_error = values_squared.mean() - values.mean()**2
        print("NN iteration:", net)
        print("NN integral:", nn_integral, ", +-", nn_error)


