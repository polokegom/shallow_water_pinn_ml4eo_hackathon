import tensorflow as tf
import numpy as np
from deepxde import deepxde as dde
from deepxde import utilities

# 2D shallow water equation
def shallow_water_equations(x, y, h, u, v):
    dh_dt = -tf.gradients(h * u, x) - tf.gradients(h * v, y)
    du_dt = -tf.gradients(u * u + 0.5 * 9.81 * h**2, x) + 0.001 * tf.gradients(u, x, x) + 0.001 * tf.gradients(u, y, y)
    dv_dt = -tf.gradients(v * u + 0.5 * 9.81 * h**2, y) + 0.001 * tf.gradients(v, x, x) + 0.001 * tf.gradients(v, y, y)
    return [dh_dt, du_dt, dv_dt]


# Define the geometry and boundary conditions
def geometry(p):
    x, y = p
    return x, y


def func(x, y):
    return [0.0, 0.0, 0.0]

geom = dde.geometry.Geometry(geometry, train_domain=[(-1, 1), (-1, 1)])
geom.update(func)


# NN architecture
layer_size = [2] + [32] * 4 + [3]
activation = "tanh"
initializer = "Glorot uniform"


def neural_net(X, Y):
    XY = tf.concat([X, Y], axis=1)
    net = dde.layers.dense(XY, layer_size[0], activation=activation, initializer=initializer)
    for width in layer_size[1:]:
        net = dde.layers.dense(net, width, activation=activation, initializer=initializer)
    return net


# Define the DeepXDE model
model = dde.Model(inputs=geom, outputs=neural_net(geom.x, geom.y))


# loss function
def custom_loss(derivatives):
    dh_dt, du_dt, dv_dt = derivatives
    return tf.reduce_mean(tf.square(dh_dt) + tf.square(du_dt) + tf.square(dv_dt))

model.compile("adam", lr=0.001)
model.compile("adam", lr=0.001)


# initial and boundary conditions
def initial_condition(p):
    x, y = p
    return [0.1 * np.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)), 0, 0]


def boundary_condition(x, on_boundary):
    return on_boundary


data = dde.data.PDE(
    geom,
    shallow_water_equations,
    [initial_condition, boundary_condition],
    num_domain=40000,
    num_boundary=2000,
    anchors=np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]),
)


# Training
losshistory, train_state = model.train(data, custom_loss, epochs=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)