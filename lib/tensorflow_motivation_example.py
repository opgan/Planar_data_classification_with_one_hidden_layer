# pylint: disable=unused-import, unused-variable, unused-argument

import tensorflow as tf
from lib.log import log


def train_step(cost_fun, w, optimizer):
    with tf.GradientTape() as tape:
        cost = cost_fun(w)  # w**2 - 10*w + 25
    trainable_variables = [w]
    grads = tape.gradient(cost, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))


def iterate_train_steps(cost_fun, steps):
    w = tf.Variable(0, dtype=tf.float32)

    # If you're certain that your code works correctly but Pylint still raises the error, you can temporarily disable the no-member check for that specific line:
    optimizer = tf.keras.optimizers.Adam(0.1)  # pylint: disable=no-member

    for i in range(steps):
        train_step(cost_fun, w, optimizer)

    result_string = (
        "Tensorflow motivating example resulted in optimised parameter as "
        + " :"
        + str(w)
    )
    print(result_string)
    log(result_string)
    # Print the TensorFlow version
    # print(tf.__version__)
