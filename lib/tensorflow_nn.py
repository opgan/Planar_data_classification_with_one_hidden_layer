# pylint: disable=unused-import, unused-variable, unused-argument

import tensorflow as tf
from lib.log import log


def compute_total_loss(logits, labels):
    """
    Computes the total loss

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3 (6, num_examples)

    Returns:
    total_loss - Tensor of the total loss value
    """

    # (1 line of code)
    # remember to set `from_logits=True`
    # total_loss = ...
    # YOUR CODE STARTS HERE
    total_loss = tf.reduce_sum( # pylint: disable= no-member
        tf.keras.losses.categorical_crossentropy(  # pylint: disable= no-member
            tf.transpose(labels), tf.transpose(logits), from_logits=True
        )
    )
    # YOUR CODE ENDS HERE
    return total_loss


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # (approx. 5 lines)                   # Numpy Equivalents (NumPy not to be used. Use TF API):
    # Z1 = ...                           # Z1 = np.dot(W1, X) + b1
    # A1 = ...                           # A1 = relu(Z1)
    # Z2 = ...                           # Z2 = np.dot(W2, A1) + b2
    # A2 = ...                           # A2 = relu(Z2)
    # Z3 = ...                           # Z3 = np.dot(W3, A2) + b3
    # YOUR CODE STARTS HERE
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1) # pylint: disable= no-member
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2) # pylint: disable= no-member
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)
    # YOUR CODE ENDS HERE

    return Z3


def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    initializer = tf.keras.initializers.GlorotNormal(seed=1) # pylint: disable= no-member

    W1 = tf.Variable(initializer(shape=[25, 12288]))
    b1 = tf.Variable(initializer(shape=[25, 1]))
    W2 = tf.Variable(initializer(shape=[12, 25]))
    b2 = tf.Variable(initializer(shape=[12, 1]))
    W3 = tf.Variable(initializer(shape=[6, 12]))
    b3 = tf.Variable(initializer(shape=[6, 1]))
    # YOUR CODE ENDS HERE

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    learning_rate=0.0001,
    num_epochs=1500,
    minibatch_size=32,
    print_cost=True,
):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []  # To keep track of the cost
    train_acc = []
    test_acc = []

    # Initialize your parameters
    # (1 line)
    parameters = initialize_parameters()

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    optimizer = tf.keras.optimizers.Adam(learning_rate) # pylint: disable= no-member

    # The CategoricalAccuracy will track the accuracy for this multiclass problem
    test_accuracy = tf.keras.metrics.CategoricalAccuracy() # pylint: disable= no-member
    train_accuracy = tf.keras.metrics.CategoricalAccuracy() # pylint: disable= no-member

    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))

    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()

    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    # X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step
    # Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_total_loss = 0.0

        # We need to reset object to start measuring from 0 the accuracy each epoch
        train_accuracy.reset_states()

        for minibatch_X, minibatch_Y in minibatches:

            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                # 2. loss
                minibatch_total_loss = compute_total_loss(Z3, tf.transpose(minibatch_Y))

            # We accumulate the accuracy of all the batches
            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_total_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_total_loss += minibatch_total_loss

        # We divide the epoch total loss over the number of samples
        epoch_total_loss /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
            print("Train accuracy:", train_accuracy.result())

            # We evaluate the test set every 10 epochs to avoid computational overhead
            for minibatch_X, minibatch_Y in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()

    return parameters, costs, train_acc, test_acc


def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.

    Arguments
    image - Tensor.

    Returns:
    result -- Transformed tensor
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(
        image,
        [
            -1,
        ],
    )
    return image


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
