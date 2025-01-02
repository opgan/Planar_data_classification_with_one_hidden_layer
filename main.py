# pylint: disable=unnecessary-lambda, unnecessary-pass, unused-argument, unused-variable
#!/usr/bin/env python3
# python main.py

import tensorflow as tf
from lib.data import injest
from lib.helper import fit_logistic_regression_model
from lib.helper import compute_accuracy
from lib.plot import plot
from lib.plot import plot_decision_boundary
from lib.plot import plot_image
from lib.one_hidden_layer_nn import nn_model
from lib.one_hidden_layer_nn import predict
from lib.plot import plot_costs
from lib.tensorflow_motivation_example import iterate_train_steps
from lib.huggingface_example import chat
from lib.keras_sequential import build_model
from lib.log import log

from lib.convolutionnn import build_conv_model
from lib.plot import plot_history

import numpy as np
import click


@click.group()
def cli():
    """This function classify a plannar dataset via Logistic Regression and one hidden layer Neural Network"""
    pass


@cli.command()
# @click.argument("digit", type=int)
def logistic_regression_model():
    """
    Builds linear regression model (weights and bias) to classify flower planar dataset

    Argument:
    none

    Returns:
    Decision boundary plan saved as png file in plots folder
    Accuracy of hidden layer saved info.log file in log folder
    """

    X_train, Y_train, X_test, Y_test, classes = injest(
        "spiral_planar_dataset"
    )  # X is (n_features, n_samples) Y is (n_label, n_samples)
    plot(X_train, Y_train)

    # Build a model with linear regression
    clf = fit_logistic_regression_model(X_train, Y_train)
    plot_title = "linear regression "
    plot_decision_boundary(lambda x: clf.predict(x), X_train, Y_train, plot_title)
    compute_accuracy(
        lambda x: clf.predict(x.T).reshape(-1, 1).T, X_train, Y_train, plot_title
    )  # X.T is (n_samples, n_features)


@cli.command()
@click.argument("n_h", type=int)
def one_hidden_layer_nn_model(n_h):
    """
    Builds a nn_model with a n_h-dimensional hidden layer

    Argument:
    n_h -- number of nodes in the hidden layer

    Returns:
    Decision boundary plan saved as png file in plots folder
    Accuracy of hidden layer saved info.log file in log folder
    """
    # n_h = 4

    X_train, Y_train, X_test, Y_test, classes = injest(
        "spiral_planar_dataset"
    )  # X is (n_features, n_samples) Y is (n_label, n_samples)
    plot(X_train, Y_train)

    parameters, costs = nn_model(
        X_train, Y_train, n_h, num_iterations=10000, print_cost=False
    )
    plot_title = "hidden layer size " + str(n_h)
    plot_decision_boundary(
        lambda x: predict(parameters, x.T), X_train, Y_train, plot_title
    )
    compute_accuracy(lambda x: predict(parameters, x), X_train, Y_train, plot_title)
    plot_costs(costs)


@cli.command()
@click.argument("steps", type=int)
def run_tensorflow_motivation_example(steps):
    """
    Run tensorflow to find w that minimise a given cost funtion (e.g. w**2 - 10*w + 25)

    Argument:
    steps -- number of steps or iterations in gradient descent optimisation

    Returns:
        parameter that minimized cost saved info.log file in log folder
    """

    iterate_train_steps(lambda w: w**2 - 10 * w + 25, steps)


@cli.command()
# @click.argument("steps", type=int)
def run_chat():
    """
    Run chatbot

    Argument:
    none

    Returns:
    reply for a question
    """

    chat()


@cli.command()
@click.argument("epocs", type=int)
def tensorflow_keras_sequential_model(epocs):
    """
    Builds a tensorflow keras sequential model

    Argument:
    none

    Returns:
    Decision boundary plan saved as png file in plots folder
    Accuracy of hidden layer saved info.log file in log folder
    """

    X_train, Y_train, X_test, Y_test, classes = injest(
        "happy_face_dataset"
    )  # X is (n_samples,n_features) Y is (n_samples, n_label)
    index = 155
    plot_image(
        X_train[index], classes[np.squeeze(Y_train[index])], "face"
    )  # X_train(600, 64, 64, 3), Y_train(600, 1)
    happy_model = build_model()
    happy_model.fit(X_train, Y_train, epochs=epocs, batch_size=16)
    loss, accuracy = happy_model.evaluate(X_test, Y_test)
    print(f"loss is {loss:.5f} and accuracy is {accuracy:.5f}")
    log(f"loss is {loss:.5f} and accuracy is {accuracy:.5f}")


@cli.command()
@click.argument("epocs", type=int)
def tensorflow_keras_functional_api_model(epocs):
    """
    Builds a tensorflow keras functional_api model

    Argument:
    none

    Returns:
    Decision boundary plan saved as png file in plots folder
    Accuracy of hidden layer saved info.log file in log folder
    """

    X_train, Y_train, X_test, Y_test, classes = injest(
        "signs_dataset"
    )  # X is (n_samples,n_features) Y is (n_samples, n_label)
    index = 9
    plot_image(
        X_train[index],
        classes[np.argmax(Y_train[index])],
        "signs",  # Find the index of the first occurrence of 1 in Y_train e.g. [0., 0., 0., 0., 1., 0.]
    )  # X_train(600, 64, 64, 3), Y_train(600, 1)

    signs_model = build_conv_model((64, 64, 3))
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    history = signs_model.fit(train_dataset, epochs=epocs, validation_data=test_dataset)
    plot_history(history)
    loss, accuracy = signs_model.evaluate(test_dataset)
    print(f"loss is {loss:.5f} and accuracy is {accuracy:.5f}")
    log(f"loss is {loss:.5f} and accuracy is {accuracy:.5f}")


@cli.command()
@click.argument("epocs", type=int)
def tensorflow_keras_functional_api_model(epocs):
    """
    Builds a tensorflow keras functional_api model

    Argument:
    none

    Returns:
    Decision boundary plan saved as png file in plots folder
    Accuracy of hidden layer saved info.log file in log folder
    """

    X_train, Y_train, X_test, Y_test, classes = injest(
        "signs_dataset"
    )  # X is (n_samples,n_features) Y is (n_samples, n_label)
    index = 9
    plot_image(
        X_train[index],
        classes[np.argmax(Y_train[index])],
        "signs",  # Find the index of the first occurrence of 1 in Y_train e.g. [0., 0., 0., 0., 1., 0.]
    )  # X_train(600, 64, 64, 3), Y_train(600, 1)

    parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=epocs)

    signs_model = build_conv_model((64, 64, 3))
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    history = signs_model.fit(train_dataset, epochs=epocs, validation_data=test_dataset)
    plot_history(history)
    loss, accuracy = signs_model.evaluate(test_dataset)
    print(f"loss is {loss:.5f} and accuracy is {accuracy:.5f}")
    log(f"loss is {loss:.5f} and accuracy is {accuracy:.5f}")



if __name__ == "__main__":
    cli()
