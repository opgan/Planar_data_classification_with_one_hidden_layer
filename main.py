# pylint: disable=unnecessary-lambda, unnecessary-pass
#!/usr/bin/env python3
# python main.py


from lib.data import injest
from lib.helper import fit_logistic_regression_model
from lib.helper import compute_accuracy
from lib.plot import plot
from lib.plot import plot_decision_boundary
from lib.one_hidden_layer_nn import nn_model
from lib.one_hidden_layer_nn import predict
from lib.plot import plot_costs

# import numpy as np
# from lib.log import log
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

    X, Y = injest()  # X is (n_features, n_samples) Y is (n_label, n_samples)
    plot(X, Y)

    # Build a model with linear regression
    clf = fit_logistic_regression_model(X, Y)
    plot_title = "linear regression "
    plot_decision_boundary(lambda x: clf.predict(x), X, Y, plot_title)
    compute_accuracy(
        lambda x: clf.predict(x.T).reshape(-1, 1).T, X, Y, plot_title
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

    X, Y = injest()  # X is (n_features, n_samples) Y is (n_label, n_samples)
    plot(X, Y)

    parameters, costs = nn_model(X, Y, n_h, num_iterations=10000, print_cost=False)
    plot_title = "hidden layer size " + str(n_h)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y, plot_title)
    compute_accuracy(lambda x: predict(parameters, x), X, Y, plot_title)
    plot_costs(costs)

if __name__ == "__main__":
    cli()
