# pylint: disable=unused-import

import numpy as np
from lib.log import log

import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, plot_title):
    """
    This function fits the logistic regression model according to the given training data

    Argument:
    model -- a lambda function takes an input x and returns the predictions made by the trained clf model on that input
    input data X -- (n_features, n_samples)
    true labels y -- (n_label, n_samples)

    Returns:
    a file of plot
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1  # x2
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1  # x1
    h = 0.01
    # Generate a 2D grid of points with distance h between them
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )  # column-wise ... arange() creates an array of evenly spaced values within a given interval.
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])

    # Z = model.predict(
    #    np.c_[xx.ravel(), yy.ravel()]
    # )  # xx must be (n_features, n_samples) ... np.c[] concatenates arrays column-wise ..  ravel() returns a contiguous flattened array containing the elements of the input array

    Z = Z.reshape(xx.shape)  # column-wise (n_samples, n_features)
    # Plot the contour and training examples
    plt.contourf(
        xx, yy, Z, cmap=plt.colormaps.get_cmap("viridis")
    )  # Create filled contour plot
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.colormaps.get_cmap("coolwarm"))
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.title(plot_title)
    plt.savefig("plots/decision_boundary" + plot_title + ".png")


def plot(X, Y):
    """
    Plot data

    Arguments:
    X -- (2, 400) array
    Y -- (1, 400) array

    Return:
    plot saved into a .png file in folder plots
    """

    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.colormaps.get_cmap("viridis"))
    plt.grid(True)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Flower Data")
    # plt.legend(handles=[plt.scatter([], [], c=i, label=f"Class {i}") for i in np.unique(Y)])
    plt.savefig("plots/flower_dataset.png")
