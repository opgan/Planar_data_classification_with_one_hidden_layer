# pylint: disable=unused-import

import numpy as np
from myLib.mylog import log_to_file
import sklearn

# import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


def compute_accuracy(model, X, y):
    """
    This function computes train set accuracy

    Argument:
    model -- returns the predictions made by the trained clf model on that input
    input data X -- (n_features, n_samples)
    true labels y -- (n_label, n_samples)

    Returns:
    accuracy result saved in the mylog.file
    """
    y_predictions = model.predict(X.T)  # X.T is (n_samples, n_features)
    accuracy_string = (
        r"Accuracy of logistic regression: %d %% (percentage of correctly labelled datapoints)"
        % float(
            (np.dot(y, y_predictions) + np.dot(1 - y, 1 - y_predictions))
            / float(y.size)
            * 100
        )
    )
    print(accuracy_string)
    log_to_file(accuracy_string)


def plot_decision_boundary(model, X, y):
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
    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )  # xx must be (n_features, n_samples) ... np.c[] concatenates arrays column-wise ..  ravel() returns a contiguous flattened array containing the elements of the input array
    Z = Z.reshape(xx.shape)  # column-wise (n_samples, n_features)
    # Plot the contour and training examples
    plt.contourf(
        xx, yy, Z, cmap=plt.colormaps.get_cmap("viridis")
    )  # Create filled contour plot
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.colormaps.get_cmap("coolwarm"))
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.title("Logistic Regression")
    plt.savefig("plots/decision_boundary.png")


def fit_logistic_regression_model(X, Y):
    """
    This function fits the logistic regression model according to the given training data

    Argument:
    X -- (n_features, n_samples) matrix (2, 400) representing 400 points, 2 (x1, x2) coordindates
    Y -- (n_label, n_samples) matrix (1, 400) representing label: red (0.0) and blue (1.0)

    Returns:
    clf -- classifier object for the fitted LogisticRegressionCV estimator
    """

    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(
        X.T, Y.T.ravel()
    )  # X.T is (n_samples, n_features) Y.T is (n_labels, n_samples) .. ravel() changes shape of y from (n_labels, n_samples) to (,n_samples)
    return clf


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


def generate_spiral_planar_dataset():
    """
    This function generates flower-shaped data points

    Argument:
    none

    Returns:
    X -- data matrix (n_features, n_samples) (2, 400) representing 400 points, 2 (x1, x2) coordindates
    Y -- labels  (n_labels, n_samples) (1, 400) representing red (0.0) and blue (1.0)
    """

    np.random.seed(1)
    m = 400  # number of points
    N = int(m / 2)  # number of points per class
    D = 2  # number of columns in X representing coordinates
    X = np.zeros((m, D))  # column wise coorindates matrix
    Y = np.zeros((m, 1), dtype="uint8")  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(
            N * j, N * (j + 1)
        )  # a range of indices, specifying where to insert the new data points into the X array and Y array

        # t is N evenly spaced numbers between 0 and 3.12 for red points and 3.12 and 6.24 for blue points
        t = (
            np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        )  # theta generate evenly spaced numbers over a specified interval and includes both the start and stop values
        # t = np.linspace(j*3.12,(j+1)*3.12,N) # theta generate evenly spaced numbers over a specified interval and includes both the start and stop values

        r = (
            a * np.sin(4 * t) + np.random.randn(N) * 0.2
        )  # r is the radius, and t is the angle in radians.
        # r = a*np.sin(4*t) # r is the radius, and t is the angle in radians.
        # log_to_file("r is:  " + str(r) )

        # stacking the r*np.sin(t) and r*np.cos(t) arrays horizontally, where each row represents an (x1, x2) coordinate pair
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    # log_to_file("X is:  " + str(X) )
    return X, Y
