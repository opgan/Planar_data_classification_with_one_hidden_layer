# pylint: disable=unused-import

import numpy as np
from lib.log import log

# from myLib.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
# from myLib.helper import generate_spiral_planar_dataset


def injest():
    """
    This function loads the flower-shaped dataset

    Argument:
    none

    Returns:
    X -- data matrix (n_features, n_samples) (2, 400) representing 400 points, 2 (x1, x2) coordindates
    Y -- labels  (n_label, n_samples) (1, 400) representing red (0.0) and blue (1.0)
    """
    # Loading data
    X, Y = generate_spiral_planar_dataset()

    # Figure out the dimensions and shapes of the problem
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_Y[1]

    print("The shape of X is: " + str(shape_X))
    print("The shape of Y is: " + str(shape_Y))
    print("There are m = %d training examples!" % (m))

    log("The shape of X is:  " + str(shape_X))
    log("The shape of Y is:  " + str(shape_Y))
    log("There are m = %d training examples!:  " % (m))

    return X, Y


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

    return X, Y