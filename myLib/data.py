# pylint: disable=unused-import


import numpy as np
from myLib.mylog import log_to_file

# from myLib.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from myLib.helper import generate_spiral_planar_dataset


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

    log_to_file("The shape of X is:  " + str(shape_X))
    log_to_file("The shape of Y is:  " + str(shape_Y))
    log_to_file("There are m = %d training examples!:  " % (m))

    return X, Y
