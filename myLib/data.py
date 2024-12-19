# pylint: disable=E1101, unused-import


import numpy as np
from myLib.mylog import log_to_file
from myLib.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt

def injest():
    # Loading data
    X, Y = load_planar_dataset()

    # Figure out the dimensions and shapes of the problem 
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_Y[1]

    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('There are m = %d training examples!' % (m))

    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.colormaps.get_cmap("viridis"))
    plt.savefig('myPlots/load_planar_dataset.png')

    return X, Y 

