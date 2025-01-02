# pylint: disable=unused-import

import numpy as np
from lib.log import log
import h5py
from lib.helper import convert_to_one_hot
from lib.helper import normalize
from lib.helper import one_hot_matrix
import tensorflow as tf

# from myLib.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
# from myLib.helper import generate_spiral_planar_dataset


def injest(dataset_name):
    """
    This function loads the flower-shaped dataset

    Argument:
    dataset_name -- spiral_planar_dataset or happyface_dataset

    Returns:
    X -- data matrix (n_features, n_samples) (2, 400) representing 400 points, 2 (x1, x2) coordindates
    Y -- labels  (n_label, n_samples) (1, 400) representing red (0.0) and blue (1.0)
    """

    # Global variable
    X = []
    Y = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    classes = []

    # Loading data
    if dataset_name == "spiral_planar_dataset":
        X, Y = generate_spiral_planar_dataset()
        X_train = X
        Y_train = Y
        X_test = X
        Y_test = Y
    elif dataset_name == "happy_face_dataset":
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = (
            load_happy_dataset()
        )
        # Normalize image vectors
        X_train = X_train_orig / 255.0
        X_test = X_test_orig / 255.0

        # Reshape
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T

        # labels description
        classes = ["not smiling", "smiling"]

    elif dataset_name == "signs_dataset":
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = (
            load_signs_dataset()
        )

        # Normalize image vectors
        X_train = X_train_orig / 255.0
        X_test = X_test_orig / 255.0

        # Reshape
        Y_train = convert_to_one_hot(Y_train_orig, 6).T
        Y_test = convert_to_one_hot(Y_test_orig, 6).T

        # labels description
        classes = ["zero", "one", "two", "three", "four", "five"]

    elif dataset_name == "tensorflow_dataset":
        X_train, Y_train, X_test, Y_test, classes = (
            load_signs_dataset_in_tensorflow_dataset()
        )



        # labels description classes is set (do not allow duplicate), while not list (allows duplicate)
        #classes = ["zero", "one", "two", "three", "four", "five"]


    # Figure out the dimensions and shapes of the problem
    if dataset_name == "tensorflow_dataset":
        shape_X_train = X_train.element_spec.shape
        shape_Y_train = Y_train.element_spec.shape
        m_train = X_train.cardinality().numpy()
        shape_X_test = X_test.element_spec.shape
        shape_Y_test = Y_test.element_spec.shape
        m_test = X_test.cardinality().numpy()        
    else:
        shape_X_train = X_train.shape
        shape_Y_train = Y_train.shape
        m_train = shape_Y_train[0]
        shape_X_test = X_test.shape
        shape_Y_test = Y_test.shape
        m_test = shape_Y_test[0]


    print(f"The shape of X train is: {shape_X_train}")
    print(f"The shape of Y train is: {shape_Y_train}")
    print(f"number of training examples =: {m_train}")

    log(f"The shape of X train is: {shape_X_train}")
    log(f"The shape of Y train is: {shape_Y_train}")
    log(f"number of training examples =: {m_train}")



    print(f"The shape of X test is: {shape_X_test}")
    print(f"The shape of Y test is: {shape_Y_test}")
    print(f"number of testing examples =: {m_test}")
    log(f"The shape of X test is: {shape_X_test}")
    log(f"The shape of Y test is: {shape_Y_test}")
    log(f"number of testing examples =: {m_test}")
 
    print(f"There are {len(classes)} classes: {classes}")
    log(f"There are {len(classes)} classes: {classes}")

    return X_train, Y_train, X_test, Y_test, classes

def load_signs_dataset_in_tensorflow_dataset():
    """
    This function loads signs dataset in tensorflow format from file stored in datasets/ directory under the root directory of main.py

    Argument:
    none

    Returns:
    X_train --  
    Y_train --  
    X_test --  
    Y_test --  
    """
    train_dataset = h5py.File("datasets/train_signs.h5", "r")
    test_dataset = h5py.File("datasets/test_signs.h5", "r")

    X_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    Y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])
    X_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    Y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    classes = set()
    for element in Y_train:
        classes.add(element.numpy())

    return X_train, Y_train, X_test, Y_test, classes


def load_signs_dataset():
    """
    This function loads signs dataset from file stored in datasets/ directory under the root directory of main.py

    Argument:
    none

    Returns:
    X_train -- (1080, 64, 64, 3)
    Y_train -- (1080, 6)
    X_test -- (120, 64, 64, 3)
    Y_test -- (120, 6)
    """
    train_dataset = h5py.File("datasets/train_signs.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("datasets/test_signs.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_happy_dataset():
    """
    This function loads happy_dataset from file stored in datasets/ directory under the root directory of main.py

    Argument:
    none

    Returns:
    X_train -- (600, 64, 64, 3)
    Y_train -- (600, 1)
    X_test -- (150, 64, 64, 3)
    Y_test -- (150, 1)
    """
    train_dataset = h5py.File("datasets/train_happy.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("datasets/test_happy.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


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
