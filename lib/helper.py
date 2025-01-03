# pylint: disable=unused-import

import numpy as np
from lib.log import log
import sklearn
import sklearn.linear_model


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def compute_accuracy(model, X, y, plot_title):
    """
    This function computes train set accuracy

    Argument:
    model -- returns the predictions made by the trained clf model on that input
    input data X -- (n_features, n_samples)
    true labels y -- (n_label, n_samples)

    Returns:
    accuracy result saved in the mylog.file
    """
    y_predictions = model(X)  # X.T (n_samples, n_features) is given

    y_predictions = y_predictions.reshape(
        1, y_predictions.shape[1]
    ).T  # transpose for np.dot operation

    acc = (
        (np.dot(y, y_predictions) + np.dot(1 - y, 1 - y_predictions))
        / float(y.size)
        * 100
    )
    accuracy_string = (
        "Accuracy of "
        + plot_title
        + " :"
        + str(np.squeeze(acc))
        + "%% (percentage of correctly labelled datapoints)"
    )

    print(accuracy_string)
    log(accuracy_string)


def compute_accuracy2(model, X, y):
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
    log(accuracy_string)


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
