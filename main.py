# pylint: disable=unnecessary-lambda

from lib.data import injest
from lib.helper import fit_logistic_regression_model
from lib.helper import compute_accuracy
from lib.plot import plot
from lib.plot import plot_decision_boundary
from lib.one_hidden_layer_nn import nn_model
from lib.one_hidden_layer_nn import predict
import numpy as np
from lib.log import log

X, Y = injest()  # X is (n_features, n_samples) Y is (n_label, n_samples)
plot(X, Y)

# Build a model with linear regression
clf = fit_logistic_regression_model(X, Y)
plot_title = "linear regression "
plot_decision_boundary(lambda x: clf.predict(x), X, Y, plot_title)

compute_accuracy(lambda x: clf.predict(x.T).reshape(-1, 1).T, X, Y, plot_title) # X.T is (n_samples, n_features)

"""
predictions = clf.predict(X.T)
predictions = predictions.reshape(-1, 1).T # (400,) to (400,1). after T becomes (1,400)
log("clf shape of y_predictions is:  " + str(predictions.shape))
"""

# Build a model with a n_h-dimensional hidden layer
n_h = 4
parameters, costs = nn_model(X, Y, n_h, num_iterations=10000, print_cost=False)
plot_title = "hidden layer size " + str(n_h)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y, plot_title)
compute_accuracy(lambda x: predict(parameters, x), X, Y, plot_title)

"""
predictions = predict(parameters, X)
log("one shape of y_predictions is:  " + str(predictions.shape))
predictions = predictions.reshape(1, predictions.shape[1]).T # transpose for np.dot operation

log("The shape of y_predictions T is:  " + str(predictions.shape))
acc=(np.dot(Y, predictions) + np.dot(1 - Y, 1 - predictions)) / float(Y.size) * 100
accuracy_string = "Accuracy of " + plot_title + " :" + str(np.squeeze(acc)) + "%% (percentage of correctly labelled datapoints)"
print(accuracy_string)
log(accuracy_string)
"""
