from lib.data import injest
from lib.helper import fit_logistic_regression_model
from lib.helper import compute_accuracy
from lib.plot import plot
from lib.plot import plot_decision_boundary

X, Y = injest()  # X is (n_features, n_samples) Y is (n_label, n_samples)
plot(X, Y)
clf = fit_logistic_regression_model(X, Y)
plot_decision_boundary(clf, X, Y)
compute_accuracy(clf, X, Y)
