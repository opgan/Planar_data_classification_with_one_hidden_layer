from myLib.data import injest
from myLib.helper import fit_logistic_regression_model
from myLib.helper import plot_decision_boundary
from myLib.helper import compute_accuracy

X, Y = injest()  # X is (n_features, n_samples) Y is (n_label, n_samples)
clf = fit_logistic_regression_model(X, Y)
plot_decision_boundary(clf, X, Y)
compute_accuracy(clf, X, Y)
