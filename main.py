from myLib.data import injest
from myLib.helper import fit_logistic_regression_model

X, Y = injest()
clf=fit_logistic_regression_model(X, Y)
