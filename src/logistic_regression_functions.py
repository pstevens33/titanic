import numpy as np
import pdb


def predict_proba(X, coeffs):
    """Calculate the predicted conditional probabilities (floats between 0 and
    1) for the given data with the given coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.

    Returns
    -------
    predicted_probabilities: The conditional probabilities from the logistic
        hypothesis function given the data and coefficients.

    """
    return 1 / (1 + np.exp(- X.dot(coeffs.T)))


def predict(X, coeffs, thresh=0.5):
    """
    Calculate the predicted class values (0 or 1) for the given data with the
    given coefficients by comparing the predicted probabilities to a given
    threashold.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.
    threas: Threashold for comparison of probabilities.

    Returns
    -------
    predicted_class: The predicted class.
    """
    return np.where(predict_proba(X, coeffs) > thresh, 1, 0)


def cost(X, y, coeffs, lam=0):
    """
    Calculate the logistic cost function of the data with the given
    coefficients.

    If lam != 0, then we assume an intercept has been added to the data.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    y: A 1 dimensional numpy array.  The actual class values of the response.
        Must be encoded as 0's and 1's.  Also, must align properly with X and
        coeffs.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X, y, and coeffs must align.

    Returns
    -------
    logistic_cost: The computed logistic cost.
    """
    coeffsz = np.copy(coeffs)
    coeffsz[0] = 0

    h = predict_proba(X, coeffs)
    cv = y*np.log(h) + (1-y)*np.log(1-h) + lam * coeffsz.T.dot(coeffsz)
    return -cv.sum()


def gradient(X, y, coeffs, lam=0):
    """
    Calculate the gradient of the cost function of the data with the given coefficients.

    If lam != 0, then we assume an intercept has been added to the data.
    """
    coeffsz = np.copy(coeffs)
    coeffsz[0] = 0

    h = predict_proba(X, coeffs)
    grad = (h - y).dot(X) + 2*lam*coeffsz
    return grad

def gradient_stochastic(X, y, coeffs, lam=0):
    """
    Calculate the gradient of the cost function of the data with the given coefficients.

    If lam != 0, then we assume an intercept has been added to the data.
    """
    coeffsz = np.copy(coeffs)
    coeffsz[0] = 0

    h = predict_proba(X, coeffs)
    grad = (h - y).dot(X) + 2*lam*coeffsz
    return grad


def add_intercept(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))

def standardize(X):
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X

def cost_regularized(X, y, coeffs):
    return cost(X, y, coeffs, lam=1)

def gradient_regularized(X, y, coeffs):
    return gradient(X, y, coeffs, lam=1)
