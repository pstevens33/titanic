import numpy as np
import logistic_regression_functions as f
from sklearn.utils import shuffle
import pdb


class GradientDescent(object):
    """
    Preform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict_func,
                 alpha=0.058,
                 num_iterations=10000, fit_intercept=True, standardize=True):
        """
        Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimizaiton has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """

        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.standardize = standardize

    def fit(self, X, y, delta=0.001):
        """
        Run the gradient descent algorithm for num_iterations repititions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        X1 = np.copy(X)

        if self.standardize:
            X1 = f.standardize(X1)

        if self.fit_intercept:
            X1 = f.add_intercept(X1)

        self.coeffs = np.ones(X1.shape[1])
        prev_cost = self.cost(X1, y, self.coeffs)

        for i in range(self.num_iterations):
            # Update coefficients according to cost gradient using all points
            self.coeffs = self.coeffs - self.alpha * self.gradient(X1, y, self.coeffs)

            # Calculate current cost
            cost = self.cost(X1, y, self.coeffs)
            diff = prev_cost - cost

            if 0 <= diff < delta:
                print('Converged after {} iterations'.format(i))
                break

            prev_cost = cost
            # print('Cost: {}'.format(self.cost(X, y, self.coeffs)))

        return self


    def fit_stochastic(self, X, y, delta=0.001):
        """
        Run the gradient descent algorithm for num_iterations repititions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        # Copy and shuffle the training data
        X1, y1 = shuffle(np.copy(X), np.copy(y), random_state=0)

        if self.standardize:
            X1 = f.standardize(X1)

        if self.fit_intercept:
            X1 = f.add_intercept(X1)

        # Make coefficients a 1xp array
        self.coeffs = np.ones(X1.shape[1]).reshape(1, X1.shape[1])
        # Set prev_cost to max float value
        prev_cost = float('inf')

        for i in range(self.num_iterations):
            j = i % y1.shape[0]
            # Select the jth row of X1, shape to 1xp
            X1j = np.array([ X1[j,:] ])
            # Select the jth element of y1, shape to 1x1
            y1j = np.array([[ y1[j] ]])

            # Update coefficients according to cost gradient computed from X1
            self.coeffs = self.coeffs - self.alpha * self.gradient(X1j, y1j, self.coeffs)

            # Calculate current cost
            cost = self.cost(X1j, y1j, self.coeffs)
            diff = prev_cost - cost

            if 0 <= diff < delta:
                print('Converged after {} iterations'.format(i))
                break

            prev_cost = cost
            # print('Cost: {}'.format(self.cost(X, y, self.coeffs)))

        # Reshape coefficients to p-length vector
        self.coeffs = self.coeffs.reshape(X1.shape[1])
        return self


    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        X1 = np.copy(X)

        if self.standardize:
            X1 = f.standardize(X1)

        if self.fit_intercept:
            X1 = f.add_intercept(X)

        return self.predict_func(X1, self.coeffs, thresh=0.73)
