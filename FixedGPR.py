import numpy as np
from numpy.linalg import norm, cholesky
from scipy.linalg import cho_solve


class RBF:
    """
    A radial basis function kernel with fixed variance and length scale
    Kernels are used to compute the covariance matrix of a Gaussian process regressor
    RBF kernels are smooth
    """

    def __init__(self, variance, length_scale):
        self.variance = variance
        self.length_scale = length_scale

    def __call__(self, x, y, *args, **kwargs):
        return self.variance * np.exp(-norm(x - y) ** 2 / (2 * self.length_scale))


class Normal:
    """
    Data class for joint normal distributions
    """

    def __init__(self, mean, cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.dimension = len(self.mean)
        assert (self.cov.shape == (self.dimension, self.dimension)), "Covariance dimension does not match mean " \
                                                                     "dimension "

    def sample(self, number):
        """
        Samples (number) points from the distribution
        Args:
            number: how many points to sample

        Returns: 2-d array of shape (number, dimension)
        """
        out = np.empty((number, self.dimension))
        for i in range(number):
            out[i] = np.random.multivariate_normal(self.mean, self.cov)


class FixedGPR:
    """
    A non-noisy Gaussian process regressor implementation using a fixed RBF kernel
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.training_inputs = None
        self.training_outputs = None
        self._training_size = 0
        self._training_covariance = None
        self._L = None
        self._L_inv = None
        self._Kf = None

    def predict(self, X):
        """
        Returns the means and covariance matrices of the predictive distribution at each input
        Args:
            X: 2d-array of shape (n-points, n-features)

        Returns: 1d-array of shape (n-points) consisting of Normal objects

        """
        number = len(X)
        out = np.empty(number)

        # Construct X covariance (K(X*, X*))
        x_cov = np.empty((number, number))
        for i in range(number):
            for j in range(i):
                x_cov[i, j] = self.kernel(X[i], X[j])
                x_cov[j, i] = x_cov[i, j]

        # Trivial (no prior) case
        if self.training_inputs is None:
            for i in range(len(X)):
                out[i] = Normal([0], x_cov)
            return out

        # Compute pairwise covariance between training and test data (K(X*, X))
        pairwise_covariance = np.empty((number, self._training_size))
        for i in range(number):
            for j in range(self._training_size):
                pairwise_covariance[i, j] = self.kernel(X[i], self.training_inputs[j])

        # Compute predictive distribution
        # pairwise_dot_inverse = pairwise_covariance @ self._inverse_covariance
        # mean = pairwise_dot_inverse @ self.training_outputs
        # conj_inv = pairwise_dot_inverse @ pairwise_covariance.transpose()
        # cov = x_cov - conj_inv


        mean = pairwise_covariance.dot(self._Kf)
        A = self._L_inv * pairwise_covariance.transpose()
        cov = x_cov - (A.transpose() @ A)
        return Normal(mean, cov)

    def fit(self, X, y):
        """
        Sets the training data of the GP and initializes some useful numbers
        Args:
            X: inputs array of shape (n-points, n-features)
            y: outputs array of shape (n-points)

        """
        self.training_inputs = np.array(X)
        self.training_outputs = np.array(y)
        assert (len(self.training_inputs) == len(
            self.training_outputs)), "Training inputs and outputs should have the same length!"
        self._training_size = len(self.training_inputs)

        # Construct covariance of training points (K(X, X))
        self._training_covariance = np.empty((self._training_size, self._training_size))
        for i in range(self._training_size):
            for j in range(i):
                self._training_covariance[i, j] = self.kernel(self.training_inputs[i], self.training_inputs[j])
                self._training_covariance[j, i] = self._training_covariance[i, j]

        # Cholesky here
        self._L = cholesky(self._training_size)
        self._L_inv = cho_solve((self._L, True), np.eye(self._training_size))
        self._Kf = cho_solve((self._L, True), self.training_outputs)

    def add_fit(self, x, y):
        """
        Adds new training points to the training set
        Args:
            X: inputs array of shape (n-points, n-features)
            y: outputs array of shape (n-points)

        """
        self.training_inputs = np.append(self.training_inputs, [x], axis=0)
        self.training_outputs = np.append(self.training_outputs, [y])

        new_training_size = self._training_size + 1

        # Update covariance matrix
        new_covariance = np.empty((new_training_size, new_training_size))
        new_covariance[0:self._training_size, 0:self._training_size] = self._training_covariance

        # Get covariance of new training point
        self._training_covariance[self._training_size, self._training_size] = self.kernel.variance

        # Get covariance of new points with old points
        cross_covariance = np.empty((self._training_size, 1))
        for i in range(self._training_size):
            cov = self.kernel(self.training_inputs[i], x)
            self._training_covariance[i, 0] = cov
            self._training_covariance[0, i] = cov

        # Update Cholesky decomposition using the Cholesky-Banachiewicz algorithm
        new_L = np.zeros((new_training_size, new_training_size))
        new_L[0:self._training_size, 0:self._training_size] = self._L
        for i in range(self._training_size):
            sum = 0
            for j in range(i):
                sum += new_L[self._training_size, j] * new_L[i, j]
            new_L[self._training_size, i] = (self._training_covariance[self._training_size, i] - sum) / new_L[i, i]
        sum = 0
        for i in range(self._training_size):
            sum += new_L[self._training_size, i] ** 2
        new_L[self._training_size, self._training_size] = np.sqrt(self._training_covariance - sum)
        self._L = new_L

        # Update inverse of Cholesky decomposition
        new_L_inv = np.zeros((new_training_size, new_training_size))
        new_L_inv[0:self._training_size, 0:self._training_size] = self._L_inv
        for i in range(new_training_size):
            sum = 0
            for j in range(i):
                sum += new_L_inv[j, i] * self._L[self._training_size, j]
            new_L_inv[self._training_size, i] = -sum / self._L[i, i]
        self._L_inv = new_L_inv

        # Update Kf
        new_Kf = np.zeros(new_training_size)
        new_Kf[0:self._training_size] = self._Kf
        for i in range(new_training_size):
            for j in range(new_training_size):
                new_Kf[i] += self._Kf[j] * self._L_inv[i, self._training_size] * self._L[self._training_size, j]
        self._Kf = new_Kf