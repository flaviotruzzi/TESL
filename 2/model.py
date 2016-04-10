"""
Provide model generation according to chapter 2.3
"""
import random
import numpy as np
from numpy.random import multivariate_normal, shuffle
import matplotlib.pyplot as plt


def process(mean, covariance=np.identity(2) / 5):
    """
    Sample from multivariate normal using one of the mean vector and
    specified covarianve.
    """
    return multivariate_normal(random.choice(mean), covariance)


class Model(object):
    """
    Generate model according to chapter 2.3.

    This model generates 10 means $m_{kc}$ for each class c.
    For each point it choose one of those means using a uniform distribution,
    and generate a gaussian value using that mean and I/5 as covariance.

    Usage:
        m = Model()
        m.sample()
        m.plot()

        print m.sample()
    """

    def __init__(self, mean=np.asarray([1, 0]), covariance=np.identity(2)):
        self.blue = multivariate_normal(mean.T, covariance, 10)
        self.orange = multivariate_normal(mean[::-1].T, covariance, 10)
        self.blue_samples = []
        self.orange_samples = []

    def plot(self):
        """Plot its internal state."""
        if len(self.blue_samples) == 0:
            self.sample()
        plt.figure()
        plt.scatter(self.blue_samples[:, 0], self.blue_samples[:, 1], color='blue')
        plt.scatter(self.orange_samples[:, 0], self.orange_samples[:, 1], color='orange')
        plt.show()

    def sample(self, sample_size=100):
        """Sample from the model and store its internal state."""
        self.__generate__(sample_size)
        sample = self.blue_samples + self.orange_samples
        shuffle(sample)
        return sample

    def __generate__(self, sample_size):
        """Generate sample.
            n: number of elements for each class
        """
        self.blue_samples = np.asarray([process(self.blue) for _ in range(sample_size)])
        self.orange_samples = np.asarray([process(self.orange) for _ in range(sample_size)])
