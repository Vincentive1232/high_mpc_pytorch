"""
Gaussian Policy with a vector of mean and covariances
"""
import os
import numpy as np

import matplotlib.pyplot as plt

class GaussianPolicy(object):
    def __init__(self, act_dim, sigma0=100.0, clip_value=None):
        self.act_dim = act_dim
        self.mu = np.array([0.5])            # define the mean of the Gaussian Distribution
        self.cov = np.diag([1.0]) * sigma0   # define the covariance of the Gaussian Distribution

        self.clip_low = clip_value[0]
        self.clip_high = clip_value[1]


    def __call__(self, *args, **kwargs):
        act = self.mu
        return np.clip(act, self.clip_low, self.clip_high)


    def sample(self):
        act = np.random.multivariate_normal(mean=self.mu, cov=self.cov)
        return np.clip(act, self.clip_low, self.clip_high)


    def fit(self, Weights, Actions):
        """
        Update policy parameters via Reward-Weighted Maximum Likelihood

        Args:
            Weights: Weights given by the reward function
            Actions: Actions of the system

        Return:
        """
        self.mu = Weights.dot(Actions) / np.sum(Weights+1e-8)
        Z = (np.sum(Weights)**2 - np.sum(Weights**2)) / np.sum(Weights)
        self.cov = np.sum([Weights[i]*(np.outer((Actions[i]-self.mu), (Actions[i]-self.mu))) for i in range(len(Weights))], 0)/Z


    def save_weight(self, save_dir, i=0):
        """
        Update policy parameters via Reward-Weighted Maximum Likelihood

        Args:
            save_dir: directory for saving the training weights
            i: division of weights

        Return:
        """
        save_dir = save_dir + "/policy_weights"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        weights_path = save_dir + "/weights_{0}.h5".format(i)
        np.savez(weights_path, mu=self.mu, scale=self.scale)


    def load_dir(self, file_path):
        npzfile = np.load(file_path)
        self.mu = npzfile['mu']
        self.cov = npzfile['scale']
