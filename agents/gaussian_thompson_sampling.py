import numpy as np
from .base_agent import BaseAgent

class GaussianThompsonSamplingAgent(BaseAgent):
    """
    Thompson Sampling agent for Gaussian bandit environment using Normal-Inverse-Gamma conjugate prior.
    This implementation maintains posterior parameters for each arm and samples from the posterior predictive distribution.
    """
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.reset()

    def reset(self):
        # Prior parameters for Normal-Inverse-Gamma: mean=0, lambda=1, alpha=1, beta=1 (uninformative)
        self.mu0 = 0.0
        self.lambda0 = 1.0
        self.alpha0 = 1.0
        self.beta0 = 1.0
        self.n = np.zeros(self.n_arms)  # Number of pulls per arm
        self.sum_x = np.zeros(self.n_arms)  # Sum of rewards per arm
        self.sum_x2 = np.zeros(self.n_arms)  # Sum of squared rewards per arm

    def select_action(self) -> int:
        # For each arm, sample mean and variance from the posterior
        samples = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            n = self.n[arm]
            if n == 0:
                # If arm not pulled, sample from prior
                mu_n = self.mu0
                lambda_n = self.lambda0
                alpha_n = self.alpha0
                beta_n = self.beta0
            else:
                # Posterior parameters
                sample_mean = self.sum_x[arm] / n
                lambda_n = self.lambda0 + n
                mu_n = (self.lambda0 * self.mu0 + n * sample_mean) / lambda_n
                alpha_n = self.alpha0 + n / 2
                sum_sq = self.sum_x2[arm]
                beta_n = self.beta0 + 0.5 * (sum_sq - n * sample_mean ** 2) + \
                    (self.lambda0 * n * (sample_mean - self.mu0) ** 2) / (2 * lambda_n)
            # Sample variance from Inverse-Gamma
            tau2 = 1 / np.random.gamma(alpha_n, 1 / beta_n)
            # Sample mean from Normal
            samples[arm] = np.random.normal(mu_n, np.sqrt(tau2 / lambda_n))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        # Update sufficient statistics for the arm
        self.n[arm] += 1
        self.sum_x[arm] += reward
        self.sum_x2[arm] += reward ** 2 