import numpy as np

class GaussianEnv:
    """
    Gaussian (Normal) bandit environment. Each arm returns a reward drawn from a normal distribution.
    """
    def __init__(self, means, stds):
        """
        Args:
            means (list or np.array): Mean of each arm.
            stds (list or np.array): Standard deviation of each arm.
        """
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.n_arms = len(means)

    def step(self, arm: int) -> float:
        """Pull the specified arm and return a reward (float)."""
        return np.random.normal(self.means[arm], self.stds[arm])

    def optimal_reward(self) -> float:
        """Return the expected reward of the best arm (highest mean)."""
        return np.max(self.means) 