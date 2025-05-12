import numpy as np

class BernoulliEnv:
    def __init__(self, probabilities):
        """
        probabilities: list or np.array of success probabilities for each arm
        """
        self.probabilities = np.array(probabilities)
        self.n_arms = len(probabilities)

    def step(self, arm: int) -> float:
        """Pulls the specified arm and returns a reward (0 or 1)."""
        return float(np.random.rand() < self.probabilities[arm])

    def optimal_reward(self) -> float:
        """Returns the expected reward of the best arm."""
        return np.max(self.probabilities) 