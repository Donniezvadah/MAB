import numpy as np
from .base_agent import BaseAgent

class GaussianUCBAgent(BaseAgent):
    """
    UCB agent for Gaussian bandit environment using UCB1-Normal formula.
    """
    def __init__(self, n_arms: int, c: float = 2):
        super().__init__(n_arms)
        self.c = c
        self.total_counts = 0
        self.squared_sums = np.zeros(n_arms)  # For variance estimation

    def select_action(self) -> int:
        self.total_counts += 1
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        # UCB1-Normal: mean + c * sqrt(2 * log(t) / n)
        ucb_values = self.values + self.c * np.sqrt(2 * np.log(self.total_counts) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] += (reward - value) / n
        # For completeness, update squared sums (not used in this simple UCB1-Normal)
        self.squared_sums[arm] += reward ** 2 