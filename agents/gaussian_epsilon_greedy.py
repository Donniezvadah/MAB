import numpy as np
from .base_agent import BaseAgent

class GaussianEpsilonGreedyAgent(BaseAgent):
    """
    Epsilon-Greedy agent for Gaussian bandit environment.
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] += (reward - value) / n 