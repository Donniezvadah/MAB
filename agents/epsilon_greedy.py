import numpy as np
from .base_agent import BaseAgent

class EpsilonGreedyAgent(BaseAgent):
    """
    Epsilon-Greedy agent for the multi-armed bandit problem.
    With probability epsilon, selects a random arm (exploration), otherwise selects the best-known arm (exploitation).
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_action(self) -> int:
        """
        Select an arm to pull using epsilon-greedy strategy.
        Returns:
            int: Index of the selected arm.
        """
        if np.random.rand() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best average reward
            return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        """
        Update the estimated value for the selected arm using incremental mean.
        Args:
            arm (int): The arm that was pulled.
            reward (float): The observed reward (0 or 1).
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] += (reward - value) / n 