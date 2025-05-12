import numpy as np
from .base_agent import BaseAgent

class ThompsonSamplingAgent(BaseAgent):
    """
    Thompson Sampling agent for Bernoulli bandits using Bayesian updates and probability matching.
    Maintains Beta posteriors for each arm and samples from them to select actions.
    """
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.reset()

    def reset(self):
        """Reset the agent's beliefs (Beta priors) for each arm."""
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)

    def select_action(self) -> int:
        """
        Select an arm to pull using probability matching (sample from Beta posterior for each arm).
        Returns:
            int: Index of the selected arm.
        """
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """
        Update the Beta posterior for the selected arm based on the observed reward.
        Args:
            arm (int): The arm that was pulled.
            reward (float): The observed reward (0 or 1).
        """
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1 