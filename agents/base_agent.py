from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.reset()

    @abstractmethod
    def select_action(self) -> int:
        """Selects an arm to pull."""
        pass

    @abstractmethod
    def update(self, arm: int, reward: float):
        """Updates the agent's knowledge based on the action and reward."""
        pass

    def reset(self):
        """Resets the agent's state for a new run."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms) 