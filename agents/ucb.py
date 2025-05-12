import numpy as np
from .base_agent import BaseAgent
from scipy.optimize import bisect

class UCBAgent(BaseAgent):
    """
    KL-UCB agent for Bernoulli bandit problem (sublinear optimality).
    Uses the KL-UCB index for each arm.
    """
    def __init__(self, n_arms: int, c: float = 3):
        super().__init__(n_arms)
        self.c = c
        self.total_counts = 0

    def kl_divergence(self, p, q):
        # KL divergence for Bernoulli distributions
        eps = 1e-15
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def kl_ucb_index(self, p_hat, n, t, c):
        # Find the largest q in [p_hat, 1] such that n * KL(p_hat, q) <= log(t) + c * log(log(t))
        if n == 0:
            return 1.0
        upper_bound = 1.0
        lower_bound = p_hat
        rhs = (np.log(t) + c * np.log(np.log(max(t, 2)))) / n
        def func(q):
            return self.kl_divergence(p_hat, q) - rhs
        try:
            return bisect(func, lower_bound, upper_bound, xtol=1e-6)
        except Exception:
            return upper_bound

    def select_action(self) -> int:
        """
        Select an arm to pull using the UCB strategy.
        Returns:
            int: Index of the selected arm.
        """
        self.total_counts += 1
        t = self.total_counts
        indices = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            p_hat = self.values[arm]
            n = self.counts[arm]
            indices[arm] = self.kl_ucb_index(p_hat, n, t, self.c)
        return int(np.argmax(indices))

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