# Multi-Armed Bandit (MAB) Simulation and Analysis

This project provides a modular framework for simulating and analyzing various multi-armed bandit (MAB) algorithms in both Bernoulli and Gaussian environments. It includes implementations of classic and advanced agents, experiment orchestration, and publication-quality plotting with confidence intervals.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](#environments)
  - [BernoulliEnv](#bernoullienv)
  - [GaussianEnv](#gaussianenv)
- [Agents](#agents)
  - [BaseAgent](#baseagent)
  - [EpsilonGreedyAgent](#epsilongreedyagent)
  - [UCBAgent (KL-UCB)](#ucbagent-kl-ucb)
  - [ThompsonSamplingAgent](#thompsonsamplingagent)
  - [GaussianEpsilonGreedyAgent](#gaussianepsilongreedyagent)
  - [GaussianUCBAgent](#gaussianucbagent)
  - [GaussianThompsonSamplingAgent](#gaussianthompsonsamplingagent)
- [Plotting Utilities](#plotting-utilities)
- [Experiment Pipeline](#experiment-pipeline)
- [Output](#output)
- [Dependencies](#dependencies)

---

## Project Structure

```
MAB/
├── agents/
│   ├── base_agent.py
│   ├── epsilon_greedy.py
│   ├── ucb.py
│   ├── thompson_sampling.py
│   ├── gaussian_epsilon_greedy.py
│   ├── gaussian_ucb.py
│   ├── gaussian_thompson_sampling.py
│   └── __init__.py
├── environments/
│   ├── bernoulli_env.py
│   ├── gaussian_env.py
│   └── __init__.py
├── plots/
│   └── plot_utils.py
├── output/
│   └── (plots, logs, and metrics)
├── main.py
└── requirements.txt
```

---

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run all experiments and generate plots:
```bash
python main.py
```
Plots and logs will be saved in the `output/` directory.

---

## Environments

### `BernoulliEnv`

- **Description:** Each arm returns a reward of 1 with probability \( p_i \), and 0 otherwise.
- **Parameters:** `probabilities` (list or np.array of arm probabilities)
- **Reward:** \( r \sim \text{Bernoulli}(p_i) \)
- **Optimal Reward:** \( \max_i p_i \)

### `GaussianEnv`

- **Description:** Each arm returns a reward drawn from a normal distribution.
- **Parameters:** `means`, `stds` (lists or np.arrays)
- **Reward:** \( r \sim \mathcal{N}(\mu_i, \sigma_i^2) \)
- **Optimal Reward:** \( \max_i \mu_i \)

---

## Agents

### `BaseAgent`

All agents inherit from `BaseAgent`, which defines the interface:
- `select_action()`: Returns the index of the arm to pull.
- `update(arm, reward)`: Updates internal state based on observed reward.
- `reset()`: Resets agent state for a new run.

---

### `EpsilonGreedyAgent`

- **Exploration/Exploitation:** With probability \( \epsilon \), select a random arm; otherwise, select the arm with the highest estimated value.
- **Value Update:** Incremental mean:
  \[
  Q_{n+1} = Q_n + \frac{1}{N} (R - Q_n)
  \]
  where \( Q_n \) is the estimated value, \( N \) is the number of times the arm has been pulled, and \( R \) is the observed reward.

---

### `UCBAgent` (KL-UCB for Bernoulli)

- **Selection Rule:** For each arm, compute the KL-UCB index:
  \[
  \text{KL}(p, q) = p \log\left(\frac{p}{q}\right) + (1-p) \log\left(\frac{1-p}{1-q}\right)
  \]
  The index for arm \( i \) is the largest \( q \in [\hat{p}_i, 1] \) such that:
  \[
  n_i \cdot \text{KL}(\hat{p}_i, q) \leq \log t + c \log \log t
  \]
  where \( \hat{p}_i \) is the empirical mean, \( n_i \) is the number of pulls, \( t \) is the total number of steps, and \( c \) is a tunable parameter.

---

### `ThompsonSamplingAgent` (Bernoulli)

- **Posterior:** Maintains Beta posteriors for each arm.
- **Selection Rule:** For each arm, sample \( \theta_i \sim \text{Beta}(\alpha_i, \beta_i) \) and select the arm with the highest sample.
- **Update:** After observing reward \( r \):
  - If \( r = 1 \): \( \alpha_i \leftarrow \alpha_i + 1 \)
  - If \( r = 0 \): \( \beta_i \leftarrow \beta_i + 1 \)

---

### `GaussianEpsilonGreedyAgent`

- **Same as EpsilonGreedyAgent**, but for Gaussian rewards.

---

### `GaussianUCBAgent` (UCB1-Normal)

- **Selection Rule:** For each arm:
  \[
  \text{UCB}_i = \hat{\mu}_i + c \sqrt{\frac{2 \log t}{n_i}}
  \]
  where \( \hat{\mu}_i \) is the sample mean, \( n_i \) is the number of pulls, \( t \) is the total number of steps, and \( c \) is a tunable parameter.

---

### `GaussianThompsonSamplingAgent`

- **Posterior:** Uses Normal-Inverse-Gamma conjugate prior for each arm.
- **Selection Rule:** For each arm:
  1. Sample variance \( \tau^2 \) from Inverse-Gamma posterior.
  2. Sample mean from Normal posterior:
     \[
     \mu \sim \mathcal{N}(\mu_n, \tau^2 / \lambda_n)
     \]
- **Update:** Updates sufficient statistics for each arm after each reward.

---

## Plotting Utilities

All plotting utilities are in `plots/plot_utils.py`:

- `plot_rewards`: Plots reward per step for a single agent.
- `plot_cumulative_rewards`: Plots cumulative reward per step for a single agent.
- `plot_cumulative_regret`: Plots cumulative regret for all agents, with:
  - Smoothed mean regret line.
  - Dotted line for the upper 95% confidence interval (worst case).
  - Light shading between the mean and upper CI for each agent.

The confidence interval is computed as:
\[
\text{Upper CI} = \text{mean} + 1.96 \times \text{standard error}
\]

---

## Experiment Pipeline

The main experiment logic is in `main.py`:

1. **Setup:** Define environments, agent configurations, and experiment parameters.
2. **Run Experiments:** For each agent and environment, run multiple independent trials.
3. **Aggregate Results:** Compute mean and upper 95% CI of cumulative regret.
4. **Plot:** Generate and save plots in `output/`.

---

## Output

- Plots: `output/bernoulli_cumulative_regret_comparison.pdf/png`, `output/gaussian_cumulative_regret_comparison.pdf/png`
- Logs: `output/metrics.txt`, `output/rewards.txt`, `output/actions.txt`, `output/run_log.txt`

---

## Dependencies

- `numpy`
- `matplotlib`

Install with:
```bash
pip install -r requirements.txt
```

---

## References

- [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning, 47(2-3), 235–256.](https://link.springer.com/article/10.1023/A:1013689704352)
- [Garivier, A., & Cappé, O. (2011). The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond. NIPS.](https://proceedings.neurips.cc/paper/2011/file/1cd138d0480a20c09d6e1b6d2c5cfa32-Paper.pdf)

---