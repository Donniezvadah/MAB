Perfect — here’s the **entire `README.md` content in one continuous code block** for you to copy and paste directly, no interruptions or comments:

````markdown
# Multi-Armed Bandit (MAB) Simulation and Analysis

This project provides a modular framework for simulating and analyzing various **multi-armed bandit (MAB)** algorithms in both **Bernoulli** and **Gaussian** reward settings. It is designed for experimentation and benchmarking, offering classic and Bayesian strategies, flexible environment configuration, and publication-ready plotting.

---

## 📂 Table of Contents

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
- [References](#references)

---

## 🗂 Project Structure

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

## ⚙️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/MAB.git
cd MAB
pip install -r requirements.txt
```

---

## ▶️ Usage

To run all experiments and generate comparative plots:

```bash
python main.py
```

All plots, logs, and metrics will be saved in the `output/` directory.

---

## 🌍 Environments

### `BernoulliEnv`

Each arm returns a binary reward sampled from a Bernoulli distribution:

**Reward Function**:

$$
r \sim \text{Bernoulli}(p_i)
$$

**Parameters**:
- `probabilities`: list or `np.array` of arm success probabilities.

**Optimal Reward**:

$$
\max_i p_i
$$

---

### `GaussianEnv`

Each arm returns a real-valued reward drawn from a normal distribution:

**Reward Function**:

$$
r \sim \mathcal{N}(\mu_i, \sigma_i^2)
$$

**Parameters**:
- `means`: list or `np.array` of arm means.
- `stds`: list or `np.array` of arm standard deviations.

**Optimal Reward**:

$$
\max_i \mu_i
$$

---

## 🧠 Agents

All agents inherit from the abstract `BaseAgent` class:

```python
class BaseAgent:
    def select_action(self):
        pass

    def update(self, arm, reward):
        pass

    def reset(self):
        pass
```

---

### `EpsilonGreedyAgent`

Classic strategy balancing exploration and exploitation.

**Strategy**:
- With probability \( \epsilon \), choose a random arm.
- With probability \( 1 - \epsilon \), choose the arm with highest estimated value.

**Value Update Rule**:

$$
Q_{n+1} = Q_n + \frac{1}{N} (R - Q_n)
$$

where:
- \( Q_n \): current estimate
- \( N \): number of times arm pulled
- \( R \): reward observed

---

### `UCBAgent` (KL-UCB for Bernoulli)

A theoretically grounded approach using KL divergence.

**KL Divergence for Bernoulli**:

$$
\text{KL}(p, q) = p \log \left(\frac{p}{q}\right) + (1 - p) \log \left(\frac{1 - p}{1 - q}\right)
$$

**Selection Rule**:

Choose arm \( i \) with the largest \( q \in [\hat{p}_i, 1] \) such that:

$$
n_i \cdot \text{KL}(\hat{p}_i, q) \leq \log t + c \log \log t
$$

where:
- \( \hat{p}_i \): empirical mean
- \( n_i \): number of times arm \( i \) pulled
- \( t \): time step
- \( c \): exploration constant

---

### `ThompsonSamplingAgent` (Bayesian)

Bayesian agent with Beta posteriors.

**Prior**:

$$
\theta_i \sim \text{Beta}(\alpha_i, \beta_i)
$$

**Selection**: sample \( \theta_i \) from posterior and select arm with highest sample.

**Update**:
- \( r = 1 \Rightarrow \alpha_i \leftarrow \alpha_i + 1 \)
- \( r = 0 \Rightarrow \beta_i \leftarrow \beta_i + 1 \)

---

### `GaussianEpsilonGreedyAgent`

Same logic as epsilon-greedy but for Gaussian rewards.

**Update Rule**:

$$
\hat{\mu}_{n+1} = \hat{\mu}_n + \frac{1}{N}(r - \hat{\mu}_n)
$$

---

### `GaussianUCBAgent` (UCB1-Normal)

Upper confidence bound for Gaussian arms.

**Selection Rule**:

$$
\text{UCB}_i = \hat{\mu}_i + c \sqrt{\frac{2 \log t}{n_i}}
$$

where:
- \( \hat{\mu}_i \): estimated mean
- \( n_i \): arm count
- \( t \): time
- \( c \): constant

---

### `GaussianThompsonSamplingAgent`

Bayesian strategy for Gaussian rewards.

**Conjugate Prior**:
- Mean: Normal
- Variance: Inverse-Gamma

Let:
- \( n \): number of samples
- \( \bar{x} \): sample mean
- \( S^2 \): sample variance

**Posterior Updates**:

- \( \mu_n = \frac{\lambda_0 \mu_0 + n \bar{x}}{\lambda_0 + n} \)
- \( \lambda_n = \lambda_0 + n \)
- \( \alpha_n = \alpha_0 + \frac{n}{2} \)
- \( \beta_n = \beta_0 + \frac{1}{2}\left(n S^2 + \frac{\lambda_0 n (\bar{x} - \mu_0)^2}{\lambda_0 + n}\right) \)

**Sampling**:

1. Sample \( \tau^2 \sim \text{Inverse-Gamma}(\alpha_n, \beta_n) \)
2. Sample \( \mu \sim \mathcal{N}(\mu_n, \tau^2 / \lambda_n) \)

---

## 📊 Plotting Utilities

Located in `plots/plot_utils.py`.

- `plot_rewards`: Instantaneous reward.
- `plot_cumulative_rewards`: Running sum.
- `plot_cumulative_regret`: With 95% CI.

**Confidence Interval**:

$$
\text{SE} = \frac{\sigma}{\sqrt{n}}, \quad \text{CI} = \mu \pm 1.96 \cdot \text{SE}
$$

---

## 🧪 Experiment Pipeline

In `main.py`:

1. Setup environments
2. Initialize agents
3. Run trials
4. Collect rewards/regret
5. Plot results

---

## 🗂 Output

Saved to `output/`:

- Plots: `.pdf`, `.png`
- Logs: `metrics.txt`, `rewards.txt`, `actions.txt`, `run_log.txt`

---

## 📦 Dependencies

Install:

```bash
pip install numpy matplotlib
```

Or use the provided:

```bash
pip install -r requirements.txt
```

---

## 📚 References

- Auer, P. et al. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*. [Springer Link](https://link.springer.com/article/10.1023/A:1013689704352)
- Garivier, A. & Cappé, O. (2011). *The KL-UCB Algorithm*. [NeurIPS Paper](https://proceedings.neurips.cc/paper/2011/file/1cd138d0480a20c09d6e1b6d2c5cfa32-Paper.pdf)

---

## 🔬 Future Work

- Add contextual bandits
- Add EXP3 for adversarial settings
- Animate reward distributions
- Write tutorial notebooks
````

Let me know if you want this turned into a downloadable `.md` file or automatically synced to a GitHub repo.
