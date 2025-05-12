import os
import numpy as np
from agents.epsilon_greedy import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson_sampling import ThompsonSamplingAgent
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from environments.bernoulli_env import BernoulliEnv
from environments.gaussian_env import GaussianEnv
from plots.plot_utils import plot_cumulative_regret

OUTPUT_DIR = 'output'
N_ARMS = 5
N_STEPS = 2000  # Increased horizon
N_RUNS = 50  # Number of independent runs for averaging

# Bernoulli environment parameters
BERNOULLI_PROBABILITIES = [0.1, 0.3, 0.5, 0.7, 0.9]
# Gaussian environment parameters
GAUSSIAN_MEANS = [0.0, 0.5, 1.0, 1.5, 2.0]
GAUSSIAN_STDS = [1.0, 1.0, 1.0, 1.0, 1.0]

# Agent class lists for each environment
env_agent_configs = {
    'bernoulli': [
        ("EpsilonGreedy", EpsilonGreedyAgent, {'epsilon': 0.1}),
        ("KL-UCB", UCBAgent, {'c': 3}),
        ("ThompsonSampling", ThompsonSamplingAgent, {}),
    ],
    'gaussian': [
        ("GaussianEpsilonGreedy", GaussianEpsilonGreedyAgent, {'epsilon': 0.1}),
        ("GaussianUCB", GaussianUCBAgent, {'c': 2}),
        ("GaussianThompsonSampling", GaussianThompsonSamplingAgent, {}),
    ]
}

def run_agent(agent, env, n_steps):
    """
    Run a single agent in the environment for n_steps.
    Returns actions, rewards, cumulative rewards, and regrets per step.
    """
    actions = []
    rewards = []
    optimal = env.optimal_reward()
    regrets = []
    cum_rewards = []
    total_reward = 0
    agent.reset()
    for t in range(n_steps):
        action = agent.select_action()
        reward = env.step(action)
        agent.update(action, reward)
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        cum_rewards.append(total_reward)
        regrets.append(optimal - reward)
    return actions, rewards, cum_rewards, regrets

def run_experiment(env_type):
    """
    Run the experiment for a given environment type ('bernoulli' or 'gaussian').
    Plots the cumulative regret for all agents in that environment.
    """
    if env_type == 'bernoulli':
        env_class = BernoulliEnv
        env_args = (BERNOULLI_PROBABILITIES,)
        plot_name = 'bernoulli_cumulative_regret_comparison'
    elif env_type == 'gaussian':
        env_class = GaussianEnv
        env_args = (GAUSSIAN_MEANS, GAUSSIAN_STDS)
        plot_name = 'gaussian_cumulative_regret_comparison'
    else:
        raise ValueError('Unknown environment type')

    agent_configs = env_agent_configs[env_type]
    all_mean_regrets = []
    all_upper_cis = []
    agent_names = []
    for agent_name, AgentClass, agent_kwargs in agent_configs:
        regrets_runs = []
        for run in range(N_RUNS):
            env = env_class(*env_args)
            agent = AgentClass(N_ARMS, **agent_kwargs)
            _, _, _, regrets = run_agent(agent, env, N_STEPS)
            regrets_runs.append(regrets)
        regrets_runs = np.array(regrets_runs)  # shape: (N_RUNS, N_STEPS)
        cumulative_regrets = np.cumsum(regrets_runs, axis=1)  # shape: (N_RUNS, N_STEPS)
        mean_cum_regret = np.mean(cumulative_regrets, axis=0)
        stderr_cum_regret = np.std(cumulative_regrets, axis=0, ddof=1) / np.sqrt(N_RUNS)
        upper_ci = mean_cum_regret + 1.96 * stderr_cum_regret
        all_mean_regrets.append(mean_cum_regret)
        all_upper_cis.append(upper_ci)
        agent_names.append(agent_name)
    # Plot cumulative regret for all agents on one plot (smoothed by averaging)
    plot_cumulative_regret(all_mean_regrets, agent_names, OUTPUT_DIR, smooth_window=10, plot_name=plot_name, linewidth=1.5, upper_cis=all_upper_cis)

def main():
    # Run for Bernoulli environment
    run_experiment('bernoulli')
    # Run for Gaussian environment
    run_experiment('gaussian')

if __name__ == "__main__":
    main() 