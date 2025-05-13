import os
import numpy as np
import matplotlib.pyplot as plt
from agents.epsilon_greedy import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson_sampling import ThompsonSamplingAgent
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from agents.llm_agent import LLMAgent
from environments.bernoulli_env import BernoulliEnv
from environments.gaussian_env import GaussianEnv

OUTPUT_DIR = 'delayed_feedback_experiments'
METRICS_FILE = os.path.join(OUTPUT_DIR, 'metrics.txt')

# Set your OpenAI API key as an environment variable or in llm_api.txt
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
if OPENAI_API_KEY is None and os.path.exists('llm_api.txt'):
    with open('llm_api.txt') as f:
        OPENAI_API_KEY = f.read().strip()

# Time horizons to test
TIME_HORIZONS = [1, 5, 10, 50, 100, 1000, 2000]
N_ARMS = 5
N_RUNS = 10  # Number of runs per setting
DELAY = 5    # Feedback delay (can be parameterized)

# Bernoulli and Gaussian settings (can be imported from config if needed)
BERNOULLI_PROBABILITIES = [0.1, 0.3, 0.5, 0.7, 0.9]
GAUSSIAN_MEANS = [0.0, 0.5, 1.0, 1.5, 2.0]
GAUSSIAN_STDS = [1.0, 1.0, 1.0, 1.0, 1.0]

# Agent configs
env_agent_configs = {
    'bernoulli': [
        ("EpsilonGreedy", EpsilonGreedyAgent, {'epsilon': 0.05}),
        ("KL-UCB", UCBAgent, {'c': 3}),
        ("ThompsonSampling", ThompsonSamplingAgent, {}),
    ],
    'gaussian': [
        ("GaussianEpsilonGreedy", GaussianEpsilonGreedyAgent, {'epsilon': 0.1}),
        ("GaussianUCB", GaussianUCBAgent, {'c': 2}),
        ("GaussianThompsonSampling", GaussianThompsonSamplingAgent, {}),
    ]
}
if OPENAI_API_KEY:
    for env in env_agent_configs:
        env_agent_configs[env].append(
            ("LLMAgent", LLMAgent, {'api_key': OPENAI_API_KEY, 'model': 'o3-mini', 'horizon': max(TIME_HORIZONS)})
        )

def run_agent_delayed(agent, env, n_steps, delay):
    actions = []
    rewards = [None] * n_steps
    optimal = env.optimal_reward()
    regrets = [None] * n_steps
    cum_rewards = []
    total_reward = 0
    agent.reset()
    feedback_queue = []  # (step, arm)
    for t in range(n_steps):
        action = agent.select_action()
        actions.append(action)
        feedback_queue.append((t, action))
        # Provide feedback for actions taken 'delay' steps ago
        if t >= delay:
            feedback_t, feedback_arm = feedback_queue[t - delay]
            reward = env.step(feedback_arm)
            agent.update(feedback_arm, reward)
            rewards[feedback_t] = reward
            regrets[feedback_t] = optimal - reward
            total_reward += reward
        cum_rewards.append(total_reward)
    # Provide feedback for remaining actions after the last step
    for t in range(max(n_steps - delay, 0), n_steps):
        feedback_t, feedback_arm = feedback_queue[t]
        reward = env.step(feedback_arm)
        agent.update(feedback_arm, reward)
        rewards[feedback_t] = reward
        regrets[feedback_t] = optimal - reward
        total_reward += reward
        cum_rewards.append(total_reward)
    return actions, rewards, cum_rewards, regrets

def log_metrics(agent_name, env_type, horizon, run_idx, rewards, regrets):
    with open(METRICS_FILE, 'a') as f:
        f.write(f"{env_type} | {agent_name} | Horizon {horizon} | Run {run_idx} | Cumulative Reward: {np.nansum(rewards)} | Total Regret: {np.nansum(regrets)}\n")

def run_experiment_delayed(env_type):
    if env_type == 'bernoulli':
        env_class = BernoulliEnv
        env_args = (BERNOULLI_PROBABILITIES,)
    elif env_type == 'gaussian':
        env_class = GaussianEnv
        env_args = (GAUSSIAN_MEANS, GAUSSIAN_STDS)
    else:
        raise ValueError('Unknown environment type')
    agent_configs = env_agent_configs[env_type]
    for horizon in TIME_HORIZONS:
        for agent_name, AgentClass, agent_kwargs in agent_configs:
            for run in range(N_RUNS):
                env = env_class(*env_args)
                agent = AgentClass(N_ARMS, **agent_kwargs)
                actions, rewards, cum_rewards, regrets = run_agent_delayed(agent, env, horizon, DELAY)
                log_metrics(agent_name, env_type, horizon, run, rewards, regrets)

def main():
    # Clear metrics file
    with open(METRICS_FILE, 'w') as f:
        f.write('')
    run_experiment_delayed('bernoulli')
    run_experiment_delayed('gaussian')

def run_experiment_table1():
    """
    Run the Table 1-style experiment: for each delay, run both agents and log average regret.
    """
    avg_regrets = {agent_name: [] for agent_name, _, _ in env_agent_configs['bernoulli']}
    # For each delay value
    for delay in [1, 5, 10, 50, 100, 500]:
        for agent_name, AgentClass, agent_kwargs in env_agent_configs['bernoulli']:
            regrets_runs = []
            for run in range(N_RUNS):
                env = BernoulliEnv(BERNOULLI_PROBABILITIES)
                agent = AgentClass(N_ARMS, **agent_kwargs)
                actions, rewards, cum_rewards, regrets = run_agent_delayed(agent, env, 1000, delay)
                # Only sum non-None regrets
                regrets_runs.append(np.nansum(regrets))
            avg_regret = np.mean(regrets_runs)
            avg_regrets[agent_name].append(avg_regret)
    # Compute ratio (UCB/TS)
    ratios = np.array(avg_regrets['KL-UCB']) / np.array(avg_regrets['ThompsonSampling'])
    # Write table to metrics.txt
    with open(METRICS_FILE, 'w') as f:
        f.write("Table 1: Influence of the delay: regret when the feedback is provided every δ steps.\n")
        f.write("| δ | " + " | ".join(str(d) for d in [1, 5, 10, 50, 100, 500]) + " |\n")
        f.write("|---" * (len([1, 5, 10, 50, 100, 500])+1) + "|\n")
        f.write("| UCB | " + " | ".join(f"{r:,.0f}" for r in avg_regrets['KL-UCB']) + " |\n")
        f.write("| TS | " + " | ".join(f"{r:,.0f}" for r in avg_regrets['ThompsonSampling']) + " |\n")
        f.write("| Ratio | " + " | ".join(f"{r:.2f}" for r in ratios) + " |\n")
    # Plot regret vs delay
    plt.figure()
    plt.plot([1, 5, 10, 50, 100, 500], avg_regrets['KL-UCB'], marker='o', label='UCB')
    plt.plot([1, 5, 10, 50, 100, 500], avg_regrets['ThompsonSampling'], marker='o', label='Thompson Sampling')
    plt.plot([1, 5, 10, 50, 100, 500], ratios, marker='o', label='Ratio (UCB/TS)')
    plt.xlabel('Delay (δ)')
    plt.ylabel('Regret / Ratio')
    plt.title('Regret vs Delay (Table 1)')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'regret_vs_delay.png'))
    plt.close()

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Run the Table 1 experiment and plot
    run_experiment_table1() 