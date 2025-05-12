import matplotlib.pyplot as plt
import os
import numpy as np

def plot_rewards(rewards, agent_name, output_dir):
    plt.figure()
    plt.plot(rewards, label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'Rewards over Time - {agent_name}')
    plt.legend()
    pdf_path = os.path.join(output_dir, f'{agent_name}_rewards.pdf')
    png_path = os.path.join(output_dir, f'{agent_name}_rewards.png')
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()

def plot_cumulative_rewards(cum_rewards, agent_name, output_dir):
    plt.figure()
    plt.plot(cum_rewards, label='Cumulative Reward')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Cumulative Rewards - {agent_name}')
    plt.legend()
    pdf_path = os.path.join(output_dir, f'{agent_name}_cumulative_rewards.pdf')
    png_path = os.path.join(output_dir, f'{agent_name}_cumulative_rewards.png')
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()

def moving_average(data, window_size=10):
    """
    Compute the moving average of a 1D array.
    Args:
        data (array-like): Input data.
        window_size (int): Window size for smoothing.
    Returns:
        np.ndarray: Smoothed data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_cumulative_regret(all_regrets, agent_names, output_dir, smooth_window=10, plot_name='cumulative_regret_comparison', linewidth=1.5, upper_cis=None):
    """
    Plot cumulative regret for all agents on a single plot, with optional smoothing and adjustable line width.
    Optionally plot the upper 95% confidence interval as a dotted line (no shading).
    If upper_cis is provided, shade the area between the main line and the upper CI.
    Args:
        all_regrets (list of list): Each sublist is the mean regret per step for an agent.
        agent_names (list of str): Names of the agents.
        output_dir (str): Directory to save the plot.
        smooth_window (int): Window size for moving average smoothing.
        plot_name (str): Base name for the plot files.
        linewidth (float): Width of the plot lines.
        upper_cis (list of list or None): Each sublist is the upper 95% CI per step for an agent (same shape as all_regrets).
    """
    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    for i, (regrets, name) in enumerate(zip(all_regrets, agent_names)):
        smoothed = moving_average(regrets, window_size=smooth_window)
        x = range(len(smoothed))
        plt.plot(x, smoothed, label=name, color=colors[i % len(colors)], linewidth=linewidth)
        if upper_cis is not None:
            upper = moving_average(upper_cis[i], window_size=smooth_window)
            plt.plot(x, upper, linestyle=':', color=colors[i % len(colors)], linewidth=2, label=f'{name} 95% CI (upper)')
            plt.fill_between(x, smoothed, upper, color=colors[i % len(colors)], alpha=0.15)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cumulative Regret (Smoothed)', fontsize=12)
    plt.title('Cumulative Regret Comparison (Averaged over Runs)', fontsize=14)
    plt.legend(fontsize=11, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{plot_name}.pdf')
    png_path = os.path.join(output_dir, f'{plot_name}.png')
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close() 