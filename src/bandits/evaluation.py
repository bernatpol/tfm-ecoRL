import numpy as np
import matplotlib.pyplot as plt

def evaluate_all_probs(eval_env, agent, initial_lstm_state, verbose=False):
    """Evaluate agent on all arm probabilities from (0,100) to (100, 0)"""
    # Setup evaluation episodes to test the agent on.
    arm1_probs = np.arange(101)/100
    arm2_probs = 1. - arm1_probs

    num_episodes = len(arm1_probs)
    eval_episodes = np.arange(num_episodes)

    # Initialize data to save
    eval_actions = np.zeros((num_episodes, eval_env._steps_per_episode))
    eval_actions[:] = np.nan

    eval_regrets = np.zeros((num_episodes, eval_env._steps_per_episode))

    for (arm1_prob, arm2_prob, ep) in zip(arm1_probs, arm2_probs, eval_episodes):

        step = 0
        done = False
        lstm_state = initial_lstm_state
        observation = eval_env.reset(arm_probs=np.array([arm1_prob, arm2_prob]))

        # Maximal expected reward
        optimal_expected_reward = max(arm1_prob, arm2_prob)
        if verbose:
          print(f"Ep {ep}: max_exp_reward = {optimal_expected_reward}")

        while not done:
            action, pi_out, v_out, new_lstm_state, _ = agent.get_action(observation, lstm_state)
            next_observation, reward, done, info = eval_env.step(action)

            # Save data
            eval_actions[ep, step] = action

            # Calculate step regret
            chosen_arm_expected_reward = arm1_prob if action == 0 else arm2_prob
            step_regret = optimal_expected_reward - chosen_arm_expected_reward
            eval_regrets[ep, step] = step_regret

            observation = next_observation
            lstm_state = new_lstm_state
            step += 1


    # Compute metrics: cumulative regret R_T(b), R_T(b) mean and std_dev
    cumulative_regrets = np.cumsum(eval_regrets, axis=1)

    print("Done evaluation.")
    return eval_actions, eval_regrets, cumulative_regrets


def evaluate_mean_cum_regret(eval_env, agent, num_episodes, initial_lstm_state, verbose=False):
    """Evaluate agent on all arm probabilities from (0,100) to (100, 0)"""
    # Setup evaluation episodes to test the agent on.
    eval_episodes = np.arange(num_episodes)

    # Initialize data to save
    eval_actions = np.zeros((num_episodes, eval_env._steps_per_episode))
    eval_actions[:] = np.nan

    eval_regrets = np.zeros((num_episodes, eval_env._steps_per_episode))

    for ep in eval_episodes:
        step = 0
        done = False
        lstm_state = initial_lstm_state
        observation = eval_env.reset()
        arm1_prob, arm2_prob = eval_env._arm_probs

        # Maximal expected reward
        optimal_expected_reward = max(arm1_prob, arm2_prob)
        if verbose:
            print(f"Ep {ep}: max_exp_reward = {optimal_expected_reward}")

        while not done:
            action, pi_out, v_out, new_lstm_state, _ = agent.get_action(observation, lstm_state)
            next_observation, reward, done, info = eval_env.step(action)

            # Save data
            eval_actions[ep, step] = action

            # Calculate step regret
            chosen_arm_expected_reward = arm1_prob if action == 0 else arm2_prob
            step_regret = optimal_expected_reward - chosen_arm_expected_reward
            eval_regrets[ep, step] = step_regret

            observation = next_observation
            lstm_state = new_lstm_state
            step += 1

    # Compute metrics: cumulative regret R_T(b), R_T(b) mean and std_dev
    cumulative_regrets = np.cumsum(eval_regrets, axis=1)

    # R_T(b) Mean and Standard error (for confidence interval)
    mean_cumulative_regret = np.mean(cumulative_regrets, axis=0)
    std_cumulative_regret = np.std(cumulative_regrets, axis=0)
    se_cumulative_regret = std_cumulative_regret / np.sqrt(num_episodes)

    print("Done evaluation.")
    return eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret

def plot_data_with_uncertainty(x, mean, sd, sem, color='#4daf4a', label='LSTM A2C "Independent"'):
    """
    Plots the mean line with shaded regions for SD and SEM.
    Frame size set to 4:1 ratio (400x100 equivalent).
    """
    # Plot the primary line (matching the LSTM A2C style)
    plt.plot(x, mean, color=color, label=label, linewidth=2)

    # Shade the uncertainty (matching the single band in the reference)
    plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3)

    # Other lines
    # plt.plot(x, mean * 1.2, color='#555555', linestyle='--', linewidth=2, label='Gittins')
    # plt.plot(x, mean * 0.7, color='#888888', linestyle='--', linewidth=2, label='Thompson')
    # plt.plot(x, mean * 1.3, color='#bbbbbb', linestyle='--', linewidth=2, label='UCB')

    # Formatting to match the image
    plt.title("Testing: Independent", fontsize=14, pad=15)
    plt.xlabel("Trial #", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.xlim(0, 100) # Ensure X axis starts exactly at 0 like the image
    plt.legend(loc='upper left', frameon=False, handlelength=1.5)

    ax = plt.gca()
    return ax