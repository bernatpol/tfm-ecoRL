import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bandits.bandit_environments import create_env

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


def evaluate_mean_cum_regret(eval_env, agent, num_episodes, initial_lstm_state, verbose=False, eval_reg_formula="penalize_1"):
    """Evaluate episodes as described in eval_env and return results
    
    Returns:
        eval_actions: actions done by the agent
        eval_regrets: regret of the agent given the action
        cumulative_regrets: cum regret (from eval_regrets)
        mean_cumulative_regret
        std_cumulative_regret
        se_cumulative_regret
        logging: dictionary with information with keys "ep_{ep}_step_{step}"
    """
    # Setup evaluation episodes to test the agent on.
    eval_episodes = np.arange(num_episodes)
    logging = {}

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
            if verbose:
                print(f"Step {step}: action={action}, arm1_prob={arm1_prob}, arm2_prob={arm2_prob}, pi_out={pi_out}, v_out={v_out}, reward={reward}")
                print(observation)
            eval_actions[ep, step] = action
            
            logging[f"ep_{ep}_step_{step}"] = {
                'action': action,
                'arm1_prob': arm1_prob,
                'arm2_prob': arm2_prob,
                'pi_out': pi_out,
                'v_out': v_out,
                'observation': observation,
            }

            # Calculate step regret
            chosen_arm_expected_reward = arm1_prob if action == 0 else arm2_prob
            step_regret = optimal_expected_reward - chosen_arm_expected_reward
            eval_regrets[ep, step] = step_regret

            observation = next_observation
            lstm_state = new_lstm_state
            step += 1

    if eval_reg_formula == "penalize_1":
        eval_regrets[np.where(eval_regrets > 0)] = 1
    
    # Compute metrics: cumulative regret R_T(b), R_T(b) mean and std_dev
    cumulative_regrets = np.cumsum(eval_regrets, axis=1)

    # R_T(b) Mean and Standard error (for confidence interval)
    mean_cumulative_regret = np.mean(cumulative_regrets, axis=0)
    std_cumulative_regret = np.std(cumulative_regrets, axis=0)
    se_cumulative_regret = std_cumulative_regret / np.sqrt(num_episodes)

    print("Done evaluation.")
    return eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging

def plot_data_with_uncertainty(x, mean, sd, sem, color='#4daf4a', label='LSTM A2C "Independent"', title="Testing"):
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
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Trial #", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.xlim(0, 20) # Ensure X axis starts exactly at 0 like the image
    plt.legend(loc='upper left', frameon=False, handlelength=1.5)

    ax = plt.gca()
    return ax

def plot_performance_summary_matrix(model_paths, eval_config, provided_evaluation=None, return_provided_evaluation=False, robust=False):
    """Plot a performance summary matrix with the mean cumulative regret for each environment type.
    
    Args:
        model_paths: Dictionary mapping environment types to paths of model results.
            It has the form: {
                "independent_bandit": "path/to/independent_model.pkl",
                "dependent_u_bandit": "path/to/dependent_u_model.pkl",
                "dependent_e_bandit": "path/to/dependent_e_model.pkl",
                "dependent_m_bandit": "path/to/dependent_m_model.pkl",
                "dependent_h_bandit": "path/to/dependent_h_model.pkl",
            }
        eval_config: Evaluation configuration containing num_eval_episodes.
        provided_evaluation: Evaluation results for each environment type, if already computed. It has the form:
            {
                "independent_bandit": {
                    "independent" : (eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging),
                    "dependent_u" : (eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging),
                    "dependent_e" : (eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging),
                    "dependent_m" : (eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging),
                    "dependent_h" : (eval_actions, eval_regrets, cumulative_regrets, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging), 
                    },
                "dependent_u_bandit": {...},
                "dependent_e_bandit": {...},
                "dependent_m_bandit": {...},
                "dependent_h_bandit": {...},
            }
    """

    if provided_evaluation is None:
        provided_evaluation = {}
        for model_type in ["independent_bandit", "dependent_u_bandit", "dependent_e_bandit", "dependent_m_bandit", "dependent_h_bandit"]:
            with open(model_paths[model_type], 'rb') as fp:
                training_results = pickle.load(fp)

            agent = training_results[0]['agent']

            # Initialize evaluation environment.
            eval_env = create_env(eval_config.eval_environment)

            # Setup initial LSTM memory state.
            initial_lstm_state = agent.get_initial_lstm_state()

            agent.eval()

            results_mean_cum_regret = {}
            for env_type in ["independent", "dependent_u", "dependent_e", "dependent_m", "dependent_h"]:
                eval_env._reward_structure = env_type
                results_mean_cum_regret[env_type] = evaluate_mean_cum_regret(eval_env, agent, eval_config.num_eval_episodes, initial_lstm_state=initial_lstm_state, verbose=False)
                print(f"{model_type} - {env_type} evaluated \n")
            
            provided_evaluation[model_type] = results_mean_cum_regret
        
    if return_provided_evaluation:
        return provided_evaluation

    # Plotting: Do a matrix with last values of mean cumulative regret for each environment type and model type

    plot_df = pd.DataFrame({
        model_type.capitalize(): {
            env_type.capitalize(): provided_evaluation[model_type][env_type][3][-1]
            for env_type in ["independent", "dependent_u", "dependent_e", "dependent_m", "dependent_h"]
        }
        for model_type in ["independent_bandit", "dependent_u_bandit", "dependent_e_bandit", "dependent_m_bandit", "dependent_h_bandit"]
    })

    # Plot the performance summary matrix
    sns.heatmap(plot_df.T, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), cbar_kws={"shrink": 0.7}, linewidth=.5, robust=robust)
    plt.title("Cumulative Regret")
    plt.xlabel("Testing Condition", size=12)
    plt.ylabel("Model Training Condition", size=12)
    plt.show()

    return provided_evaluation, plot_df