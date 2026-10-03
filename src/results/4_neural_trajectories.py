#%%
import sys
import ml_collections
import numpy as np
import pathlib
import pickle
import pandas as pd
import torch
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bandits.bandit_environments import create_env
from utils.plotting_basics import plot_decision_trajectory
from neural_networks import agents_3_RNN

# Map the old module path that pickle expects to the new module
sys.modules['bandits.agents_3_RNN'] = agents_3_RNN

def load_eval_config(training_config=None):
    eval_config = ml_collections.ConfigDict()

    eval_config.phase = 'eval'

    eval_config.num_workers = 0
    eval_config.num_evaluators = 1

    # Saving
    eval_config.dir = OUT_DIR / "bandit_models_rnn" / "dependent_e_models" # "independent", "dependent_{u,e,m,h}"
    eval_config.path = OUT_DIR / "bandit_models_rnn" / "dependent_e_models" / "dependent_e_45.pkl" # "independent", "dependent_{u,e,m,h}"
    eval_config.params_filename = eval_config.path.stem

    # Evaluation
    eval_config.random_seed = 42
    eval_config.num_eval_episodes = 100
    eval_config.log_dynamics = True

    # Evaluation environment
    eval_config.eval_environment = ml_collections.ConfigDict()
    eval_config.eval_environment.env_name = "bandit_eval"
    eval_config.eval_environment.steps_per_episode = int(100)
    eval_config.eval_environment.reward_structure = "dependent_e" # "independent", "dependent_{u,e,m,h}"

    # Agent
    eval_config.agent = ml_collections.ConfigDict()
    if training_config is not None:
        eval_config.agent = training_config["agent"]

    # Training environment
    eval_config.train_environment = ml_collections.ConfigDict()
    if training_config is not None:
        eval_config.train_environment.reward_structure = training_config["environment"]["reward_structure"]

    return eval_config

def load_agent(agent_path):
    with open(agent_path, 'rb') as fp:
        agent_full = pickle.load(fp)[0] # single element in list
    agent = agent_full['agent']
    training_config = agent_full['config']

    return agent, training_config

def evaluate_agent(agent,eval_env,  eval_config, verbose=False, eval_reg_formula="penalize_1"):
    """Evaluate agent on all arm probabilities from (0,100) to (100, 0)"""
    # Setup evaluation episodes to test the agent on.
    eval_episodes = np.arange(eval_config.num_eval_episodes)
    logging = {}

    # Initialize data to save
    eval_actions = np.zeros((eval_config.num_eval_episodes, eval_config.eval_environment.steps_per_episode))
    eval_actions[:] = np.nan

    eval_regrets = np.zeros((eval_config.num_eval_episodes, eval_config.eval_environment.steps_per_episode))

    # Setup agent for evaluation
    agent.eval()

    for ep in eval_episodes:
        step = 0
        done = False
        lstm_state = agent.get_initial_rnn_state()
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
                'episode': ep,
                'step': step,
                'action': action,
                'reward': reward,
                'arm1_prob': arm1_prob,
                'arm2_prob': arm2_prob,
                'pi_out': pi_out,
                'v_out': v_out,
                'observation': observation,
                'h_t': new_lstm_state[-1, 0, :].detach().cpu().numpy(),
            }

            # Calculate step regret
            chosen_arm_expected_reward = arm1_prob if action == 0 else arm2_prob
            step_regret = optimal_expected_reward - chosen_arm_expected_reward
            eval_regrets[ep, step] = step_regret

            observation = next_observation
            lstm_state = new_lstm_state
            step += 1

    results = {}
    results['logging_df'] = pd.DataFrame(logging).T
    results['logging_df']['step_index'] = results['logging_df']["episode"]*eval_config.eval_environment.steps_per_episode + results['logging_df']["step"]

    if eval_reg_formula == "penalize_1":
        eval_regrets[np.where(eval_regrets > 0)] = 1
    
    # Compute metrics: cumulative regret R_T(b), R_T(b) mean and std_dev
    cumulative_regrets = np.cumsum(eval_regrets, axis=1)
    mean_cumulative_regret = np.mean(cumulative_regrets, axis=0)
    std_cumulative_regret = np.std(cumulative_regrets, axis=0)
    se_cumulative_regret = std_cumulative_regret / np.sqrt(eval_config.num_eval_episodes)

    results['eval_actions'] = eval_actions
    results['eval_regrets'] = eval_regrets
    results['cumulative_regrets'] = cumulative_regrets
    results['mean_cumulative_regret'] = mean_cumulative_regret
    results['std_cumulative_regret'] = std_cumulative_regret
    results['se_cumulative_regret'] = se_cumulative_regret

    print("Done evaluation.")
    return results

# Now with all points
def plot_all_points_with_prob(
    xy, action, reward, prob_left, prob_right, marker_probs, gamma=0.3, return_legend=False, alpha=0.9
):
    """
    marker_probs: array-like of 100 values (or len(xy)) between 0 and 1.
                  0 -> Blue, 0.5 -> White, 1 -> Red.
    """
    # Set global publication-style parameters
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "text.usetex": False,
            "axes.unicode_minus": False,
        }
    )

    # Decide if it is a left or right trial (depends on which arm has higher probability)
    # Note: Trajectory line still uses this color scheme for time progression
    trial_type = "blue" if prob_left > prob_right else "red"
    line_cmap = plt.cm.Blues_r if trial_type == "blue" else plt.cm.Reds_r

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # 1. Background elements: Vertical reference line at x = 0
    ax.axvline(0, color="#999999", linestyle="--", linewidth=2.5, zorder=1)

    # -----------------------------------------------------------------------


    # Define Marker Colormap (0=Blue, 0.5=White, 1=Red)
    marker_cmap = plt.cm.bwr
    marker_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # 2. Sequential Marker Drawing with Color & Style Encoding
    MARKER_SIZE = 400
    BLUE_EDGELINE = 1
    RED_EDGELINE = 1
    num_points = len(xy)

    for i in range(num_points):
        # Shape determines Action
        if action[i] == 0:  # Left action
            edge_color = "blue"
            lw = BLUE_EDGELINE
        else:  # Right action
            edge_color = "red"
            lw = RED_EDGELINE

        # Edge Style determines Outcome
        if reward[i] > 0:
            marker_shape = "*"
            current_s = MARKER_SIZE * 1.3
        else:
            marker_shape = "^"
            current_s = MARKER_SIZE

        # New Color Definition based on passed probabilities
        marker_color = marker_cmap(marker_norm(marker_probs[i]))

        ax.scatter(
            xy[i, 0],
            xy[i, 1],
            marker=marker_shape,
            color=marker_color,
            s=current_s,
            edgecolor=edge_color,
            linewidth=lw,
            zorder=3 + num_points - i,
            alpha=alpha,
        )

    # Dummy scatter updated to represent the Marker Probabilities Colorbar
    dummy_sc = ax.scatter(xy[:, 0], xy[:, 1], c=marker_probs, cmap=marker_cmap, norm=marker_norm, s=0)

    # 3. Custom Axis Styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)

    ax.tick_params(direction="in", width=2.5, length=7, labelsize=18, pad=8)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-3, -2, -1, 0, 1, 2])

    ax.set_xlim(min(xy[:, 0].min() - 0.7, -0.5), max(0.8, xy[:, 0].max() + 0.7))
    ax.set_ylim(xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7)

    ax.set_xlabel("PCA 1", fontsize=24, labelpad=15, fontweight="normal")
    ax.set_ylabel("PCA 2", fontsize=24, labelpad=15, fontweight="normal")

    # 6. Bottom Horizontal Colorbar (Now reflects the 0 - 0.5 - 1 marker probability)
    cadd_ax = fig.add_axes([0.05, -0.03, 0.35, 0.03])
    cbar = fig.colorbar(dummy_sc, cax=cadd_ax, orientation="horizontal")
    cbar.outline.set_linewidth(1.5)

    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"], fontsize=20)
    cbar.ax.tick_params(length=4, direction='in')

    plt.subplots_adjust(bottom=0.22, left=0.12, right=0.95, top=0.88)

    return ax

OUT_DIR = pathlib.Path(__file__).parents[2] / "output"
seed = 45
reward_structure = "dependent_e"
agent, training_config = load_agent(OUT_DIR / "bandit_models_rnn" / f"{reward_structure}_models" / f"{reward_structure}_{seed}.pkl")

# 1. Prepare eval environment and config, and evaluate agents
eval_config = load_eval_config(training_config = training_config)
eval_env = create_env(eval_config.eval_environment)
observation = eval_env.reset() #[0,0,0]
np.random.seed(42)

result = evaluate_agent(agent,eval_env, eval_config, verbose=True)

#%% 2. PCA and plot trajectory for a specific episode
logging_df = result['logging_df'].set_index("step_index")

all_h_t = np.stack(logging_df['h_t'].values)  # steps, hidden_dim
all_h_t_per_episode = all_h_t.reshape(eval_config.num_eval_episodes, eval_config.eval_environment.steps_per_episode, -1)

pca = PCA(n_components=2)
pca.fit(all_h_t)

traj_2d = [pca.transform(traj)[:, [0, 1]] for traj in all_h_t_per_episode]

ep = 45
prob_left = logging_df.loc[logging_df['episode'] == ep, 'arm1_prob'].values[0]
prob_right = logging_df.loc[logging_df['episode'] == ep, 'arm2_prob'].values[0]

xy = traj_2d[ep]
t = np.arange(len(xy))  # time steps
action = logging_df.loc[logging_df['episode'] == ep, 'action'].values
reward = logging_df.loc[logging_df['episode'] == ep, 'reward'].values

# Plot trajectory for 1 episode
ax = plot_decision_trajectory(
    xy=xy, 
    t=t, 
    action=action, 
    reward=reward, 
    prob_left=prob_left, 
    prob_right=prob_right,
    gamma=0.6,
    #cmap=plt.cm.YlOrBr_r  # 'turbo_r' perfectly matches the Red -> Blue progression
)
ax.tick_params(labelsize=40)
ax.set_xlabel("PC1", fontsize=45, labelpad=10, fontweight="normal")
ax.set_ylabel("PC2", fontsize=45, labelpad=10, fontweight="normal")
ax.legend().remove()

# Save as SVG with transparent background for publication-quality figure
# plt.savefig(OUT_DIR / "chapter_1_plots" / "plots" / f"pca_trajectory_ep{ep}_{reward_structure}_{seed}.svg", bbox_inches='tight', transparent=True, format="svg")

plt.show()

#%% 3. PCA and plot ALL points
logging_df["pi_out_right_prob"] = np.array([torch.softmax(x, dim=0)[1] for x in logging_df['pi_out'].values])

all_h_t = np.stack(logging_df['h_t'].values)  # steps, hidden_dim
all_h_t_per_episode = all_h_t.reshape(eval_config.num_eval_episodes, eval_config.eval_environment.steps_per_episode, -1)

pca = PCA(n_components=2)
pca.fit(all_h_t)

xy = pca.transform(all_h_t)
action = logging_df['action'].values
reward = logging_df['episode'].values
pi_out_right = logging_df['pi_out_right_prob'].values

n_points_to_plot = 10000  # Adjust this number as needed
ax = plot_all_points_with_prob(
    xy=xy[:n_points_to_plot],
    action=action[: n_points_to_plot], 
    reward=reward[:n_points_to_plot], 
    prob_left=prob_left, 
    prob_right=prob_right,
    marker_probs=pi_out_right[:n_points_to_plot],
    alpha=0.9,
)
ax.tick_params(labelsize=40)
ax.set_xlabel("PC1", fontsize=45, labelpad=10, fontweight="normal")
ax.set_ylabel("PC2", fontsize=45, labelpad=10, fontweight="normal")

# Save as SVG with transparent background for publication-quality figure
# plt.savefig(OUT_DIR / "chapter_1_plots" / "plots" / f"pca_trajectory_ep{ep}_{reward_structure}_{seed}.svg", bbox_inches='tight', transparent=True, format="svg")

plt.show()