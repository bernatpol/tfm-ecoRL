"""
This script works with a model already trained. It evaluates the model on a set of evaluation episodes and then performs PCA on the hidden states of the RNN to visualize the decision trajectory in a 2D space.
"""
#%% Import libraries
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import pathlib
import pickle
import pandas as pd
import torch

from bandits.bandit_environments import create_env
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import seaborn as sns

def plot_pca_variance(pca, ax=None):
    """
    Plots individual and cumulative explained variance from a fitted PCA object.
    
    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        A fitted PCA object.
    ax : matplotlib.axes.Axes, optional
        An existing axis to plot on. If None, a new figure and axis are created.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis object containing the plot.
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    # Extract variance ratios and convert to percentages
    ind_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(ind_var)
    num_components = len(ind_var)
    x_ticks = np.arange(1, num_components + 1)
    
    # 1. Plot the cumulative variance as grey bars
    bars = ax.bar(x_ticks, cum_var, color='#A9A9A9', alpha=0.9, width=0.6, label='% Cumulative')
    
    # 2. Plot the individual variance as a black line with markers
    line = ax.plot(x_ticks, ind_var, color='black', marker='o', markersize=6, 
                   linewidth=1.5, label='% Individual')
    
    # 3. Add text labels above the individual data points / bars
    for x, ind, cum in zip(x_ticks, ind_var, cum_var):
        ax.text(x, cum + 1.5, f"{cum:.2f}", ha='center', va='bottom', fontsize=9)
        ax.text(x, ind + 1.5, f"{ind:.2f}", ha='center', va='bottom', fontsize=9)
        
    # Formatting Styling
    ax.set_title("Proportion of variance", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Principal Component", fontsize=11, fontweight='bold', labelpad=10)
    
    # X-axis configuration
    ax.set_xticks(x_ticks)
    ax.set_xlim(0.3, num_components + 0.7)
    
    # Y-axis configuration (ticks every 50 with minor ticks)
    ax.set_yticks([0, 50, 100])
    ax.set_ylim(0, 115) # Leave room for top labels
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', length=4)
    ax.tick_params(axis='both', labelsize=11)

    # Set horizontal line at 100
    ax.axhline(100, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    
    # Customize spine visibility
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Legend setup
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), frameon=False, fontsize=11)
    
    return ax


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

def plot_decision_trajectory_with_prob(
    xy, t, action, reward, prob_left, prob_right, marker_probs, gamma=0.3, return_legend=False,
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

    fig, ax = plt.subplots(figsize=(7, 7), dpi=500)

    # 1. Background elements: Vertical reference line at x = 0
    ax.axvline(0, color="#999999", linestyle="--", linewidth=2.5, zorder=1)

    # --- Create a Warped Colormap for the Trajectory Line ---
    base_norm = mcolors.PowerNorm(gamma=gamma, vmin=t.min(), vmax=t.max())
    cmap_samples = np.linspace(t.min(), t.max(), 256)
    warped_colors = line_cmap(base_norm(cmap_samples))
    warped_cmap = mcolors.ListedColormap(warped_colors)
    line_norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
    # -----------------------------------------------------------------------

    # 1. Generate a dense, smooth set of points along the time dimension
    t_smooth = np.linspace(t.min(), t.max(), 500)

    # Interpolate X and Y coordinates independently relative to time
    spline_x = make_interp_spline(t, xy[:, 0], k=3)
    spline_y = make_interp_spline(t, xy[:, 1], k=3)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)

    # 2. Reshape smooth points into consecutive line segments for the gradient
    points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 3. Create and add the colored LineCollection (behind markers via zorder=2)
    lc = LineCollection(
        segments, cmap=warped_cmap, norm=line_norm, linewidth=3, alpha=0.4, zorder=2
    )
    lc.set_array(t_smooth)
    ax.add_collection(lc)

    # 4. Add an arrowhead matching the color of the final step of the trajectory
    final_line_color = warped_cmap(line_norm(t.max()))
    ax.annotate(
        "",
        xy=(x_smooth[-1], y_smooth[-1]),
        xytext=(x_smooth[-5], y_smooth[-5]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=final_line_color,
            facecolor=final_line_color,
            patchB=None,
            shrinkB=0,
            lw=0,
            mutation_scale=22,
        ),
        zorder=2,
    )

    # Define Marker Colormap (0=Blue, 0.5=White, 1=Red)
    marker_cmap = plt.cm.bwr
    marker_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # 2. Sequential Marker Drawing with Color & Style Encoding
    MARKER_SIZE = 1500
    BLUE_EDGELINE = 1.5
    RED_EDGELINE = 1.5
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
            alpha=0.9,
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

    # 4. Top Probability Titles
    # Keeping global dynamic choice based on trial type
    title_color = "blue" if prob_left > prob_right else "red"
    pL_text = f"$p_L$ = {prob_left:.2f}"
    pR_text = f"$p_R$ = {prob_right:.2f}"
    ax.text(0.30, 1.07, pL_text, transform=ax.transAxes, ha="center", va="bottom", fontsize=30, c=title_color)
    ax.text(0.70, 1.07, pR_text, transform=ax.transAxes, ha="center", va="bottom", fontsize=30, c=title_color)

    # 5. Legend
    actions_handle = Line2D([0], [0], color="w", label=r"$\bf{Actions}$")
    outcome_handle = Line2D([0], [0], color="w", label=r"$\bf{Outcomes}$")

    legend_elements = [
        outcome_handle,
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#333333", markeredgecolor="black", linewidth=0, markersize=14, label="  Rewarded"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#333333", markeredgecolor="black", linewidth=0, markersize=13, label="  Not rewarded"),
        actions_handle,
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#333333", markeredgecolor="blue", linewidth=0, markersize=10, markeredgewidth=BLUE_EDGELINE, label="  Left action"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#333333", markeredgecolor="red", linewidth=0, markersize=10, markeredgewidth=RED_EDGELINE, label="  Right action")
    ]

    leg = ax.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(0.98, 0.02), frameon=False, fontsize=16, handlelength=0.7, handletextpad=0.1)

    if return_legend:
        return legend_elements

    # 6. Bottom Horizontal Colorbar (Now reflects the 0 - 0.5 - 1 marker probability)
    cadd_ax = fig.add_axes([0.05, -0.03, 0.35, 0.03])
    cbar = fig.colorbar(dummy_sc, cax=cadd_ax, orientation="horizontal")
    cbar.outline.set_linewidth(1.5)

    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"], fontsize=20)
    cbar.ax.tick_params(length=4, direction='in')

    plt.subplots_adjust(bottom=0.22, left=0.12, right=0.95, top=0.88)

    return ax

#%% MAIN
OUT_DIR = pathlib.Path("C:\\Users\\pol12\\Desktop\\Universitat\\MAMME\\TFM\\tfm-ecoRL\\output")
seed = 45 #Seed with all plots in presentation: 45
reward_structure = "dependent_e"
agent, training_config = load_agent(OUT_DIR / "bandit_models_low_rank_1" / f"{reward_structure}_models" / f"{reward_structure}_{seed}.pkl")

# Prepare eval environment and config
eval_config = load_eval_config(training_config = training_config)
eval_env = create_env(eval_config.eval_environment)
observation = eval_env.reset() #[0,0,0]
np.random.seed(42)

result = evaluate_agent(agent,eval_env, eval_config, verbose=True)

#%% First PCA plot of the trajectory
logging_df = result['logging_df'].set_index("step_index")
logging_df["pi_out_right_prob"] = np.array([torch.softmax(x, dim=0)[1] for x in logging_df['pi_out'].values])
all_h_t = np.stack(logging_df['h_t'].values)  # steps, hidden_dim
all_h_t_per_episode = all_h_t.reshape(eval_config.num_eval_episodes, eval_config.eval_environment.steps_per_episode, -1)

pca = PCA(n_components=2)
pca.fit(all_h_t)
traj_2d = [pca.transform(traj)[:, [0, 1]] for traj in all_h_t_per_episode]
print(f"Explained variance %: {pca.explained_variance_ratio_*100}")
print(f"Cumulative Explained variance %: {np.cumsum(pca.explained_variance_ratio_)*100}")
plot_pca_variance(pca)

#%%
ep = 37 # Choose an episode to visualize
prob_left = logging_df.loc[logging_df['episode'] == ep, 'arm1_prob'].values[0]
prob_right = logging_df.loc[logging_df['episode'] == ep, 'arm2_prob'].values[0]

xy = traj_2d[ep]
t = np.arange(len(xy))  # time steps
action = logging_df.loc[logging_df['episode'] == ep, 'action'].values
reward = logging_df.loc[logging_df['episode'] == ep, 'reward'].values
pi_out_right = logging_df.loc[logging_df['episode'] == ep, 'pi_out_right_prob'].values

# PLOT PC1 vs PC2 trajectory
ax = plot_decision_trajectory_with_prob(
    xy=xy, 
    t=t, 
    action=action, 
    reward=reward, 
    prob_left=prob_left, 
    prob_right=prob_right,
    marker_probs=pi_out_right,
)

# Increase xtick, ytick label
ax.tick_params(labelsize=40)
ax.set_xlabel("PC1", fontsize=45, labelpad=10, fontweight="normal")
ax.set_ylabel("PC2", fontsize=45, labelpad=10, fontweight="normal")

ax.legend().remove()

# Save as SVG with transparent background for publication-quality figure
# plt.savefig(OUT_DIR / "training_percentile_models" / "plots" / f"pca_trajectory_ep{ep}_{reward_structure}_{seed}.svg", bbox_inches='tight', transparent=True, format="svg")

plt.show()

#%% plot all points
xy_all = pca.transform(all_h_t)
action_all = logging_df['action'].values
reward_all = logging_df['episode'].values
prob_left_all = logging_df['arm1_prob'].values
prob_right_all = logging_df['arm2_prob'].values
pi_out_right_all = logging_df['pi_out_right_prob'].values
n_points_to_plot = 1000  # Adjust this number as needed
ax = plot_all_points_with_prob(
    xy=xy_all[:n_points_to_plot],
    action=action_all[: n_points_to_plot], 
    reward=reward_all[:n_points_to_plot], 
    prob_left=prob_left, 
    prob_right=prob_right,
    marker_probs=pi_out_right_all[:n_points_to_plot],
    alpha=0.9,
)

# %% Build Histograms of probabilities of each PC1
# 1. Compute PC1 for all steps across all episodes
# Since all_h_t is ordered identically to logging_df, the transformed rows match exactly
logging_df['PC1'] = pca.transform(all_h_t)[:, 0]
logging_df['PC2'] = pca.transform(all_h_t)[:, 1]

# 2. Filter PC1 values based on the 'pi_out_right_prob' threshold
pc1_high_prob = logging_df.loc[logging_df['pi_out_right_prob'] > 0.5, 'PC1']
pc1_low_prob = logging_df.loc[logging_df['pi_out_right_prob'] <= 0.5, 'PC1']

# 3. Create the overlaid histograms using subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the histogram for pi_out_right_prob > 0.5
ax.hist(
    pc1_high_prob, 
    bins=40, 
    alpha=0.6, 
    label=r'$\pi_{\mathrm{out\_right\_prob}} > 0.5$', 
    color='royalblue', 
    edgecolor='black'
)

# Plot the histogram for pi_out_right_prob <= 0.5
ax.hist(
    pc1_low_prob, 
    bins=40, 
    alpha=0.6, 
    label=r'$\pi_{\mathrm{out\_right\_prob}} <= 0.5$', 
    color='darkorange', 
    edgecolor='black'
)

# Customize labels and layout using LaTeX formatting
ax.set_title('Distribution of $PC_1$ Values Across All Episodes')
ax.set_xlabel('$PC_1$ Value')
ax.set_ylabel('Frequency')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

# Save the visualization
plt.tight_layout()
plt.savefig('pc1_distribution_histogram.png', dpi=300)
# %%
sns.scatterplot(
    data=logging_df, 
    x='PC1',
    y='pi_out_right_prob', 
    hue='pi_out_right_prob', 
    palette='coolwarm', 
    alpha=0.7
)
plt.show()
sns.scatterplot(
    data=logging_df, 
    x='PC2',
    y='pi_out_right_prob', 
    hue='pi_out_right_prob', 
    palette='coolwarm', 
    alpha=0.7
)
# %% NOW we compute the diff. equation trajectories for [RIGHT, REWARDED]
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class BanditRNNAnalyzer:
    def __init__(self, rnn_module):
        """
        Extracts weights from a PyTorch nn.RNN layer.
        """
        # PyTorch nn.RNN stores weights as attributes
        self.W_ih = rnn_module.W_ih.weight.detach() # Shape: [N, 3]
        self.b_ih = rnn_module.W_ih.bias.detach()   # Shape: [N]
        self.W_hh = rnn_module.U_hh.weight.detach() @ rnn_module.V_hh.weight.detach() # Shape: [N, N]
        self.b_hh = rnn_module.U_hh.bias.detach()
        self.b_total = self.b_ih + self.b_hh

    def compute_velocity(self, h, u):
        """
        Computes the discrete "velocity" step: f(h) = h_next - h
        Supports batched inputs.
        h: [Batch, 48]
        u: [3]
        """
        # u @ W_ih.T + b
        input_term = torch.matmul(u, self.W_ih.t()) + self.b_total
        # h @ W_hh.T
        recurrent_term = torch.matmul(h, self.W_hh.t())
        
        h_next = torch.tanh(recurrent_term + input_term)
        return h_next - h

    def find_fixed_points(self, u, ICs, lr=0.01, max_steps=5000, q_threshold=1e-5):
        """
        Optimizes batched Initial Conditions to find fixed points for input u.
        ICs: Tensor of shape [Batch, 48] (Your collected trajectory states)
        """
        # Ensure we are optimizing a copy of the ICs
        h_opt = ICs.clone().detach().requires_grad_(True)
        u_tensor = torch.tensor(u, dtype=torch.float32)

        # PyTorch's Adam optimizer is great for batched gradient descent
        optimizer = torch.optim.Adam([h_opt], lr=lr)

        for step in range(max_steps):
            optimizer.zero_grad()
            
            # 1. Compute velocity
            velocity = self.compute_velocity(h_opt, u_tensor)
            
            # 2. Compute q(h) = 1/2 * ||velocity||^2 for each IC
            q_values = 0.5 * torch.sum(velocity**2, dim=1)
            
            # 3. Mean loss for the batch
            loss = torch.mean(q_values)
            
            # 4. Backpropagate and step
            loss.backward()
            optimizer.step()

        # Filter out points that didn't converge to a true fixed point
        final_q = 0.5 * torch.sum(self.compute_velocity(h_opt, u_tensor)**2, dim=1)
        valid_fps = h_opt[final_q < q_threshold].detach().numpy()

        if len(valid_fps) == 0:
            print(f"No fixed points found for input {u} below q-threshold.")
            return np.array([])

        # Cluster identical fixed points (within a small distance tolerance)
        unique_fps = []
        for fp in valid_fps:
            if not unique_fps:
                unique_fps.append(fp)
            else:
                distances = cdist([fp], unique_fps)[0]
                if np.min(distances) > 1e-3: # If it's distinct from known FPs
                    unique_fps.append(fp)
                    
        return np.array(unique_fps)

    def plot_phase_portrait(self, u, trajectories, fps, pca, resolution=30):
        """
        Plots the 2D vector field and fixed points using PCA.
        trajectories: np.array of shape [Total_Steps, 48]
        fps: np.array of shape [Num_FPs, 48]
        pca: fitted PCA object
        """
        u_tensor = torch.tensor(u, dtype=torch.float32)
        
        # Determine plot bounds from trajectories
        traj_2d = pca.transform(trajectories)
        x_min, x_max = traj_2d[:, 0].min(), traj_2d[:, 0].max()
        y_min, y_max = traj_2d[:, 1].min(), traj_2d[:, 1].max()
        margin = 0.2
        
        # 2. Create a 2D meshgrid
        xx, yy = np.meshgrid(np.linspace(x_min - margin, x_max + margin, resolution),
                             np.linspace(y_min - margin, y_max + margin, resolution))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]

        # 3. Project grid back up to 48D to compute vector field
        grid_48d = pca.inverse_transform(grid_2d)
        grid_48d_tensor = torch.tensor(grid_48d, dtype=torch.float32)

        with torch.no_grad():
            velocity_48d = self.compute_velocity(grid_48d_tensor, u_tensor).numpy()

        # 4. Project 48D velocities down to 2D
        # Velocity in 2D is the velocity dotted with the PCA components
        velocity_2d = np.dot(velocity_48d, pca.components_.T)
        u_vel = velocity_2d[:, 0].reshape(xx.shape)
        v_vel = velocity_2d[:, 1].reshape(yy.shape)

        # 5. Plotting 
        plt.figure(figsize=(10, 8))
        
        # Plot the vector field (streamplot looks smoother than quiver)
        speed = np.sqrt(u_vel**2 + v_vel**2)
        plt.streamplot(xx, yy, u_vel, v_vel, color=speed, cmap='viridis', linewidth=1)
        
        # Plot actual trajectories (optional, lightly in background)
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], color='gray', alpha=0.3, linewidth=0.5, label="Actual Trajectory")

        # Plot fixed points
        if len(fps) > 0:
            fps_2d = pca.transform(fps)
            plt.scatter(fps_2d[:, 0], fps_2d[:, 1], color='red', s=100, zorder=5, marker='X', label="Fixed Points")

        plt.title(f"Phase Portrait for Input: {u}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.colorbar(label="Flow Speed")
        plt.show()
    
    def plot_q_landscape(self, u, trajectories, pca, resolution=100, vmin=None, vmax=None, fps=None):
        """
        Plots the q-landscape (log10 q) as a heatmap using PCA.
        trajectories: np.array of shape [Total_Steps, 48] (used to set bounds)
        pca: fitted PCA object
        resolution: int, sets the density of the grid for the heatmap
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        
        u_tensor = torch.tensor(u, dtype=torch.float32)
        
        # 1. Determine plot bounds from trajectories
        traj_2d = pca.transform(trajectories)
        x_min, x_max = traj_2d[:, 0].min(), traj_2d[:, 0].max()
        y_min, y_max = traj_2d[:, 1].min(), traj_2d[:, 1].max()
        margin = 0.2
        
        # 2. Create a dense 2D meshgrid for the heatmap
        xx, yy = np.meshgrid(np.linspace(x_min - margin, x_max + margin, resolution),
                             np.linspace(y_min - margin, y_max + margin, resolution))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]

        # 3. Project the 2D grid back to the 48D hidden state space
        grid_48d = pca.inverse_transform(grid_2d)
        grid_48d_tensor = torch.tensor(grid_48d, dtype=torch.float32)

        # 4. Compute velocity and q for every point on the grid
        with torch.no_grad():
            velocity_48d = self.compute_velocity(grid_48d_tensor, u_tensor)
            
            # q = 1/2 * ||velocity||^2
            q_values = 0.5 * torch.sum(velocity_48d**2, dim=1).numpy()

        # 5. Calculate log10(q)
        # We add a tiny epsilon (1e-18) to prevent taking the log of absolute zero
        log_q = np.log10(q_values + 1e-18)
        log_q_grid = log_q.reshape(xx.shape)

        # 6. Plot the heatmap
        plt.figure(figsize=(10, 8))
        
        # The 'cool' colormap provides the Cyan -> Magenta gradient seen in the reference
        mesh = plt.pcolormesh(xx, yy, log_q_grid, cmap='cool', shading='auto', vmin=vmin, vmax=vmax)
        
        plt.title(f"q-Landscape (log10 q) for Input: {u}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        
        # Add the colorbar with the appropriate label
        cbar = plt.colorbar(mesh)
        cbar.set_label("log10 q")

        # Add fixed points if provided, putting a label in order to include them in the legend
        if fps is not None and len(fps) > 0:
            fps_2d = pca.transform(fps)
            cmap = plt.colormaps['rainbow']
            for i, fp in enumerate(fps_2d):
                current_color = cmap(i / max(1, len(fps_2d) - 1))
                plt.scatter(fp[0], fp[1], color=current_color, s=100, zorder=5, marker='X', label=f"Fixed Point {i+1}")
            plt.legend()

        
        plt.show()
    
    def plot_q_landscape_with_optimization(self, u, trajectories, pca, init_points_2d, 
                                           lr=0.02, max_steps=200, resolution=100):
        """
        Plots the log10(q) heatmap and overlays the optimization trajectories 
        moving from initial 2D PCA points toward fixed points.
        
        init_points_2d: np.array of shape [N, 2] or list of [x, y] coordinates
        lr: learning rate for the optimization path
        max_steps: number of optimization steps to track
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        # Ensure input points are in a 2D numpy array format
        init_points_2d = np.atleast_2d(init_points_2d)
        u_tensor = torch.tensor(u, dtype=torch.float32)

        # --- 1. Compute and Plot Background Heatmap ---
        traj_2d = pca.transform(trajectories)
        x_min, x_max = traj_2d[:, 0].min(), traj_2d[:, 0].max()
        y_min, y_max = traj_2d[:, 1].min(), traj_2d[:, 1].max()
        margin = 0.2
        
        xx, yy = np.meshgrid(np.linspace(x_min - margin, x_max + margin, resolution),
                             np.linspace(y_min - margin, y_max + margin, resolution))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_48d = pca.inverse_transform(grid_2d)
        grid_48d_tensor = torch.tensor(grid_48d, dtype=torch.float32)

        with torch.no_grad():
            velocity_48d = self.compute_velocity(grid_48d_tensor, u_tensor)
            q_values = 0.5 * torch.sum(velocity_48d**2, dim=1).numpy()
        log_q_grid = np.log10(q_values + 1e-18).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        mesh = plt.pcolormesh(xx, yy, log_q_grid, cmap='cool', shading='auto')
        cbar = plt.colorbar(mesh)
        cbar.set_label("log10 q")

        # --- 2. Track Optimization Trajectories in 48D ---
        # Project 2D initial guesses back into the 48D hidden space
        init_points_48d = pca.inverse_transform(init_points_2d)
        h_opt = torch.tensor(init_points_48d, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([h_opt], lr=lr)

        # Optimization loop tracking history
        history = []
        for step in range(max_steps):
            history.append(h_opt.detach().clone().numpy())
            optimizer.zero_grad()
            velocity = self.compute_velocity(h_opt, u_tensor)
            q_vals = 0.5 * torch.sum(velocity**2, dim=1)
            loss = torch.mean(q_vals)
            loss.backward()
            optimizer.step()
        history.append(h_opt.detach().clone().numpy())
        history = np.array(history)  # Shape: [steps, num_starting_points, 48]

        # Grab the final 48D position of that specific middle fixed point
        final_h = h_opt[1].unsqueeze(0) # assuming it's the second seed

        # Compute its actual velocity in the full 48D space
        with torch.no_grad():
            true_vel = self.compute_velocity(final_h, u_tensor)
            true_q = 0.5 * torch.sum(true_vel**2).item()

        print(f"True 48D q-value at the 'fixed point': {true_q}")
        print(f"Log10 of true q: {np.log10(true_q)}")
        print(f"Velocity vector at the 'fixed point': {true_vel.numpy()}")

        # --- 3. Project History Down and Plot Trajectories ---
        num_points = init_points_2d.shape[0]
        for i in range(num_points):
            pt_history_48d = history[:, i, :] 
            pt_history_2d = pca.transform(pt_history_48d)  # Shape: [steps, 2]
            
            # Plot the optimization line
            plt.plot(pt_history_2d[:, 0], pt_history_2d[:, 1], color='red', linewidth=2.5, 
                     zorder=3, label="Optimization Path" if i == 0 else "")
            
            # Add directional arrowheads along the path
            arrow_indices = np.linspace(0, len(pt_history_2d) - 10, 4, dtype=int)
            for idx in arrow_indices:
                x1, y1 = pt_history_2d[idx]
                # Look ahead a few steps to establish a stable direction vector
                x2, y2 = pt_history_2d[idx + 4] 
                
                plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", color='red', lw=2.5, mutation_scale=15),
                             zorder=3)
                             
            # Mark the start of optimization
            plt.scatter(pt_history_2d[0, 0], pt_history_2d[0, 1], color='white', 
                        edgecolor='red', s=40, zorder=4)
            
            # Mark where optimization finished (the found fixed point candidate)
            plt.scatter(pt_history_2d[-1, 0], pt_history_2d[-1, 1], color='green', 
                        marker='X', s=150, zorder=5, label="End Trajectory" if i == 0 else "")

        plt.title(f"q-Landscape with Fixed-Point Optimization Paths (Input: {u})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.xlim(x_min - margin, x_max + margin)
        plt.ylim(y_min - margin, y_max + margin)
        
        # Clean up the legend to avoid duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        plt.show()

    def analyze_fixed_point_stability(self, fp, u, N=48):
        """
        Linearizes the system around a given fixed point to determine its type.
        
        fp: np.array of shape [N] (the fixed point hidden state)
        u: np.array or list of shape [3] (the input vector)
        N: int, the dimensionality of the system
        """
        import torch
        import numpy as np
        
        fp_tensor = torch.tensor(fp, dtype=torch.float32, requires_grad=True)
        u_tensor = torch.tensor(u, dtype=torch.float32)
        
        # 1. Compute velocity for the single fixed point
        # Ensure input is batched [1, N] for compute_velocity, then squeeze back to [N]
        vel = self.compute_velocity(fp_tensor.unsqueeze(0), u_tensor).squeeze(0)
        
        # 2. Compute the NxN Jacobian matrix numerically via autograd
        jacobian_list = []
        for i in range(N):
            grad_outputs = torch.zeros(N)
            grad_outputs[i] = 1.0
            # Retain graph because we are computing gradients N times on the same graph
            grad = torch.autograd.grad(vel, fp_tensor, grad_outputs=grad_outputs, retain_graph=True)[0]
            jacobian_list.append(grad.numpy())
            
        jacobian = np.array(jacobian_list)
        
        # 3. Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(jacobian)
        real_parts = np.real(eigenvalues)
        
        # 4. Classify based on the signs of the real parts of the eigenvalues
        # Since this is a discrete-time system mapping (h_next - h), stability 
        # is determined by whether the real parts are less than 0 or greater than 0.
        num_stable = np.sum(real_parts < -1e-5)
        num_unstable = np.sum(real_parts > 1e-5)
        num_marginal = np.sum(np.isclose(real_parts, 0, atol=1e-5))
        
        print("--- Fixed Point Linearization Analysis ---")
        print(f"Total Dimensions: {len(real_parts)}")
        print(f"Stable directions (Real < 0): {num_stable}")
        print(f"Unstable directions (Real > 0): {num_unstable}")
        print(f"Marginal/Center directions (Real approx 0): {num_marginal}")
        
        # Determine the topological classification
        if num_unstable == 0 and num_marginal == 0:
            fp_type = "Stable Node (Attractor)"
            description = "All trajectories pull directly into this point."
        elif num_stable == 0 and num_marginal == 0:
            fp_type = "Unstable Node (Repeller)"
            description = "All trajectories push away from this point in every direction."
        elif num_stable > 0 and num_unstable > 0:
            fp_type = "Saddle Point"
            description = "Trapped/funneled along some dimensions, but pushed away along others."
        else:
            fp_type = "Marginal/Bifurcation Point"
            description = "Contains zero or purely imaginary eigenvalues; complex boundary dynamics."
        
        print(f"Classification: {fp_type}")
        print(f"Behavior: {description}\n")
        
        return {
            "jacobian": jacobian,
            "eigenvalues": eigenvalues,
            "type": fp_type,
            "num_stable": num_stable,
            "num_unstable": num_unstable
        }


#%% 1. Initialize analyzer
analyzer = BanditRNNAnalyzer(agent.rnn)
filtered_df = logging_df.groupby('episode').head(25)
# Convert your trajectory states to a PyTorch tensor to use as Initial Conditions
trajectories = np.array(filtered_df['h_t'].tolist(), dtype=np.float32) # shape[Total_Steps, 48]
ICs = torch.tensor(trajectories, dtype=torch.float32)
N = ICs.shape[1]

# Define your 4 discrete inputs
inputs = [
    [1.0, 0.0, 0.0], # Left, No Reward
    [1.0, 0.0, 1.0], # Left, Reward
    [0.0, 1.0, 0.0], # Right, No Reward
    [0.0, 1.0, 1.0]  # Right, Reward
]

# Run analysis for each input landscape
for u in inputs:
    print(f"Finding fixed points for input {u}...")
    
    # We sample a random subset of 1000 trajectory points as ICs to save time
    sampled_indices = np.random.choice(len(ICs), size=2500, replace=False)
    batch_ICs = ICs[sampled_indices]
    
    # 2. Find fixed points for this input
    fps = analyzer.find_fixed_points(u, batch_ICs, lr=0.001, max_steps=20000, q_threshold=1e-9)
    print(f"Found {len(fps)} unique fixed points.")
    
    # 3. Plot the landscape
    analyzer.plot_phase_portrait(u, trajectories, fps, pca)

    # 4. Plot the q-landscape
    analyzer.plot_q_landscape(u, trajectories, pca, resolution=100, vmin=-2.5, vmax=1, fps=fps)

    for idx, fp in enumerate(fps):
        print(f"\nAnalyzing Fixed Point #{idx + 1}")
        analyzer.analyze_fixed_point_stability(fp, u=u, N=N)
    

# init_points_traj_2d = np.array([[0.0, 0.0], [-2.0, -2.0], [0.0, -4.0], [-4, 0]])
# analyzer.plot_q_landscape_with_optimization(u, trajectories, pca, init_points_2d=init_points_traj_2d, lr=0.001, max_steps=4000, resolution=200)
# %%



