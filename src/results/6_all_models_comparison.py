#%%
import sys
import pathlib
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from neural_networks import agents_2_LSTM_torch, agents_3_RNN, agents_4_lowrank_RNN, agents_5_GRU

# Map the old module path that pickle expects to the new module
sys.modules['bandits.agents_2'] = agents_2_LSTM_torch
sys.modules['bandits.agents_3_RNN'] = agents_3_RNN
sys.modules['bandits.agents_4_lowrank_RNN'] = agents_4_lowrank_RNN
sys.modules['bandits.agents_5_GRU'] = agents_5_GRU

#%%
OUT_DIR = pathlib.Path(__file__).parents[2] / "output"

environments = [
    "dependent_e_models", 
    "dependent_m_models", 
    "dependent_u_models",
    "independent_models", 
    "dependent_h_models"
]
seeds =  [44, 45, 46, 47, 48] + [50, 51, 52, 53, 54]

line_value = {
    "dependent_e_models": 0.9,
    "dependent_m_models": 0.75,
    "dependent_u_models": 0.75,
    "dependent_h_models": 0.6,
    "independent_models": 0.66
}

model_name_title = {
    "dependent_e_models": "Dep. Easy",
    "dependent_m_models": "Dep. Medium",
    "dependent_u_models": "Dep. Uniform",
    "independent_models": "Independent",
    "dependent_h_models": "Dep. Hard",
}

agent_network = {
    "torch": "LSTM",
    "gru": "GRU",
    "rnn": f"RNN\n(48 hidden)",
    "low_rank_1_256_hidden_states": f"1-Rank RNN\n(256 hidden)",
    "low_rank_2_256_hidden_states": f"2-Rank RNN\n(256 hidden)",
    "low_rank_3_256_hidden_states": f"3-Rank RNN\n(256 hidden)",
    "low_rank_1":f"1-Rank RNN\n(48 hidden)",
    "low_rank_2": f"2-Rank RNN\n(48 hidden)",
    "low_rank_16": f"16-Rank RNN\n(48 hidden)",
}

agents = ["torch", "gru", "rnn", "low_rank_1_256_hidden_states", "low_rank_2_256_hidden_states", "low_rank_3_256_hidden_states", "low_rank_1", "low_rank_2", "low_rank_16"]

# BUILD plot_df with all last rewards for all agent, seeds, environments
mean_reward_all_df = pd.DataFrame(columns=["Last_Reward", "Seed", "Agent", "Environment"])
for agent in agents:
    # Iterate through each environment
    for i, model_name in enumerate(environments):
        MODEL_DIR = OUT_DIR / f"bandit_models_{agent}" / model_name
        
        all_models_plot_df = {}
        for seed in seeds:
            model_params_name = f'{MODEL_DIR.name.rsplit("_", 1)[0]}_{seed}.pkl'
            try:
                with open(MODEL_DIR / model_params_name, 'rb') as fp:
                    all_models_plot_df[f'{MODEL_DIR.name}_{seed}'] = pickle.load(fp)[0]["training_df"]
                    # Optional: limit the xaxis
                    all_models_plot_df[f'{MODEL_DIR.name}_{seed}'] = all_models_plot_df[f'{MODEL_DIR.name}_{seed}'][all_models_plot_df[f'{MODEL_DIR.name}_{seed}']["Global_step"] < 2e6]
            except FileNotFoundError:
                continue

            # HERE we construct the plot_df adding one row at a time
            last_reward = all_models_plot_df[f'{MODEL_DIR.name}_{seed}']["Mean_Reward"].iloc[-50:-1].mean()
            mean_reward_all_df.loc[len(mean_reward_all_df)] = {
                "Last_Reward": last_reward,
                "Seed": seed,
                "Agent": agent,
                "Environment": model_name
            }

        if not all_models_plot_df:
            continue

dependent_e_df = mean_reward_all_df[mean_reward_all_df["Environment"]=="dependent_e_models"]
dependent_m_df = mean_reward_all_df[mean_reward_all_df["Environment"]=="dependent_m_models"]

#%% Plot
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Pretty names
# -----------------------------

plot_df = dependent_e_df.copy()
plot_df["Agent"] = plot_df["Agent"].map(agent_network)

order = [
    "LSTM",
    "GRU",
    f"RNN\n(48 hidden)",
    f"1-Rank RNN\n(256 hidden)",
    f"2-Rank RNN\n(256 hidden)",
    f"3-Rank RNN\n(256 hidden)",
    f"1-Rank RNN\n(48 hidden)",
    f"2-Rank RNN\n(48 hidden)",
    # "Low Rank 16 (48 hidden)",
]

# -----------------------------
# Paper / NeurIPS style
# -----------------------------
sns.set_theme(
    style="ticks",
    context="poster",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "xtick.major.size": 0,
    "ytick.major.size": 4,
})

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Light violin for distribution shape
sns.violinplot(
    data=plot_df,
    x="Agent",
    y="Last_Reward",
    order=order,
    inner="box",
    linewidth=0,
    color="#d9d9d9",
    cut=0,
    ax=ax,
    width=1,
)

# Swarm points
sns.swarmplot(
    data=plot_df,
    x="Agent",
    y="Last_Reward",
    order=order,
    size=7,
    color="green",
    alpha=0.9,
    ax=ax
)

# -----------------------------
# Labels and formatting
# -----------------------------
ax.set_xlabel("Model Architecture", fontsize=35)
ax.set_ylabel("Final Reward", fontsize=35)

ax.set_ylim(0.4, 0.95)

# Tilt x labels like your example
plt.xticks(
    rotation=50,
    ha="right",
    fontsize=25
)

ax.tick_params(axis='y', labelsize=30)

plt.title("Dependent Easy Environment for all seeds", size=34, fontweight="bold")

sns.despine(left=False, bottom=False)
ax.axhline(0.9, linestyle="--", color="k", linewidth=2, alpha=1)
ax.set_yticks([0.5, 0.7, 0.9])

plt.tight_layout()
plt.savefig(OUT_DIR / "chapter_1_plots" / "plots" / f"final_comparison_violins_all_models_dependent_easy.svg", bbox_inches='tight', transparent=True, format="svg")

plt.show()
# %%
