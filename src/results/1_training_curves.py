#%%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import sys
from neural_networks import agents_2_LSTM_torch, agents_3_RNN

# Map the old module path that pickle expects to the new module
sys.modules['bandits.agents_2'] = agents_2_LSTM_torch
sys.modules['bandits.agents_3_RNN'] = agents_3_RNN

OUT_DIR = pathlib.Path(__file__).parents[2] / "output"

environments = [
    "dependent_e_models", 
    "dependent_m_models", 
    "dependent_u_models",
    "independent_models", 
    "dependent_h_models"
]
seeds = [44, 45, 46, 47, 48] + [50, 51, 52, 53, 54]

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
    "torch_retrained": "LSTM",
    "torch": "LSTM",
    "gru": "GRU",
    "rnn": "RNN",
    "low_rank_1_256_hidden_states": "Low Rank 1",
    "low_rank_2_256_hidden_states": "Low Rank 2",
    "low_rank_3_256_hidden_states": "Low Rank 3",
    "low_rank_1": "Low Rank 1 (48 hidden states)",
    "low_rank_2": "Low Rank 2 (48 hidden states)",
    "low_rank_16": "Low Rank 16 (48 hidden states)",
}

agents = ["torch", "rnn"] # ["torch", "gru", "rnn", "low_rank_1_256_hidden_states", "low_rank_2_256_hidden_states", "low_rank_3_256_hidden_states", "low_rank_1", "low_rank_2", "low_rank_16"]
for agent in agents:
    # Plotting parameters
    x_col = "Global_step"
    y_col = "Mean_Reward"
    window_size =100
    y_start = 0.5

    # Calculate global Y limits so all plots share the exact same vertical scale
    global_y_max = max(line_value.values())
    global_y_min = y_start

    # 1. Setup the Figure Grid (1 row, 5 columns)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), dpi=500)

    # 2. Iterate through each environment and its corresponding axis
    for i, (model_name, ax) in enumerate(zip(environments, axes)):
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

        if not all_models_plot_df:
            continue 

        x_max = max([df[x_col].max() for df in all_models_plot_df.values()])
        y_target = line_value[model_name]

        # Initialize a list to hold the dataframes of only the successful (black) models
        black_lines_data = []

        for key, plot_df in all_models_plot_df.items():
            plot_df = plot_df.dropna(subset=[x_col, y_col]).sort_values(by=x_col).iloc[1:].copy()
            plot_df["Rolling_Mean"] = plot_df[y_col].rolling(window=window_size, min_periods=1).mean()

            # Condition for color based on how good is the model at the end of training
            last_value_mean_reward = plot_df["Rolling_Mean"].iloc[-1]
            if model_name == "dependent_e_models":
                color = "black" if last_value_mean_reward > y_target - 0.1 else "#A0522D"
            elif model_name == "dependent_m_models":
                color = "black" if last_value_mean_reward > y_target - 0.1 else "#A0522D"
            elif model_name == "independent_models":
                color = "black" if last_value_mean_reward > y_target - 0.1 else "#A0522D"
            elif model_name == "dependent_u_models":
                color = "black" if last_value_mean_reward > y_target - 0.15 else "#A0522D"
            elif model_name == "dependent_h_models":
                color = "#A0522D"

            # If the line is black, save its data so we can average it later
            if color == "black":
                black_lines_data.append(plot_df[[x_col, "Rolling_Mean"]])
                color="grey"

            ax.plot(
                plot_df[x_col],
                plot_df["Rolling_Mean"],
                color=color,
                linewidth=2,
                zorder=3,
                alpha=0.65
            )

        # --- Calculate and plot the mean line for the black lines ---
        if black_lines_data:
            # Combine all the black dataframes into one
            combined_black_df = pd.concat(black_lines_data)
            # Group by the x-axis step and calculate the mean of the Rolling_Mean
            mean_black_df = combined_black_df.groupby(x_col)["Rolling_Mean"].mean().reset_index()
            
            # Plot the mean line
            ax.plot(
                mean_black_df[x_col],
                mean_black_df["Rolling_Mean"],
                color="k",
                linestyle="-",
                linewidth=3, # Made slightly thicker so it stands out
                zorder=4,    # Placed higher than the individual lines (zorder=3)
                alpha=1.0
            )
        # -----------------------------------------------------------------

        ax.axhline(y=line_value[model_name], color="black", linestyle="--", linewidth=1.5, zorder=2, alpha=0.8)

        # 3. Styling the specific Axis
        ax.set_title(model_name_title[model_name], fontsize=24, fontweight='bold')
        
        if i == 0:
            ax.set_ylabel("Mean Reward", fontsize=30)
        else:
            ax.set_ylabel("")

        ax.set_yticks([y_start, line_value[model_name]])
        ax.set_yticklabels([f'{y_start:.2f}', f'{y_target:.2f}'])

        ax.set_ylim(y_start - 0.02, global_y_max + 0.02)
        
        ax.set_xlim(0, x_max)
        
        xticks = [0, x_max]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{tick/1e6:.1f}' for tick in xticks])
        
        ax.tick_params(axis='both', which='major', labelsize=22)

    # Using trim=False so the y-axis extends to the very top/bottom limits
    sns.despine(fig=fig, offset=5, trim=False)

    # Put xlabel closer
    fig.supxlabel("Step (millions)", fontsize=30, y=0.07) 
    plt.suptitle(f"Training Curves for {agent_network[agent]} Agents", fontsize=32, y=0.98, fontweight='bold')

    plt.tight_layout()

    # Save as PDF for vector graphics in LaTeX
    # plt.savefig(OUT_DIR / "bandit_models_rnn" / "training_curves_with_mean.svg", format="svg", bbox_inches='tight', transparent=True)
    plt.show()
# %%
