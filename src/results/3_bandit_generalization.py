#%%
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import ml_collections

from bandits.bandit_environments import create_env
from utils.evaluation import evaluate_mean_cum_regret
from neural_networks import agents_2_LSTM_torch, agents_3_RNN

# Map the old module path that pickle expects to the new module
sys.modules['bandits.agents_2'] = agents_2_LSTM_torch
sys.modules['bandits.agents_3_RNN'] = agents_3_RNN

OUT_DIR = pathlib.Path(__file__).parents[2] / "output"

#%% In order to make the plots in this section we must have the output
# of the cell for the corresponding seed (in the folder "evaluation_results").

def evaluate_config(seed, training_reward_structure, eval_reward_structure = None):
    """Config for evaluation of a given agent trained on seed and reward_structure."""
    eval_config = ml_collections.ConfigDict()

    eval_config.phase = 'eval'

    # Saving
    eval_config.dir = OUT_DIR / "bandit_models_rnn" / f"{training_reward_structure}_models" / "evaluation_results"
    eval_config.path = OUT_DIR / "bandit_models_rnn" / f"{training_reward_structure}_models" / "evaluation_results" / f"eval_{training_reward_structure}_{seed}.pkl" # "independent", "dependent_{u,e,m,h}"
    eval_config.params_filename = eval_config.path.stem

    # Evaluation
    eval_config.num_workers = 0
    eval_config.num_evaluators = 1
    eval_config.random_seed = 42
    eval_config.num_eval_episodes = 400
    eval_config.log_dynamics = True

    # Evaluation environment
    eval_config.eval_environment = ml_collections.ConfigDict()
    eval_config.eval_environment.env_name = "bandit_eval"
    eval_config.eval_environment.steps_per_episode = int(100)
    eval_config.eval_environment.reward_structure = eval_reward_structure # "independent", "dependent_{u,e,m,h}"

    # Agent
    eval_config.agent = ml_collections.ConfigDict()
    eval_config.agent.path = OUT_DIR / "bandit_models_rnn" / f"{training_reward_structure}_models" / f"{training_reward_structure}_{seed}.pkl"

    return eval_config

def evaluate_single_agent(eval_config):
    """Evaluate a single agent already trained (defined in eval_config)
    
    Returns:
        dictionary with one key for environment trained with info
        Example:
        {
            "dependent_e": {
                "eval_actions": pd.DataFrame,
                "eval_regrets": pd.DataFrame,
                "cumulative_regrets": pd.DataFrame,
                "logging": {...}
            }
        }
    """

    # Load agent
    with open(eval_config.agent.path, 'rb') as fp:
        loaded_dict_agent = pickle.load(fp)[0]
        agent = loaded_dict_agent["agent"]
        training_config = loaded_dict_agent["config"]

    print(agent)
    print(f"Trained on {training_config["environment"]["reward_structure"]}")

    # Initialize evaluation environment
    eval_config.eval_environment.reward_structure = "independent"
    eval_env = create_env(eval_config.eval_environment)

    # Setup initial LSTM memory state.
    try:
        initial_lstm_state = agent.get_initial_lstm_state()
    except AttributeError:
        if "rnn" in dir(agent):
            initial_lstm_state = agent.get_initial_rnn_state()
    lstm_state = initial_lstm_state
    agent.eval()

    results_evaluation = {}
    for env_type in ["independent", "dependent_u", "dependent_e", "dependent_m", "dependent_h"]:
        eval_env._reward_structure = env_type
        temp_results = evaluate_mean_cum_regret(eval_env, agent, eval_config.num_eval_episodes, initial_lstm_state=initial_lstm_state, verbose=False)

        # Save results
        results_evaluation[env_type] = {}
        results_evaluation[env_type]["eval_actions"] = temp_results[0]
        results_evaluation[env_type]["eval_regrets"] = temp_results[1]
        results_evaluation[env_type]["cumulative_regrets"] = temp_results[2]
        results_evaluation[env_type]["logging"] = temp_results[6]
        print(f"{env_type} evaluated \n")

    # Ensure the save directory exists
    save_dir = eval_config.dir
    save_dir.mkdir(parents=True, exist_ok=True) # Added mkdir to prevent errors
    
    save_path = eval_config.path
    
    with open(save_path, 'wb') as fp:
        pickle.dump(results_evaluation, fp)
    
    print(f"Seed {seed} finished training.")

    return results_evaluation


#%% 1. Train the following models (seed, training_reward_structure)
train_models = True
if train_models:
    models_to_evaluate = [(45, 'dependent_e'), (45, 'dependent_m')] #[(45, 'dependent_e'), (46, 'dependent_e'), (47, 'dependent_e'), (51, 'dependent_e'), (52, 'dependent_e'), (53, 'dependent_e'), (45, 'dependent_m'), (46, 'dependent_m'), (45, 'dependent_u'), (47, 'dependent_u'), (51, 'dependent_u'), (45, 'independent'), (47, 'independent'), (50, 'independent'), (51, 'independent'), (54, 'independent')]

    evaluation_results = []
    for seed, training_reward_structure in models_to_evaluate:
        print(f"Evaluating seed {seed} with train reward {training_reward_structure}")
        eval_config = evaluate_config(seed=seed, training_reward_structure=training_reward_structure)
        evaluation_results.append(evaluate_single_agent(eval_config=eval_config))

#%% If you want to get the models with the best seed, the following code prints it:
get_models_with_best_seed = True
if get_models_with_best_seed:
    seeds = [45, 46, 47, 48] #+ [50, 51, 52, 53, 54, 55]
    reward_structures = [
        "dependent_e", 
        "dependent_m", 
        "dependent_u",
        "independent",
    ]
    line_value = {
        "dependent_e_models": 0.9,
        "dependent_m_models": 0.75,
        "dependent_u_models": 0.75,
        "dependent_h_models": 0.6,
        "independent_models": 0.66
    }

    all_models_plot_df = {}
    good_seeds = []
    for reward_structure in reward_structures:
        model_name = f"{reward_structure}_models"
        MODEL_DIR = OUT_DIR / "bandit_models_rnn" / model_name
        
        for seed in seeds:
            # Get model corresponding to seed
            model_params_name = f'{MODEL_DIR.name.rsplit("_", 1)[0]}_{seed}.pkl'
            try:
                with open(MODEL_DIR / model_params_name, 'rb') as fp:
                    training_df = pickle.load(fp)[0]["training_df"]
            except FileNotFoundError:
                continue

            window_size = 60    
            training_df = training_df.dropna(subset=["Global_step", "Mean_Reward"]).sort_values(by="Global_step").iloc[1:].copy()
            training_df["Rolling_Mean"] = training_df["Mean_Reward"].rolling(window=window_size, min_periods=1).mean()

            # Get the good seeds
            last_value_mean_reward = training_df["Rolling_Mean"].iloc[-1]
            y_target = line_value[model_name]
            if model_name == "dependent_e_models":
                if last_value_mean_reward > y_target - 0.1:
                    good_seeds.append((seed, reward_structure))
                    print(f"{model_name}: Seed {seed} has last mean reward: {last_value_mean_reward}")
            elif model_name == "dependent_m_models":
                if last_value_mean_reward > y_target - 0.1:
                    good_seeds.append((seed, reward_structure))
                    print(f"{model_name}: Seed {seed} has last mean reward: {last_value_mean_reward}")
            elif model_name == "independent_models":
                if last_value_mean_reward > y_target - 0.1:
                    good_seeds.append((seed, reward_structure))
                    print(f"{model_name}: Seed {seed} has last mean reward: {last_value_mean_reward}")
            elif model_name == "dependent_u_models":
                if last_value_mean_reward > y_target - 0.15:
                    good_seeds.append((seed, reward_structure))
                    print(f"{model_name}: Seed {seed} has last mean reward: {last_value_mean_reward}")

    print(good_seeds)

#%% 2. Load the results
best_agents = [(45, 'dependent_e'), (46, 'dependent_e'), (47, 'dependent_e'), (51, 'dependent_e'), (52, 'dependent_e'), (53, 'dependent_e'), (45, 'dependent_m'), (46, 'dependent_m'), (45, 'dependent_u'), (47, 'dependent_u'), (51, 'dependent_u'), (45, 'independent'), (47, 'independent'), (50, 'independent'), (51, 'independent'), (54, 'independent')]
training_environments = ["dependent_e", "dependent_m", "dependent_u", "independent"] # We do it for all training environments except "dependent_h"

all_evaluation_results = {}
for training_environment in training_environments:
    seeds = [seed for seed, reward_structure in best_agents if reward_structure == training_environment]
    EVAL_DIR = OUT_DIR / "bandit_models_rnn" / f"{training_environment}_models" / "evaluation_results"

    # Load all evaluation results for the seeds of the selected training environment
    all_evaluation_results[training_environment] = {}
    for seed in seeds:
        path_to_eval_results = EVAL_DIR / f"eval_{training_environment}_{seed}.pkl"
        with open(path_to_eval_results, 'rb') as fp:
            all_evaluation_results[training_environment][seed] = pickle.load(fp)
        print(f"Loaded evaluation results for seed {seed} in environment {training_environment}")


#%% 3. Compute matrix to plot
testing_environments = ['dependent_e', 'dependent_m', 'dependent_h', 'dependent_u', 'independent']
training_environments = ["dependent_e", "dependent_m", "dependent_u", "independent"] # We do it for all training environments except "dependent_h"

all_cum_reg_means_df = pd.DataFrame(columns=testing_environments, index=training_environments)
performance_means_df = pd.DataFrame(columns=testing_environments, index=training_environments)
cumulative_performance_means_df = pd.DataFrame(columns=testing_environments, index=training_environments)

for training_environment in training_environments:
    # Initialize to save the results
    final_cum_reg = {test_env: 0 for test_env in testing_environments}
    final_performance = {test_env: 0 for test_env in testing_environments}
    cumulative_performance = {test_env: 0 for test_env in testing_environments}
    num_seeds = len(all_evaluation_results[training_environment])

    for seed in all_evaluation_results[training_environment]:
        for test_env in all_evaluation_results[training_environment][seed]:
            # Take the mean (over all apisodes) of the last step performance for cum regret and performance
            cum_reg_mean = all_evaluation_results[training_environment][seed][test_env]['cumulative_regrets'].mean(axis=0)[-1]
            all_evaluation_results[training_environment][seed][test_env]['eval_performance'] = 1 - all_evaluation_results[training_environment][seed][test_env]['eval_regrets']
            performance_mean = all_evaluation_results[training_environment][seed][test_env]['eval_performance'].mean(axis=0)[-1]
            # Compute cumulative "eval_performance" and take the mean
            cumulative_performance_mean = all_evaluation_results[training_environment][seed][test_env]['eval_performance'].sum(axis=0)[-1]

            # Save the mean of all seeds
            final_cum_reg[test_env] += cum_reg_mean / num_seeds
            final_performance[test_env] += 100 * performance_mean / num_seeds
            cumulative_performance[test_env] += cumulative_performance_mean / num_seeds

    
    # Add row with mean cum regret to final df
    all_cum_reg_means_df.loc[training_environment] = final_cum_reg
    performance_means_df.loc[training_environment] = final_performance
    cumulative_performance_means_df.loc[training_environment] = cumulative_performance

all_cum_reg_means_df

#%%
def plot_heatmap_performance_summary_matrix(plot_df, save_title = None, title=None, title_cbar=None,robust=True, reverse_cmap=False):
    env_types = ["dependent_e", "dependent_m", "dependent_u", "independent", "dependent_h"]
    clean_labels = [
        "Dep. Easy", 
        "Dep. Medium",
        "Dep. Unif.",
        "Independent",
        "Dep. Hard",
    ]
    clean_labels_dict = dict(zip(env_types, clean_labels))

    plot_df.rename(columns=clean_labels_dict, index=clean_labels_dict, inplace=True)

    # Convert plot_df to numeric values (if not already)
    plot_df = plot_df.apply(pd.to_numeric)
    # Plot the performance summary matrix
    fig, ax = plt.subplots(figsize=(3, 2), dpi=500)
    hm = sns.heatmap(plot_df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True, reverse=reverse_cmap), cbar_kws={"shrink": 0.7, "pad": 0.13}, linewidth=.5, robust=robust, ax=ax)

    # Customize colorbar
    cbar = hm.collections[0].colorbar
    cbar.ax.set_title(title_cbar, fontsize=10)

    # Remove ticks
    ax.tick_params(axis='both', which='both', length=0)

    plt.title(title)
    plt.xlabel("Testing Env.", size=15)
    plt.ylabel("Training Env.", size=15)

    if save_title is not None:
        plt.savefig(OUT_DIR / "chapter_1_plots" / "plots" / f"{save_title}.svg", bbox_inches='tight', transparent=True, format="svg")
    plt.show()
    
# with open(OUT_DIR / "evaluation_results" / "provided_evaluation_45.pkl", 'rb') as fp:
#     provided_evaluation_45 = pickle.load(fp)

# provided_evaluation_45_dif, plot_df = plot_performance_summary_matrix(
#     evaluation_results=all_evaluation_results,
#     training_environments = ["dependent_e", "dependent_m", "dependent_u", "independent"],
#     robust=True
# )

plot_heatmap_performance_summary_matrix(all_cum_reg_means_df, save_title="matrix_all_cum_reg", title="Cumulative Regret", title_cbar="Cum.\nRegret", robust=True, reverse_cmap=True)
plot_heatmap_performance_summary_matrix(performance_means_df, save_title="matrix_performance", title="Performance", title_cbar="Perf. (%)", robust=True, reverse_cmap=True)
plot_heatmap_performance_summary_matrix(cumulative_performance_means_df, save_title="matrix_cumulative_performance", title="Cumulative Performance", title_cbar="Cum.\nPerformance", robust=True, reverse_cmap=True)
#%%