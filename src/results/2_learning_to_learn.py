#%%
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import ml_collections
import copy

from utils.logger import create_logger
from utils.evaluation import evaluate_mean_cum_regret
from neural_networks import agents_2_LSTM_torch, agents_3_RNN
from neural_networks.agents_3_RNN import create_agent
from bandits.bandit_environments import create_env

# Map the old module path that pickle expects to the new module
sys.modules['bandits.agents_2'] = agents_2_LSTM_torch
sys.modules['bandits.agents_3_RNN'] = agents_3_RNN

# Preliminary function
def compute_quantiles(df_with_rewards, num_quantiles=7, window_size=None, plot_maximum=False, set_plateau_x=None):
    """Compute quantiles from the beggining to when the mean_reward stops improving"""
    if set_plateau_x is None:
        if window_size is None:
            window_size = int(df_with_rewards.shape[0] / 8)
        df_with_rewards['smoothed'] = df_with_rewards['Mean_Reward'].rolling(window=window_size, center=True).mean()
        max_smoothed = df_with_rewards['smoothed'].max()

        # 2. Calculate the gradient (slope)
        df_with_rewards['gradient'] = np.gradient(df_with_rewards['smoothed'])

        # 3. Define the plateau: where the slope is effectively flat
        # We look for the first index where the gradient is consistently small
        threshold = 1e-5
        plateau_x = df_with_rewards[df_with_rewards['smoothed'] > max_smoothed*0.8][df_with_rewards['gradient'].abs() < threshold]['Global_step'].iloc[0]

    if plot_maximum:
        plt.plot(df_with_rewards["Global_step"],df_with_rewards["smoothed"])
        plt.vlines(plateau_x, ymin=0, ymax=1, color="k")

    quantiles = [q for q in range(0, int(plateau_x)-1, int(plateau_x/num_quantiles))]
    quantiles.append(plateau_x)
    return quantiles

def train_config(seed, reward_structure):
    train_config = ml_collections.ConfigDict()

    train_config.phase = 'train'

    # Saving
    train_config.path = './'
    train_config.params_filename = 'my_trained_agent_independent_arms'

    # Training
    train_config.random_seed = seed
    train_config.num_workers = 1
    train_config.num_evaluators = 1
    train_config.eval_every_steps = 20_000*10
    train_config.num_eval_episodes = 400

    # Environment
    train_config.environment = ml_collections.ConfigDict()
    train_config.environment.env_name = "bandit"
    train_config.environment.steps_per_episode = int(100)
    train_config.environment.reward_structure = reward_structure # "independent", "dependent_{u,e,m,h}"
    train_config.environment.include_prev_action = True
    train_config.environment.include_prev_reward = True
    train_config.environment.num_arms = 2

    # Agent
    train_config.agent = ml_collections.ConfigDict()
    train_config.agent.total_training_steps = int(100*25_000)
    train_config.agent.random_seed = seed
    train_config.agent.num_lstm_units = 48
    train_config.agent.learning_rate_start = 3e-4
    train_config.agent.learning_rate_end = 0.0
    train_config.agent.gamma = 0.9
    train_config.agent.v_loss_coef = 0.05
    train_config.agent.e_loss_coef_start = 0.0
    train_config.agent.e_loss_coef_end = 0.0
    train_config.agent.max_unroll_steps = 200
    train_config.agent.global_norm_grad_clip = 50.0

    train_config.log_every_steps = int(20*100) # every this steps, logs one full episode

    return train_config

# 1. Wrap the training logic in a function
def train_single_agent_saving_quantiles(seed, reward_structure, base_config, out_dir, quantiles_to_save=None):
    # Deepcopy the config so different processes don't accidentally modify the same object
    config = copy.deepcopy(base_config)
    config.environment.reward_structure = reward_structure
    config.random_seed = seed
    config.agent.random_seed = seed

    # Set random seed
    np.random.seed(config.random_seed)

    # Initialize environment
    env = create_env(config.environment)
    observation = env.reset()

    # Initialize agent
    agent = create_agent(observation=observation, num_actions=env.num_actions, agent_config=config.agent)

    # Initialize logger
    logger = create_logger(logger_name='bandit', config=config, log_to_console=True, print_every_steps=10000)

    # Initialize RNN recurrent state
    rnn_state = agent.get_initial_rnn_state()
    step, episode, loss_val = 0, 0, 0.0

    while step < config.agent.total_training_steps:
        action, pi_out, v_out, new_lstm_state, _ = agent.get_action(observation, rnn_state)
        next_observation, reward, done, info = env.step(action)

        agent.buffer.append(
            obs=observation,
            action=action,
            reward=reward,
            next_obs=next_observation,
            done=done,
            rnn_state=rnn_state,
        )

        observation = next_observation
        rnn_state = new_lstm_state

        current_loss, grads, num_steps_updated = agent.update(done, update_params=True)
        if current_loss is not None:
            loss_val = current_loss 

        logger.log_step(
            global_step=step,
            worker_step=step,
            reward=reward,
            info=info,
            loss=loss_val, 
            entropy_coef=agent.e_loss_coef,
            action=action, 
            next_observation=next_observation, 
            arms_proba=env._arm_probs, 
        )

        # If the current step is in the list of quantiles_to_save then save the agent
        if quantiles_to_save is not None:
            if step in quantiles_to_save:
                model_params_name = f'{config.environment.reward_structure}_{config.random_seed}_step_{step}.pkl'
                save_dir = out_dir / 'bandit_models_rnn' / f'{reward_structure}_models'
                save_dir.mkdir(parents=True, exist_ok=True) # Added mkdir to prevent errors
                save_path = save_dir / model_params_name
                
                with open(save_path, 'wb') as fp:
                    pickle.dump(agent, fp)

                print(f"Saved model at step {step} to {save_path}")
        
        step += 1

        if done:
            episode += 1
            done = False
            rnn_state = agent.get_initial_rnn_state()
            observation = env.reset()

    logger.close_logger()

    results = {
        "agent": agent,
        "config": config.to_dict(),
        "training_df": logger.df,
        "granular_df": logger.df_granular,
    }

    # Ensure the save directory exists
    save_dir = out_dir / 'bandit_models_rnn' / f'{reward_structure}_models_another_trial_same_seed'
    save_dir.mkdir(parents=True, exist_ok=True) # Added mkdir to prevent errors
    
    model_params_name = config.environment.reward_structure
    save_path = save_dir / f'{model_params_name}_{config.random_seed}.pkl'
    
    with open(save_path, 'wb') as fp:
        pickle.dump([results], fp)
        
    return f"Seed {seed} finished training."


OUT_DIR = pathlib.Path(__file__).parents[2] / "output"
MODELS_OUTPUT_DIR = OUT_DIR / "training_percentile_models"

# Script Parameters
train_model = False # If true, it will train the model defined saving the agents from quantiles
plot_evaluation_each_step = False
evaluate = True

# Select the seed with the best final performance
model_name = "dependent_e_models" # "independent_models", "dependent_{u,e,m,h}_models"
seeds = [45, 46, 47, 48] + [50, 51, 52, 53, 54]
MODEL_DIR = OUT_DIR / "bandit_models_rnn" / model_name

#%% 1. Selecting model percentiles to use
seed = "seed_max_reward" # seed_number or "seed_max_reward"
reward_structure = model_name.removesuffix("_models")
num_quantiles_to_save = 10 #np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])* 2e6
window_size = {
    "dependent_e_models": None,
    "dependent_m_models": None,
    "dependent_u_models": 12,
    "independent_models": 13,
}

all_models_df = {}
for seed_i in seeds:
    model_params_name = f'{MODEL_DIR.name.rsplit("_", 1)[0]}_{seed_i}.pkl'
    with open(MODEL_DIR / model_params_name, 'rb') as fp:
        all_models_df[f'{model_name}_{seed_i}'] = pickle.load(fp)[0]["training_df"]

last_mean_rewards = [(seed, all_models_df[f"{model_name}_{seed}"]["Mean_Reward"].iloc[-1]) for seed in seeds]
seed_max_reward = max(last_mean_rewards, key=lambda x: x[1])[0]
max_reward = max(last_mean_rewards, key=lambda x: x[1])[1]
seed = seed_max_reward if seed == "seed_max_reward" else seed

print(f"The (seed, mean_reward) for {reward_structure} are: {last_mean_rewards}")
print(f"Seed {seed_max_reward} has the best final performance!")

quantiles_to_save = compute_quantiles(
    all_models_df[f'{model_name}_{seed}'],
    num_quantiles=num_quantiles_to_save,
    plot_maximum=True,
    window_size=window_size[model_name],
)

quantiles_to_save.append(int(1.5e6))
print(f"Quantiles for seed {seed} are {quantiles_to_save}")

#%% 2. Train the best model, saving the agents correspongding to the quantiles of the training steps distribution
# Remember than in order to execute it we must have train_model = True
base_config = train_config(seed=seed_max_reward, reward_structure=reward_structure)

if train_model:
    train_single_agent_saving_quantiles(
        seed=seed,
        reward_structure=reward_structure,
        base_config=base_config,
        out_dir=MODELS_OUTPUT_DIR,
        quantiles_to_save=quantiles_to_save
    )

#%% 3. Evaluate all percentile agents
AGENT_DIR = MODELS_OUTPUT_DIR / 'bandit_models_rnn' / f'{reward_structure}_models'
EVAL_RESULTS_DIR = MODELS_OUTPUT_DIR / 'bandit_models_rnn' / f'{reward_structure}_models' / 'evaluation_results'

quantiles = quantiles_to_save #[200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000]

if evaluate == True:
    data_dict = {}
    data_dict["quantiles"] = quantiles
    data_dict["num_quantiles_to_save"] = len(quantiles) - 1
    for step in quantiles:
        # Load agent
        agent_path = AGENT_DIR / f'{base_config.environment.reward_structure}_{base_config.random_seed}_step_{step}.pkl'
        with open(agent_path, 'rb') as fp:
            agent = pickle.load(fp)
        
        # Evaluate agent
        eval_env = create_env(base_config.environment)
        result = evaluate_mean_cum_regret(
            eval_env = eval_env,
            agent = agent,
            num_episodes = base_config.num_eval_episodes,
            initial_lstm_state = agent.get_initial_rnn_state(),
            verbose=True
        )
        eval_actions_cum_regret, eval_regrets_cum_regret, cumulative_regrets_cum_regret, mean_cumulative_regret, std_cumulative_regret, se_cumulative_regret, logging = result

        # Save in dataframe: rows are each episode and columns are steps
        eval_regrets_cum_regret_df = pd.DataFrame(eval_regrets_cum_regret)

        #Plot
        performance_df = pd.DataFrame(1 - eval_regrets_cum_regret)
        if plot_evaluation_each_step:
            df_long = performance_df.melt(var_name='column', value_name='value')
            sns.lineplot(data=df_long, x='column', y='value', errorbar=('ci', 95))
            plt.title(f"Cumulative Regret Evolution - Step {step}")
            plt.show()

        # Save all data in file and dictionary
        # path = EVAL_RESULTS_DIR / f'performance_df_{base_config.environment.reward_structure}_{base_config.random_seed}_step_{step}.pkl'
        # with open(path, 'wb') as fp:
        #     pickle.dump(performance_df, fp)

        data_dict[step] = performance_df

    path = EVAL_RESULTS_DIR / f'data_dict_performance_{base_config.environment.reward_structure}_{base_config.random_seed}.pkl'
    with open(path, 'wb') as fp:
        pickle.dump(data_dict, fp)

#%% 4. Plot how performance evolves with the training steps
reward_structures = {"dependent_e": 47, "dependent_m": 45}
paths_to_data = [MODELS_OUTPUT_DIR / 'bandit_models_rnn' / f'{reward}_models' / 'evaluation_results' / f'data_dict_performance_{reward}_{reward_structures[reward]}.pkl' for reward in reward_structures]

load_data_dict = True

reward_structure_title = {
    "dependent_e": "Dep. Easy",
    "dependent_m": "Dep. Medium",
    "dependent_u": "Dep. Uniform",
    "independent": "Independent",
    "dependent_h": "Dep. Hard",
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300, sharey=True)

for idx, ax in enumerate(axes):
    # Load data
    if load_data_dict == True:
        with open(paths_to_data[idx], 'rb') as fp:
            data_dict = pickle.load(fp)
    
    num_steps = len(data_dict["quantiles"])-1
    colors = sns.color_palette("magma", n_colors=num_steps) # "viridis", "crest", or "flare" work well

    # Plot lines
    for i, color in zip(range(0, num_steps), colors):
        step = data_dict["quantiles"][i]
        df_step = data_dict[step]
        df_long = df_step.melt(var_name='column', value_name='value')
        #Plot one step (one line)
        sns.lineplot(
            data=df_long,
            x='column',
            y='value',
            # errorbar=('ci', 85),
            errorbar=None,
            ax=ax,
            label=f'{int(step/data_dict["quantiles"][num_steps-1]*100)}th',
            color=color,
            legend=None,
            lw=2,
        )

    # Styling the Axes (The "Image Look")
    ax.set_xlabel("Steps", fontsize=19)
    ax.set_ylabel("Performance (%)", fontsize=19)

    # Despine: Remove the top and right spines
    sns.despine(offset=10, trim=False)

    # Optional: Set specific limits to match the "boxed in" look of the data
    ax.set_ylim(None, 1) # Based on your reference image scale
    ax.set_title(reward_structure_title[list(reward_structures.keys())[idx]], size=25)

    #ax.yaxis.set_tick_params(labelleft=True)
    ax.tick_params(axis='both', which='major', labelsize=16)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1, 1), frameon=False, title="Training\nPercentiles", fontsize=13, title_fontsize=15)

plt.savefig(OUT_DIR / "plots" / f"training_curves_over_percentiles.pdf", bbox_inches='tight')
plt.show()
# %%
