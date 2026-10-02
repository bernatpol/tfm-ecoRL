import copy
import pathlib
import pickle
import ml_collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from bandits.bandit_environments import create_env
from neural_networks.agents_4_lowrank_RNN import create_agent
from utils.logger import create_logger

def train_config():
    train_config = ml_collections.ConfigDict()

    train_config.phase = 'train'

    # Saving
    train_config.path = './'
    train_config.params_filename = 'my_trained_agent_independent_arms'

    # Training
    train_config.random_seed = 44
    train_config.num_workers = 1
    train_config.num_evaluators = 1
    train_config.eval_every_steps = 40_000*10
    train_config.num_eval_episodes = 400

    # Environment
    train_config.environment = ml_collections.ConfigDict()
    train_config.environment.env_name = "bandit"
    train_config.environment.steps_per_episode = int(100)
    train_config.environment.reward_structure = "dependent_m" # "independent", "dependent_{u,e,m,h}"
    train_config.environment.include_prev_action = True
    train_config.environment.include_prev_reward = True
    train_config.environment.num_arms = 2

    # Agent
    train_config.agent = ml_collections.ConfigDict()
    train_config.agent.total_training_steps = int(100*60_000)
    train_config.agent.random_seed = 44
    train_config.agent.num_lstm_units = 256
    train_config.agent.rnn_rank = 2
    train_config.agent.learning_rate_start = 3e-4
    train_config.agent.learning_rate_end = 0.0
    train_config.agent.gamma = 0.9
    train_config.agent.v_loss_coef = 0.05
    train_config.agent.e_loss_coef_start = 0.0
    train_config.agent.e_loss_coef_end = 0.0
    train_config.agent.max_unroll_steps = 200
    train_config.agent.global_norm_grad_clip = 50.0

    train_config.log_every_steps = int(40*100) # every this steps, logs one full episode

    return train_config

# 1. Wrap the training logic in a function
def train_single_agent(seed, reward_structure, base_config, out_dir):
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

    # Initialize RNN recurrent state to zeros
    rnn_state = agent.get_initial_rnn_state()

    step, episode, loss_val = 0, 0, 0.0

    while step < config.agent.total_training_steps:
        action, pi_out, v_out, new_rnn_state, _ = agent.get_action(observation, rnn_state)
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
        rnn_state = new_rnn_state

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
    save_dir = out_dir / f'bandit_models_low_rank_{config.agent.rnn_rank}_{config.agent.num_lstm_units}_hidden_states' / f'{reward_structure}_models'
    save_dir.mkdir(parents=True, exist_ok=True) # Added mkdir to prevent errors
    
    model_params_name = config.environment.reward_structure
    save_path = save_dir / f'{model_params_name}_{config.random_seed}.pkl'
    
    with open(save_path, 'wb') as fp:
        pickle.dump([results], fp)
        
    return f"Seed {seed} finished training."

# 2. Execute the function across multiple cores
if __name__ == '__main__':
    OUT_DIR = pathlib.Path(__file__).parents[3] / "output"
    train_config = train_config()
    reward_structures = ["dependent_m"] #["independent", "dependent_u", "dependent_e", "dependent_m", "dependent_h"]
    seeds = [44, 45, 46, 47, 48] + [50, 51, 52, 53, 54]
    
    # We create a pool of workers. max_workers limits how many cores to use at once.
    # If you have an 8-core CPU, max_workers=5 will train all 5 at the same time.
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for reward_structure in reward_structures:
            for seed in seeds:
                # Submit the job to the pool
                future = executor.submit(train_single_agent, seed, reward_structure, train_config, OUT_DIR)
                futures.append(future)
        
        # This will wait for all tasks to finish and print the return messages
        for future in futures:
            print(future.result())
            
    print('All agents done training!')