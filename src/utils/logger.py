"""Loggers that take data, process it, and periodically log metrics to file."""

from typing import Dict
import time
import ml_collections
from absl import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_logger(logger_name: str, config=None, log_to_console=False, print_every_steps: int = 10000):
    """Create a logger."""
    if logger_name == "bandit":
        logger = BanditLogger(config=config, log_to_console=log_to_console, print_every_steps=print_every_steps)
    else:
        raise ValueError(f'Logger {logger_name} does not exist.')
    return logger

def initialize_logger(config: ml_collections.ConfigDict) -> None:
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    logging.get_absl_handler().use_absl_log_file('log', config.path)
    logging.get_absl_handler().setFormatter(None)
    logging.info(f"{config.to_json_best_effort()}")
    logging.info(f"Started at: {cur_time}")

class BanditLogger:
    """Logger for bandit env."""

    def __init__(self, config: ml_collections.ConfigDict, log_to_console: bool, print_every_steps: int) -> None:
        self._log_to_console = log_to_console
        self._log_every_steps = config.log_every_steps
        self.steps_per_episode = config.environment.steps_per_episode
        self._print_every_steps = print_every_steps
        initialize_logger(config)

        self._episode_reward_log = 0
        self._log_step_count = 0
        self._start_time = time.time()

        self._step_reward_log = 0

        self.df = pd.DataFrame(columns=["Global_step", "Worker_step", "T", "Reward", "Mean_Reward", "Entropy_coef", "Loss", "action", "next_observation", "arm_0_prob", "arm_1_prob"])
        self._log_rows = []
        self._flush_every = 1000
        self.df_granular = pd.DataFrame(columns=["Global_step", "Reward", "Loss", "arm_0_prob", "arm_1_prob"])
        self._granular_rows = []

    def log_step(
        self,
        global_step: int,
        worker_step: int,
        reward: float,
        info: Dict,
        loss: float,
        entropy_coef: float,
        action: np.ndarray,
        next_observation: np.ndarray,
        arms_proba: np.ndarray,
        restless_lambda: float = None
    )  -> None:
        """Method to call on every step to log step or episode metrics."""
        # Step loggers
        self._step_reward_log += reward

        if restless_lambda is not None:
            self._granular_rows.append({
                "Global_step": global_step,
                "Reward": reward,
                "Loss": float(loss),
                "arm_0_prob": arms_proba[0],
                "arm_1_prob": arms_proba[1],
                "restless_lambda": restless_lambda,
            })
        else:
            self._granular_rows.append({
                "Global_step": global_step,
                "Reward": reward,
                "Loss": float(loss),
                "arm_0_prob": arms_proba[0],
                "arm_1_prob": arms_proba[1],
            })

        if (worker_step / self._log_every_steps) >= self._log_step_count:
            self._log_step_count += 1
            batch_time = time.time() - self._start_time
            self._start_time = time.time()

            row = {
                "Global_step": global_step,
                "Worker_step": worker_step,
                "T": batch_time,
                "Reward": reward,
                "Mean_Reward": self._step_reward_log / self._log_every_steps,
                "Entropy_coef": entropy_coef,
                "Loss": float(loss),
                "action": action,
                "next_observation": next_observation["vector_input"].tolist(),
                # Example: If agent chose Arm 0 (action=0) and received a reward of 1.0 in the previous step,
                # the vector_input for the current observation would be: np.concatenate(([1., 0.], [1.]))
                "arm_0_prob": arms_proba[0],
                "arm_1_prob": arms_proba[1],
            }

            self._log_rows.append(row)
            if len(self._log_rows) >= self._flush_every:
                self._flush_rows()

            if self._log_to_console:
                print(
                    f"Global step:\t{global_step}\t|"
                    f" Worker step:\t{worker_step}\t|"
                    f" T:\t{batch_time:0.2f}\t|"
                    f" Mean Reward:\t{(self._step_reward_log / self._log_every_steps):0.4f}\t|"
                    f" Entropy coef:\t{entropy_coef:0.4f}\t|"
                    f" Loss:\t{loss:0.5f}\t|"
                    f" Action:\t{action}\t|"
                    f" Next observation:\t{next_observation}\t|"
                    )
            else:
                logging.info(
                    f"Global step:\t{global_step}\t|"
                    f" Worker step:\t{worker_step}\t|"
                    f" T:\t{batch_time:0.2f}\t|"
                    f" Mean Reward:\t{(self._step_reward_log / self._log_every_steps):0.4f}\t|"
                    f" Entropy coef:\t{entropy_coef:0.4f}\t|"
                    f" Loss:\t{loss:0.5f}\t|"
                    )
            self._step_reward_log = 0

    def _flush_rows(self) -> None:
        if not self._log_rows:
            return
        self.df = pd.concat([self.df, pd.DataFrame(self._log_rows)], ignore_index=True)
        self._log_rows = []

    def close_logger(self) -> None:
        self._flush_rows()
        self.df_granular = pd.DataFrame(self._granular_rows)
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        logging.info('Shutting down.')
        logging.info(f"Ended at: {cur_time}")