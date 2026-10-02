"""Environments module."""

from typing import Tuple, Dict, Any
import numpy as np

class BanditEnv:
    """Environment for an N-armed bandit task.

    An agent has access to as many discrete actions as there are unique bandit arms.
    Each bandit arm has a win probability determined at episode start.
    Win probabilities are constant within an episode.
    Episodes consist of several bandit decisions, also known as trials, and continue for steps_per_episode trials.
    The env can have several reward structures:
    - "independent": Win probabilities of each arm are independent [0, 1].
    - "correlated": Win probabilities of all arms sum to 1.
    """

    def __init__(self,
        env_name: str = "bandit",
        steps_per_episode: int = 100,
        num_arms: int = 2,
        include_prev_action: bool = True,
        include_prev_reward: bool = True,
        reward_structure: str = "correlated",
        ) -> None:
        """Initialize Env class for a multi-armed bandit task.

        Args:
            steps_per_episode: Steps/trials per episode.
            num_arms: Number of available bandit arms.
            include_prev_action: Include action from last step in observation.
            include_prev_reward: Include reward from last step in observation.
            reward_structure:
              - "independent": Win probabilities of each arm are independent
              - "dependent_u": Sum to 1, p ~ U([0,1])
              - "dependent_e": Sum to 1, p ~ U({0.1,0.9})
              - "dependent_m": Sum to 1, p ~ U({0.25,0.75})
              - "dependent_h": Sum to 1, p ~ U({0.4,0.6})

        Returns:
            None
        """
        self.name = env_name
        self._steps_per_episode = steps_per_episode
        self._num_arms = num_arms
        self._include_prev_action = include_prev_action
        self._include_prev_reward = include_prev_reward
        self._reward_structure = reward_structure

        self._arm_probs = np.zeros(self._num_arms)

        self.num_actions = self._num_arms
        self._done = False

        self.reset()

    def _get_new_bandits(self) -> None:
        """Get bandit arm win probabilites for this episode."""

        self._arm_probs = np.zeros(self._num_arms)

        if self._reward_structure == "independent":
            # Win probabilities of each arm are independent
            self._arm_probs = np.random.uniform(size=self._num_arms)
        elif self._reward_structure == "dependent_u":
            # Win probabilities of all arms sum to 1
            self._arm_probs = np.random.uniform(size=self._num_arms)
            self._arm_probs = self._arm_probs / np.sum(self._arm_probs)
        elif self._reward_structure == "dependent_e":
            # Win probabilities of all arms sum to 1, and are either 0.1 or 0.9
            if self._num_arms != 2:
                raise ValueError(f'Reward structure {self._reward_structure} only works with 2 arms.')
            arm_prob_1 = np.random.choice(np.array([0.1, 0.9]))
            arm_prob_2 = 1. - arm_prob_1
            self._arm_probs = np.array([arm_prob_1, arm_prob_2])
        elif self._reward_structure == "dependent_m":
            # Win probabilities of all arms sum to 1, and are either 0.25 or 0.75
            if self._num_arms != 2:
                raise ValueError(f'Reward structure {self._reward_structure} only works with 2 arms.')
            arm_prob_1 = np.random.choice(np.array([0.25, 0.75]))
            arm_prob_2 = 1. - arm_prob_1
            self._arm_probs = np.array([arm_prob_1, arm_prob_2])
        elif self._reward_structure == "dependent_h":
            # Win probabilities of all arms sum to 1, and are either 0.4 or 0.6
            if self._num_arms != 2:
                raise ValueError(f'Reward structure {self._reward_structure} only works with 2 arms.')
            arm_prob_1 = np.random.choice(np.array([0.4, 0.6]))
            arm_prob_2 = 1. - arm_prob_1
            self._arm_probs = np.array([arm_prob_1, arm_prob_2])
        else:
            raise NameError(f'Reward structure {self._reward_structure} does not exist.')

    def reset(self, arm_probs = None) -> Dict[str, np.ndarray]:
        """Reset environment at the start of each episode.

        Args:
            arm_probs: Optional numpy array of arm win probabilities.

        Returns:
            observation: Dictionary of numpy arrays for network input.
                Example: If agent chose Arm 0 (action=0) and received a reward of 1.0 in the previous step,
                the vector_input for the current observation would be: np.concatenate(([1., 0.], [1.]))
        """
        if arm_probs is not None:
            # Manual win probabilities for each arm
            if not isinstance(arm_probs, np.ndarray):
                raise TypeError(f'arm_probs must be a numpy array.')
            if np.shape(arm_probs) != np.shape(self._arm_probs):
                raise ValueError(f'arm_probs shape {np.shape(arm_probs)} does not match required shape {np.shape(self._arm_probs)}.')
            if np.any(self._arm_probs < 0.0) or np.any(self._arm_probs > 1.0):
                raise ValueError(f'Arm win probs {self._arm_probs} need to be between 0 and 1.')
            self._arm_probs = arm_probs
        else:
            # Automatic win probabilities for each arm based on reward_structure
            self._get_new_bandits()

        vector_input = np.array([])

        if self._include_prev_action:
            vector_input = np.concatenate((vector_input, np.zeros(self.num_actions)))
        if self._include_prev_reward:
            vector_input = np.concatenate((vector_input, np.zeros(1)))

        self._info = {
            "episode_reward": 0,
        }
        self._cur_step = 0
        self._done = False

        return {
            "vector_input": vector_input,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the environment.

        Args:
            action: Discrete action taken by the agent on this step.

        Returns:
            observation: Dictionary of numpy arrays for network input.
                Example: If agent chose Arm 0 (action=0) and received a reward of 1.0 in the previous step,
                the vector_input for the current observation would be: np.concatenate(([1., 0.], [1.]))
            reward: Scalar reward value for this step.
            done: Boolean for terminal episode state.
            info: Dictionary of episode information.
        """
        action = int(action)
        reward = 0.0

        cur_arm_prob = self._arm_probs[action]
        win = np.random.uniform() < cur_arm_prob

        if win:
            reward += 1.

        vector_input = np.array([])

        if self._include_prev_action:
            one_hot_prev_action = np.zeros(self.num_actions)
            one_hot_prev_action[action] = 1
            vector_input = np.concatenate((vector_input, one_hot_prev_action))
        if self._include_prev_reward:
            vector_input = np.append(vector_input, reward)

        if self._cur_step == self._steps_per_episode - 1:
            self._done = True
        self._cur_step += 1

        return (
            {
                "vector_input": vector_input,
            },
            reward,
            self._done,
            self._info
        )

class RestlessBanditEnv:
    """Environment for a 2-armed restless bandit task.
 
    An agent has access to as many discrete actions as there are unique bandit arms.
    Arm win probabilities are restricted to {0.1, 0.9} (perfectly anti-correlated:
    p2 = 1 - p1) and jump between these two values during the episode according to
    a Poisson-like process. Each episode is randomly assigned a volatility condition
    ("low" or "high") that determines the per-step jump probability, but this
    condition is never exposed to the agent -- it must be inferred from the reward
    sequence. Episodes consist of several bandit decisions, also known as trials,
    and continue for steps_per_episode trials.
    """
 
    def __init__(self,
        env_name: str = "restless_bandit",
        steps_per_episode: int = 100,
        num_arms: int = 2,
        include_prev_action: bool = True,
        include_prev_reward: bool = True,
        arm_values: Tuple[float, float] = (0.1, 0.9),
        lambda_low: float = 0.015, # Jump every 67 steps on average
        lambda_high: float = 0.1, # Jump every 10 steps on average
        reward_structure: str = "restless",
        ) -> None:
        """Initialize Env class for a restless multi-armed bandit task.
 
        Args:
            steps_per_episode: Steps/trials per episode.
            num_arms: Number of available bandit arms (must be 2).
            include_prev_action: Include action from last step in observation.
            include_prev_reward: Include reward from last step in observation.
            arm_values: The two possible values p1 can take (arm 2 is always
                1 - arm 1). Defaults to (0.1, 0.9).
            lambda_low: Per-step jump probability for "low volatility" episodes.
            lambda_high: Per-step jump probability for "high volatility" episodes.
 
        Returns:
            None
        """
        if num_arms != 2:
            raise ValueError(f'RestlessBanditEnv only supports num_arms=2, got {num_arms}.')
 
        self.name = env_name
        self._steps_per_episode = steps_per_episode
        self._num_arms = num_arms
        self._include_prev_action = include_prev_action
        self._include_prev_reward = include_prev_reward
        self._arm_values = arm_values
        self._lambda_low = lambda_low
        self._lambda_high = lambda_high
        self._reward_structure = reward_structure
 
        self._arm_probs = np.zeros(self._num_arms)
 
        self.num_actions = self._num_arms
        self._done = False
 
        self.reset()
 
    def _get_new_bandits(self) -> None:
        """Sample a new volatility condition and initial arm win probabilities
        for this episode.
 
        Win probabilities sum to 1 and are restricted to {arm_values[0], arm_values[1]}
        (e.g. {0.1, 0.9}). The episode's volatility ("low" or "high") is sampled with
        equal probability and determines the jump rate used in step().
        """
        # Volatility condition for this episode (hidden from the agent).
        self._volatility = np.random.choice(["low", "high"])
        self._lam = self._lambda_low if self._volatility == "low" else self._lambda_high
 
        # Initial arm 1 probability, coin flip between the two possible values.
        arm_prob_1 = np.random.choice(np.array(self._arm_values))
        arm_prob_2 = 1. - arm_prob_1
        self._arm_probs = np.array([arm_prob_1, arm_prob_2])
 
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment at the start of each episode.
 
        Returns:
            observation: Dictionary of numpy arrays for network input.
                Example: If agent chose Arm 0 (action=0) and received a reward of 1.0 in the previous step,
                the vector_input for the current observation would be: np.concatenate(([1., 0.], [1.]))
        """
        self._get_new_bandits()
 
        vector_input = np.array([])
 
        if self._include_prev_action:
            vector_input = np.concatenate((vector_input, np.zeros(self.num_actions)))
        if self._include_prev_reward:
            vector_input = np.concatenate((vector_input, np.zeros(1)))
 
        self._info = {
            "episode_reward": 0,
            "volatility": self._volatility,
            "n_jumps": 0,
            "arm_probs_history": [self._arm_probs.copy()],
            "jump_history": [False],
        }
        self._cur_step = 0
        self._done = False
 
        return {
            "vector_input": vector_input,
        }
 
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the environment.
 
        On each step, before computing the reward, the arm probabilities may jump
        (flip between the two possible values) with probability self._lam, which
        depends on the episode's volatility condition. This drift is exogenous --
        it happens regardless of which arm the agent chooses.
 
        Args:
            action: Discrete action taken by the agent on this step.
 
        Returns:
            observation: Dictionary of numpy arrays for network input.
                Example: If agent chose Arm 0 (action=0) and received a reward of 1.0 in the previous step,
                the vector_input for the current observation would be: np.concatenate(([1., 0.], [1.]))
            reward: Scalar reward value for this step.
            done: Boolean for terminal episode state.
            info: Dictionary of episode information.
        """
        action = int(action)
 
        # Exogenous drift: possibly jump arm probabilities before resolving the trial.
        jumped = bool(np.random.uniform() < self._lam)
        if jumped:
            self._arm_probs = self._arm_probs[::-1].copy()
            self._info["n_jumps"] += 1
 
        reward = 0.0
        cur_arm_prob = self._arm_probs[action]
        win = np.random.uniform() < cur_arm_prob
 
        if win:
            reward += 1.
 
        vector_input = np.array([])
 
        if self._include_prev_action:
            one_hot_prev_action = np.zeros(self.num_actions)
            one_hot_prev_action[action] = 1
            vector_input = np.concatenate((vector_input, one_hot_prev_action))
        if self._include_prev_reward:
            vector_input = np.append(vector_input, reward)
 
        # Track per-trial bookkeeping (best arm always pays self._arm_values[1], e.g. 0.9).
        self._info["episode_reward"] += reward
        self._info["arm_probs_history"].append(self._arm_probs.copy())
        self._info["jump_history"].append(jumped)
        self._info["optimal_arm"] = int(np.argmax(self._arm_probs))
        self._info["optimal_reward_prob"] = float(np.max(self._arm_probs))
 
        if self._cur_step == self._steps_per_episode - 1:
            self._done = True
        self._cur_step += 1
 
        return (
            {
                "vector_input": vector_input,
            },
            reward,
            self._done,
            self._info
        )
    
def create_env(env_config=None, env_type = None):
    """Create a reinforcement learning environment.

    Parameters:
        env_config: Configuration dictionary for the environment.
        env_type: Type of environment to create ("bandit" or "restless_bandit").
    """
    if env_type == "restless_bandit":
        env = RestlessBanditEnv(**env_config)
    else:
        env = BanditEnv(**env_config)
    return env