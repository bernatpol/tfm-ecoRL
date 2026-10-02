#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import NamedTuple, Dict, Any, Tuple


# --- Trajectory and Buffer from -bgt_AP7ojF2 (and adapted for PyTorch) ---
class Trajectory(NamedTuple):
    """Structure and store a complete sequence of an agent's experience
    over multiple steps within an episode or a segment of an episode.
    """
    observations: np.ndarray # This will be processed to a torch tensor later
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    rnn_state: Any # Will be torch.Tensor

class Buffer:
    """Buffer for reinforcement learning trajectories.

    Temporarily store individual steps of experience as they occur during
    interactions with the environment.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset buffers."""
        self._observations = []
        self._actions = []
        self._rewards = []
        self._discounts = []
        self._rnn_state = None
        self.t = 0

    def append(
        self,
        obs: Dict,
        action: any = None,
        reward: float = None,
        next_obs: np.ndarray = None,
        rnn_state: torch.Tensor = None,
        done: bool = None,
    ) -> None:
        """Appends an observation, action, reward, and discount to the buffer."""
        if len(self._observations) == 0:
            self._observations.append(obs)
            self._rnn_state = rnn_state

        self._observations.append(next_obs)
        self._actions.append(action)
        self._rewards.append(float(reward))
        if done:
            self._discounts.append(0.0)
        else:
            self._discounts.append(1.0)
        self.t += 1

    def drain(self) -> Trajectory:
        """Return Trajectory of experience, and then clear Trajectory."""
        trajectory = Trajectory(
            observations = np.array(self._observations),
            actions = np.array(self._actions),
            rewards = np.array(self._rewards),
            discounts = np.array(self._discounts),
            rnn_state = self._rnn_state,
        )
        self.reset()
        return trajectory

class LowRankRNN(nn.Module):
    """Custom RNN cell with a low-rank factorized hidden-to-hidden matrix."""
    def __init__(self, input_size: int, hidden_size: int, rank: int):
        super(LowRankRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Input-to-hidden transformation
        self.W_ih = nn.Linear(input_size, hidden_size)
        
        # Hidden-to-hidden Low-Rank Factorization (W_hh ≈ U @ V)
        # V projects the hidden state down to the lower 'rank' dimension
        self.V_hh = nn.Linear(hidden_size, rank, bias=False) 
        # U projects it back up to the 'hidden_size' dimension
        self.U_hh = nn.Linear(rank, hidden_size)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [seq_len, batch_size, input_size]
        # h_0 shape: [1, batch_size, hidden_size]
        seq_len, batch_size, _ = x.shape
        
        outputs = []
        h_t = h_0[0] # Extract the actual hidden state: [batch_size, hidden_size]
        
        for t in range(seq_len):
            x_t = x[t] # [batch_size, input_size]
            
            # 1. Project hidden state down to rank, then back up
            low_rank_h = self.U_hh(self.V_hh(h_t))
            
            # 2. Standard RNN tanh update
            h_t = torch.tanh(self.W_ih(x_t) + low_rank_h)
            
            outputs.append(h_t.unsqueeze(0))
            
        # Stack all time steps back together: [seq_len, batch_size, hidden_size]
        output_seq = torch.cat(outputs, dim=0)
        
        # Return sequence and the final hidden state (re-adding the num_layers=1 dimension)
        return output_seq, h_t.unsqueeze(0)

# --- DefaultAgent (refactored to PyTorch) ---
class DefaultAgent(nn.Module):
    """Advantage actor-critic agent that responds with discrete actions."""

    def __init__(self,
            observation: np.ndarray,
            random_seed: int = 42,
            num_lstm_units: int = 64, # Keeping your variable name
            rnn_rank: int = 16,       # <-- NEW PARAMETER
            num_actions: int = 2,
            learning_rate_start: float = 0.0001,
            learning_rate_end: float = 0.0001,
            total_training_steps: int = 100000,
            gamma: float = 1.0,
            v_loss_coef: float = 0.05, # beta_v (state-value estimate cost)
            e_loss_coef_start: float = 0.0, # beta_e (entropy cost) will decay linearly to e_loss_coef_end
            e_loss_coef_end: float = 0.0,
            max_unroll_steps: int = 300, # Number of steps in an episode
            global_norm_grad_clip: float = 50.0,
            ) -> None:

        super(DefaultAgent, self).__init__()
        self.name = 'Default'
        self._num_lstm_units = num_lstm_units
        self._rnn_rank = rnn_rank
        self._num_actions = num_actions
        self._learning_rate_start = learning_rate_start
        self._learning_rate_end = learning_rate_end
        self._gamma = gamma
        self._v_loss_coef = v_loss_coef
        self._max_unroll_steps = max_unroll_steps # When there is a gradient update
        self._total_training_steps = total_training_steps
        self._global_norm_grad_clip = global_norm_grad_clip

        # Initialize the buffer
        self.buffer = Buffer()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define network layers:
        # - input_size is determined from the initial observation ([previous action (num. arms), previous reward])
        input_size = observation['vector_input'].shape[0] if isinstance(observation, dict) else observation.shape[0]
        self.rnn = LowRankRNN(input_size=input_size,hidden_size=self._num_lstm_units,rank=self._rnn_rank)        
        self.action_head = nn.Linear(self._num_lstm_units, self._num_actions)
        self.value_head = nn.Linear(self._num_lstm_units, 1)

        # Move model to device
        self.to(self.device)

        # Approximate how many learning steps
        approx_learning_steps = self._total_training_steps // self._max_unroll_steps
        self._learning_rate_decay_steps = approx_learning_steps

        # Get schedule for annealing entropy loss term (simplified for PyTorch)
        self._e_loss_coef_start = e_loss_coef_start
        self._e_loss_coef_end = e_loss_coef_end
        self.e_loss_schedule = self._e_loss_schedule_fn # linear decay from start to end value over decay steps

        self.e_loss_count = 0
        self.e_loss_coef = self.e_loss_schedule(self.e_loss_count)

        self._init_optimizer()

    def forward(
            self,
            task_input: torch.Tensor,
            rnn_state: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actor-critic RNN with one LSTM layer.

        Args:
            task_input: Inputs [Time/Batch, Features].
            rnn_state: Previous RNN hidden state.

        Returns:
            pi_out: Activations for each action [Time, num_actions].
            v_out: Value baseline [Time, 1].
            state: New LSTM hidden and cell state (h, c tuple).
        """
        # task_input shape: [seq_len, batch_size, input_size]
        # For single step, seq_len=1, batch_size=1
        rnn_output, state = self.rnn(task_input, rnn_state)

        pi_out = self.action_head(rnn_output)
        v_out = self.value_head(rnn_output)

        return pi_out.squeeze(1), v_out.squeeze(1), state # Squeeze batch dimension if batch_size=1

    def _init_model(self, observation): # This method is largely unused in PyTorch with nn.Module
        """Initialize model and parameters. (This is primarily handled in __init__ for PyTorch)"""
        pass # Model layers are defined in __init__

    def get_initial_rnn_state(self) -> torch.Tensor:
        """Create initial RNN state of zeros."""
        h_0 = torch.zeros(1, 1, self._num_lstm_units, dtype=torch.float32).to(self.device)
        return h_0
    
    def _init_optimizer(self) -> None:
        """Initialize Adam optimizer for training."""
        # Learning rate schedule (linear decay)
        self.opt = optim.Adam(self.parameters(), lr=self._learning_rate_start)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda_fn)

    def _e_loss_schedule_fn(self, step):
        # Linear decay of entropy loss coefficient from start to end value over decay steps
        if self._e_loss_coef_start != self._e_loss_coef_end and self._learning_rate_decay_steps > 0:
            return self._e_loss_coef_start - (self._e_loss_coef_start - self._e_loss_coef_end) * (step / self._learning_rate_decay_steps)
        return self._e_loss_coef_start

    def _lr_lambda_fn(self, step):
        if self._learning_rate_decay_steps > 0:
            return max(0.0, 1.0 - (step / self._learning_rate_decay_steps))
        return 1.0

    @torch.no_grad()
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        rnn_state: torch.Tensor
        ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wrapper for model_step."""
        action, pi_out, v_out, rnn_state, rnn_output = self.model_step(
            observation,
            rnn_state)
        return action, pi_out, v_out, rnn_state, rnn_output

    def model_step(self, observation, rnn_state):
        """Step the model once and select action via softmax policy."""
        task_input = torch.tensor(observation['vector_input'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # task_input shape: [1, 1, input_size]

        pi_out_logits, v_out, next_rnn_state = self.forward(task_input, rnn_state)
        # pi_out_logits shape: [1, num_actions]
        # v_out shape: [1, 1]

        probs = F.softmax(pi_out_logits, dim=-1) # Apply softmax to get probabilities
        dist = Categorical(probs)
        action = dist.sample() # Sample action from the distribution

        # For logging, we need the raw rnn_output (the output from the RNN layer before linear heads)
        # To get this, we'd need to modify forward or re-run RNN. Let's re-run for simplicity here, it's just one step.
        rnn_output, _ = self.rnn(task_input, rnn_state) # Rerun to get rnn_output directly if needed for buffer

        return action.item(), pi_out_logits.squeeze(0), v_out.squeeze(0), next_rnn_state, rnn_output.squeeze(0).squeeze(0)

    def update(self, done, update_params=True):
        """Update model from an episode trajectory."""
        loss = None
        grads = None
        num_steps = 0
        if self.buffer.t == self._max_unroll_steps or done:
            num_steps = self.buffer.t
            trajectory = self.buffer.drain()
            trajectory = self.process_trajectory_observations(trajectory)

            if update_params:
                loss = self.loss_fn(
                    trajectory=trajectory,
                    e_loss_coef=self.e_loss_coef)

                self.opt.zero_grad() # Clear previous gradients
                loss.backward() # Compute gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), self._global_norm_grad_clip) # Gradient clipping
                self.opt.step() # Update model parameters
                self.scheduler.step() # Update learning rate

                self.decrement_e_loss()
                grads = {name: param.grad.cpu().numpy() for name, param in self.named_parameters() if param.grad is not None} # For logging
                loss_val = loss.item() # For logging
            else:
                # If not updating parameters, just calculate loss for reporting if needed
                with torch.no_grad():
                    loss = self.loss_fn(
                        trajectory=trajectory,
                        e_loss_coef=self.e_loss_coef)
                    loss_val = loss.item()

        return loss_val if loss is not None else None, grads, int(num_steps)

    def process_trajectory_observations(self, trajectory) -> Dict[str, Any]:
        """Unpack observations from dictionary into their own torch.Tensor."""
        all_vector_input_np = np.stack([obs['vector_input'] for obs in trajectory.observations], axis=0)

        # Convert numpy arrays to torch tensors and move to device
        vector_input_tensor = torch.tensor(all_vector_input_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        actions_tensor = torch.tensor(trajectory.actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(trajectory.rewards, dtype=torch.float32).to(self.device)
        discounts_tensor = torch.tensor(trajectory.discounts, dtype=torch.float32).to(self.device)
        rnn_state_tensor = trajectory.rnn_state.to(self.device)

        trajectory = {
            "vector_input": vector_input_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "discounts": discounts_tensor,
            "rnn_state": rnn_state_tensor
        }
        return trajectory

    def decrement_e_loss(self) -> None:
        self.e_loss_count += 1
        self.e_loss_coef = self.e_loss_schedule(self.e_loss_count)

    def loss_fn(self, trajectory: Dict[str, Any], e_loss_coef: float) -> torch.Tensor:
        """Discrete actor-critic loss."""

        # Run experienced trajectory through model
        pi_out_logits, v_out, _ = self.forward(
            trajectory['vector_input'],
            trajectory['rnn_state'],
            )
        # pi_out_logits shape: [seq_len, num_actions]
        # v_out shape: [seq_len, 1]

        # Calculate discounted td errors
        # v_tm1 = V(s_t) and v_t = V(s_{t+1})
        v_tm1 = v_out[:-1].squeeze(-1) # Values for states S_0 to S_{T-1}
        v_t = v_out[1:].squeeze(-1)    # Values for states S_1 to S_T

        # R_t is only with k=0, so
        #  delta_t := R_t - V(s_t) = 
        # = r_t + gamma * V(s_{t+1}) - V(s_t)
        target = trajectory['rewards'] + trajectory['discounts'] * self._gamma * v_t.detach()
        delta_t_errors = target - v_tm1

        # Critic loss (Mean Squared Error for value prediction)
        critic_loss = torch.mean(delta_t_errors**2) # Corrected from F.mse_loss(...)

        # Actor loss (Policy Gradient with advantages)
        # Log probabilities of all taken actions
        log_probs = F.log_softmax(pi_out_logits[:-1], dim=-1)
        taken_action_log_probs = log_probs.gather(-1, trajectory['actions'].unsqueeze(-1)).squeeze(-1)
        actor_loss = -(taken_action_log_probs * delta_t_errors.detach()).mean()

        # Entropy loss
        probs = F.softmax(pi_out_logits[:-1], dim=-1) # Fix: use pi_out_logits[:-1] for entropy
        entropy = -(probs * log_probs).sum(dim=-1).sum() # - sum(p * log(p)) averaged over time steps

        # Weighted sum of all loss terms
        all_loss = actor_loss + (self._v_loss_coef * critic_loss) - (e_loss_coef * entropy) # Note: entropy is usually subtracted, should we change it?
        return all_loss

def create_agent(
    observation: Dict[str, np.ndarray],
    num_actions: int,
    agent_config=None,
    ):
    """Create selected agent."""
    agent = DefaultAgent(
        observation=observation,
        num_actions=num_actions,
        **agent_config)
    return agent