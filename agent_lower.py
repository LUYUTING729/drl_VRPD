"""
agent_lower.py

This module defines the LowerAgent class which implements a PPO-based actor-critic agent for making
assignment decisions for incoming customer requests in the same©\day delivery dispatching and routing
problem with vehicles and drones (SD3RVD).

The LowerAgent uses two neural networks from model.py:
    - ActorNetwork: outputs a probability distribution over three actions: assign to drone (0),
                      assign to vehicle (1), or reject the request (2).
    - CriticNetwork: estimates the state value for a given request state.
    
The agent collects transitions in memory during an episode and then performs PPO updates using the
clipped surrogate objective, a critic (value) loss, and an entropy bonus. Hyperparameters (such as
learning rate, clip epsilon, batch size, epochs per update, and discount factor) are provided via a 
configuration dictionary (typically from config.yaml).

Author: [Your Name]
Date: [Current Date]
"""

from typing import Any, Dict, List, Tuple
import random
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import get_logger
from model import ActorNetwork, CriticNetwork


class LowerAgent:
    """
    LowerAgent implements a PPO-based actor-critic agent for request assignment.
    It selects for each request among three discrete actions:
        0: assign to drone
        1: assign to vehicle
        2: reject
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the LowerAgent with PPO hyperparameters and neural networks.
        
        Args:
            params (Dict[str, Any]): Parameter dictionary with PPO settings.

        Expected keys in params (with defaults if missing):
            - "ppo": {
                    "learning_rate": float (default: 3e-4),
                    "clip_epsilon": float (default: 0.2),
                    "batch_size": int (default: 64),
                    "epochs_per_update": int (default: 3),
                    "gamma": float (default: 0.99),
                    "hidden_layers": List[int] (default: [128, 128])
                }
            - "state_dim_lower": int (dimension for lower agent state; default: 10)
            - "device": str, "cpu" or "cuda" (default: "cpu")
            - Optionally, "value_loss_weight": float (default: 0.5)
            - Optionally, "entropy_coeff": float (default: 0.01)
        """
        # Logger for debugging and info.
        self.logger = get_logger("LowerAgent")

        ppo_conf: Dict[str, Any] = params.get("ppo", {})
        self.learning_rate: float = float(ppo_conf.get("learning_rate", 3e-4))
        self.clip_epsilon: float = float(ppo_conf.get("clip_epsilon", 0.2))
        self.batch_size: int = int(ppo_conf.get("batch_size", 64))
        self.epochs_per_update: int = int(ppo_conf.get("epochs_per_update", 3))
        self.gamma: float = float(ppo_conf.get("gamma", 0.99))
        self.hidden_layers: List[int] = ppo_conf.get("hidden_layers", [128, 128])
        self.value_loss_weight: float = float(ppo_conf.get("value_loss_weight", 0.5))
        self.entropy_coeff: float = float(ppo_conf.get("entropy_coeff", 0.01))

        # State dimension for the lower agent (request state features)
        # Use provided "state_dim_lower" key in params or default to 10
        self.state_dim: int = int(params.get("state_dim_lower", 10))
        # Action space for request assignment: 3 discrete actions (drone, vehicle, reject)
        self.action_dim: int = 3

        # Device configuration
        self.device: str = params.get("device", "cpu")
        if self.device not in ["cpu", "cuda"]:
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.logger.info("LowerAgent will use device: %s", self.device)

        # Initialize the actor and critic networks from model.py
        self.actor: ActorNetwork = ActorNetwork(
            request_state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_layers=self.hidden_layers
        )
        self.actor.to(self.device)

        self.critic: CriticNetwork = CriticNetwork(
            request_state_dim=self.state_dim,
            hidden_layers=self.hidden_layers
        )
        self.critic.to(self.device)

        # Create a single Adam optimizer for both actor and critic parameters
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                    lr=self.learning_rate)

        # Memory buffers for PPO (to store transitions between updates)
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

        self.logger.info("LowerAgent initialized with state_dim=%d, action_dim=%d, learning_rate=%.2e",
                         self.state_dim, self.action_dim, self.learning_rate)

    def select_action(self, request_state: Any) -> int:
        """
        Select an assignment action for a request using the current actor and critic networks.
        
        Args:
            request_state (Any): A tensor or array-like representing the state features for a request.
            
        Returns:
            int: The selected action as an integer (0: assign to drone, 1: assign to vehicle, 2: reject).
            
        During selection, the agent:
            - Converts request_state to a torch.Tensor.
            - Computes action probabilities via the actor network (softmax applied to logits).
            - Samples an action stochastically.
            - Computes the log probability of the selected action.
            - Computes the state value via the critic network.
            - Stores the transition tuple (state, action, log_prob, value) for future update.
        """
        # Ensure the request_state is a torch.Tensor of type float32 and on the proper device.
        if not isinstance(request_state, torch.Tensor):
            state_tensor = torch.tensor(request_state, dtype=torch.float32)
        else:
            state_tensor = request_state.float()
        state_tensor = state_tensor.to(self.device)

        # Forward pass through actor to get action probabilities.
        self.actor.eval()  # set to eval for inference
        with torch.no_grad():
            action_probs = self.actor(state_tensor.unsqueeze(0)).squeeze(0)  # shape (action_dim,)
        # Create a categorical distribution and sample an action.
        dist = torch.distributions.Categorical(probs=action_probs)
        action = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        
        # Forward pass through critic to get state value.
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(state_tensor.unsqueeze(0)).squeeze(0).item()

        # Store transition components for later PPO update.
        self.states.append(state_tensor.detach())
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value)
        # The reward and done flag will be assigned later externally; here, append placeholder.
        # (Alternatively, the Trainer might call a separate method store_reward.)
        self.dones.append(False)

        self.logger.debug("Select action: state=%s, action=%d, log_prob=%.6f, value=%.6f",
                          state_tensor.cpu().numpy(), action, log_prob.item(), value)
        return action

    def store_reward(self, reward: float, done: bool) -> None:
        """
        Store the reward and done flag received from the environment for the current transition.
        
        Args:
            reward (float): Reward received for the transition.
            done (bool): Boolean flag indicating if the episode ended after this transition.
        """
        self.rewards.append(reward)
        self.dones[-1] = done  # update the last stored done flag

    def update(self) -> None:
        """
        Perform a PPO update of the actor and critic networks using the collected trajectory.
        
        The update process:
            - Computes discounted returns from collected rewards.
            - Computes advantages as (return - value).
            - For a defined number of epochs (epochs_per_update), it splits the trajectory into batches,
              and computes:
                * The new log-probabilities for the stored actions.
                * The probability ratio between new and old log-probabilities.
                * The clipped surrogate loss for the actor.
                * The mean squared error loss for the critic.
                * An entropy bonus.
            - Combines these losses (with appropriate weighting), backpropagates the gradients, and updates the networks.
            - After updating, clears the memory buffers.
        """
        # Convert collected lists into tensors.
        if len(self.rewards) == 0:
            self.logger.warning("No transitions to update.")
            return

        # Number of transitions
        T = len(self.rewards)

        # Compute discounted returns. Simple discounted sum over rewards.
        returns = [0.0] * T
        discounted_sum = 0.0
        for t in reversed(range(T)):
            # If done, reset discounted sum.
            if self.dones[t]:
                discounted_sum = 0.0
            discounted_sum = self.rewards[t] + self.gamma * discounted_sum
            returns[t] = discounted_sum
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Convert stored values, actions, log_probs to tensors.
        values_tensor = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.stack(self.log_probs).detach()  # shape (T,)

        # Compute advantages: advantage = returns - values.
        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack states into a tensor.
        states_tensor = torch.stack(self.states)  # shape (T, state_dim)

        # PPO update: perform for epochs_per_update epochs.
        dataset_size = T
        indices = np.arange(dataset_size)
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                # Recompute action probabilities and log probabilities with current actor.
                action_probs = self.actor(batch_states)  # shape (batch_size, action_dim)
                dist = torch.distributions.Categorical(probs=action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute probability ratio (new / old) using exponentiation of log differences.
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Compute surrogate loss for actor: take the minimum of unclipped and clipped ratio * advantage.
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Critic loss: Mean squared error between predicted values and computed returns.
                predicted_values = self.critic(batch_states).squeeze(1)  # shape (batch_size)
                critic_loss = F.mse_loss(predicted_values, batch_returns)

                # Total loss with entropy bonus.
                total_loss = actor_loss + self.value_loss_weight * critic_loss - self.entropy_coeff * entropy

                # Gradient descent step.
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.logger.info("PPO update performed: %d transitions processed.", T)

        # Clear memory buffers after updating.
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def save_model(self, path: str) -> None:
        """
        Save the actor and critic network parameters as well as the optimizer state.

        Args:
            path (str): Path to save the checkpoint.
        """
        checkpoint: Dict[str, Any] = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "batch_size": self.batch_size,
            "epochs_per_update": self.epochs_per_update,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, path)
        self.logger.info("LowerAgent model saved to %s", path)

    def load_model(self, path: str) -> None:
        """
        Load the actor and critic network parameters as well as the optimizer state from saved checkpoint.

        Args:
            path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint.get("actor_state_dict", {}))
        self.critic.load_state_dict(checkpoint.get("critic_state_dict", {}))
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
        self.clip_epsilon = checkpoint.get("clip_epsilon", self.clip_epsilon)
        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self.epochs_per_update = checkpoint.get("epochs_per_update", self.epochs_per_update)
        self.gamma = checkpoint.get("gamma", self.gamma)
        self.logger.info("LowerAgent model loaded from %s", path)
