"""
agent_upper.py

This module defines the UpperAgent class which implements a Deep Q-Network (DQN)
for making dispatch decisions in the hierarchical deep reinforcement learning (HDDRL)
framework for the same©\day delivery dispatching and routing problem with vehicles and drones (SD3RVD).

The UpperAgent decides whether a vehicle at the depot should wait or dispatch immediately.
It employs a DQN with experience replay, target network updates, and an ¦Å-greedy strategy.

The agent¡¯s hyperparameters and settings are loaded from a configuration dictionary (from config.yaml).
It uses network components from model.py and utility functions from utils.py.
"""

import os
import random
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import get_logger  # For logging and configuration
from model import DQNNetwork  # DQN network architecture from model.py

# Define a simple ReplayBuffer class for experience replay.
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Initialize the ReplayBuffer with given capacity.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity: int = capacity
        self.buffer: List[Tuple[Any, int, float, Any, bool]] = []

    def push(self, transition: Tuple[Any, int, float, Any, bool]) -> None:
        """
        Save a transition into the buffer.

        Args:
            transition (Tuple): A tuple of (state, action, reward, next_state, done).
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[Any, int, float, Any, bool]]:
        """
        Sample a batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            List[Tuple]: A list of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """
        Return the current number of transitions stored.

        Returns:
            int: Number of transitions.
        """
        return len(self.buffer)


class UpperAgent:
    """
    UpperAgent implements a DQN-based agent for dispatch decisions.
    It selects whether a vehicle should wait (action 0) or dispatch immediately (action 1)
    based on the current state of the environment. It uses experience replay, a target network,
    and an ¦Å-greedy policy.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the UpperAgent with configuration parameters.

        Args:
            params (Dict[str, Any]): Parameter dictionary containing hyperparameters
                                     and configuration settings.
        """
        self.params: Dict[str, Any] = params

        # Extract DQN parameters from config
        dqn_conf: Dict[str, Any] = params.get("dqn", {})
        self.initial_lr: float = float(dqn_conf.get("initial_learning_rate", 1e-4))
        self.decay_factor: float = float(dqn_conf.get("decay_factor", 0.96))
        self.decay_interval: int = int(dqn_conf.get("decay_interval", 6000))
        self.batch_size: int = int(dqn_conf.get("batch_size", 32))
        self.gamma: float = float(dqn_conf.get("gamma", 0.99))

        # Exploration parameters (¦Å-greedy)
        self.epsilon_start: float = float(params.get("epsilon_start", 1.0))
        self.epsilon_min: float = float(params.get("epsilon_min", 0.1))
        self.epsilon_decay: float = float(params.get("epsilon_decay", 0.995))
        self.epsilon: float = self.epsilon_start

        # Target network update frequency and replay buffer capacity
        self.target_update_frequency: int = int(params.get("target_update_frequency", 1000))
        self.replay_buffer_capacity: int = int(params.get("replay_buffer_capacity", 10000))

        # State dimension and action space
        self.state_dim: int = int(params.get("state_dim", 10))
        self.num_actions: int = 2  # Binary: 0 = wait, 1 = dispatch

        # Device configuration
        self.device: str = params.get("device", "cpu")
        if self.device not in ["cpu", "cuda"]:
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # Initialize the main Q-network using DQNNetwork from model.py.
        # Use hidden_layers from config if provided; default to [128, 128].
        hidden_layers: List[int] = dqn_conf.get("hidden_layers", [128, 128])
        self.q_network: DQNNetwork = DQNNetwork(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            hidden_layers=hidden_layers
        )
        self.q_network.to(self.device)

        # Initialize the target Q-network as a copy of the main network.
        self.target_network: DQNNetwork = DQNNetwork(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            hidden_layers=hidden_layers
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(self.device)

        # Set up the optimizer (Adam) with the initial learning rate.
        self.optimizer: optim.Adam = optim.Adam(self.q_network.parameters(), lr=self.initial_lr)

        # Initialize the experience replay buffer.
        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity=self.replay_buffer_capacity)

        # Update counter for scheduling target network updates and learning rate decay.
        self.update_count: int = 0

        # Logger for debugging and information.
        self.logger = get_logger("UpperAgent")
        self.logger.info("UpperAgent initialized with state_dim=%d, num_actions=%d, device=%s",
                         self.state_dim, self.num_actions, self.device)

    def select_action(self, state: torch.Tensor) -> int:
        """
        Select an action (0 for wait, 1 for dispatch) using an ¦Å-greedy policy.

        Args:
            state (torch.Tensor): State tensor with shape matching input dimension.

        Returns:
            int: The chosen action (0 or 1).
        """
        # Ensure state is a torch tensor and move to the agent's device.
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        # Choose action using ¦Å-greedy strategy.
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
            self.logger.debug("Exploration: Selected random action %d (epsilon=%.4f)", action, self.epsilon)
            return action
        else:
            self.q_network.eval()
            with torch.no_grad():
                # Add batch dimension if necessary.
                q_values = self.q_network(state.unsqueeze(0))  # Shape: (1, num_actions)
            action = int(torch.argmax(q_values, dim=1).item())
            self.logger.debug("Exploitation: Selected action %d from Q-values %s (epsilon=%.4f)",
                              action, q_values.cpu().numpy(), self.epsilon)
            return action

    def store_transition(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor, done: bool) -> None:
        """
        Store a transition in the replay buffer.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state after action.
            done (bool): Flag indicating if the episode ended.
        """
        # Convert tensors to CPU numpy arrays for storage.
        state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        next_state_np = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
        transition = (state_np, action, reward, next_state_np, done)
        self.replay_buffer.push(transition)

    def update(self, current_epoch: int = 0) -> None:
        """
        Update the Q-network from a batch of transitions sampled from the replay buffer.
        Also performs learning rate decay and target network updates as specified.

        Args:
            current_epoch (int, optional): Current training epoch; used for scheduler adjustments.
                                           Defaults to 0.
        """
        if len(self.replay_buffer) < self.batch_size:
            self.logger.debug("Not enough samples in replay buffer (%d/%d) to update.",
                              len(self.replay_buffer), self.batch_size)
            return

        # Sample a batch of transitions.
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert data into torch tensors.
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute predicted Q-values for the actions taken.
        self.q_network.train()
        q_values = self.q_network(states_tensor)  # Shape: (batch_size, num_actions)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # Shape: (batch_size)

        # Compute target Q-values using the target network.
        self.target_network.eval()
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)  # Shape: (batch_size, num_actions)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            # Calculate target: if done then target = reward, else reward + gamma * max_next_q_value
            q_target = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        # Calculate Mean Squared Error loss between predicted Q-value and target.
        loss = F.mse_loss(q_value, q_target)

        # Perform gradient descent step.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.debug("Update %d: Loss = %.6f", self.update_count, loss.item())

        # Learning rate decay: Apply decay every decay_interval updates.
        if (self.update_count + 1) % self.decay_interval == 0:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group["lr"]
                new_lr = old_lr * self.decay_factor
                param_group["lr"] = new_lr
                self.logger.debug("Learning rate decayed from %.6e to %.6e", old_lr, new_lr)

        # Decay epsilon for exploration.
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.logger.debug("Epsilon decayed from %.6f to %.6f", old_epsilon, self.epsilon)

        # Update target network periodically.
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.logger.debug("Target network updated at step %d", self.update_count)

    def save_model(self, path: str) -> None:
        """
        Save the Q-network model and optimizer state to the specified file path.

        Args:
            path (str): File path for model saving.
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count
        }
        torch.save(checkpoint, path)
        self.logger.info("Model saved at: %s", path)

    def load_model(self, path: str) -> None:
        """
        Load the Q-network model and optimizer state from the specified file path.
        Updates the target network accordingly.

        Args:
            path (str): File path from which to load the model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint.get("q_network_state_dict", {}))
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        self.epsilon = checkpoint.get("epsilon", self.epsilon_start)
        self.update_count = checkpoint.get("update_count", 0)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.info("Model loaded from: %s", path)
        
# End of agent_upper.py
