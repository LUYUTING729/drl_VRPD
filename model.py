"""
model.py

This module defines the common neural network components used by both the UpperAgent (DQN-based)
and the LowerAgent (PPO-based). It includes:

  - A generic MLP builder function ("build_mlp") that creates a fully connected neural network.
  - A weight initialization utility ("init_weights") that applies Xavier uniform initialization 
    to Linear layers.
  - The DQNNetwork class used by the UpperAgent to output Q-values for dispatch decisions.
  - The ActorNetwork class used by the LowerAgent to produce action probabilities 
    (for assignment decisions, such as assign to drone, vehicle, or reject).
  - The CriticNetwork class used by the LowerAgent to estimate the state value as a scalar.
    
All network architectures are built with explicit types and default parameters. Default hidden 
layer sizes and activation functions are set as defaults if not specified.

All configuration parameters (e.g., hidden layer dimensions) can be passed as arguments by the agents. 
A default configuration is provided in the function signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def build_mlp(
    in_features: int,
    hidden_layers: List[int],
    out_features: int,
    activation_hidden: Optional[nn.Module] = nn.ReLU,
    activation_output: Optional[nn.Module] = None
) -> nn.Sequential:
    """
    Build a multi-layer perceptron (MLP) given the input dimension, a list of hidden layer sizes,
    and the output dimension.

    Args:
        in_features (int): Dimensionality of the input features.
        hidden_layers (List[int]): A list specifying the number of neurons in each hidden layer.
        out_features (int): Dimensionality of the output.
        activation_hidden (Optional[nn.Module]): Activation function for hidden layers. Default: ReLU.
        activation_output (Optional[nn.Module]): Activation function for the output layer.
            If None, no activation is applied for the output layer.

    Returns:
        nn.Sequential: A sequential model representing the MLP.
    """
    layers: List[nn.Module] = []
    prev_features: int = in_features
    for hidden_size in hidden_layers:
        # Linear layer followed by activation
        layers.append(nn.Linear(prev_features, hidden_size))
        layers.append(activation_hidden())
        prev_features = hidden_size
    # Output layer without activation unless specified
    layers.append(nn.Linear(prev_features, out_features))
    if activation_output is not None:
        layers.append(activation_output())
    return nn.Sequential(*layers)


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for a given module using Xavier uniform initialization for all Linear layers,
    and set biases to zero.
    
    Args:
        module (nn.Module): The module to initialize.
    """
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Linear):
            nn.init.xavier_uniform_(sub_module.weight)
            if sub_module.bias is not None:
                nn.init.constant_(sub_module.bias, 0.0)


class DQNNetwork(nn.Module):
    """
    DQNNetwork implements a fully connected network for the UpperAgent using the DQN method.
    It takes a compact state feature vector as input and outputs Q-values for each dispatch action.
    Typical actions: 0 = wait, 1 = dispatch.
    """
    def __init__(
        self,
        state_dim: int,
        num_actions: int = 2,
        hidden_layers: Optional[List[int]] = None
    ) -> None:
        """
        Constructor for DQNNetwork.

        Args:
            state_dim (int): Dimensionality of the state input.
            num_actions (int): Number of dispatch actions (default: 2).
            hidden_layers (Optional[List[int]]): List of hidden layer sizes. Defaults to [128, 128].
        """
        super(DQNNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.network: nn.Sequential = build_mlp(
            in_features=state_dim,
            hidden_layers=hidden_layers,
            out_features=num_actions,
            activation_hidden=nn.ReLU,
            activation_output=None  # Linear layer for Q-values
        )
        init_weights(self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Output Q-values with shape (batch_size, num_actions).
        """
        return self.network(x)


class ActorNetwork(nn.Module):
    """
    ActorNetwork implements the policy network for the LowerAgent using PPO.
    It takes as input the state features for a new request and outputs a probability distribution
    over assignment actions (e.g., assign to drone, assign to vehicle, or reject).
    """
    def __init__(
        self,
        request_state_dim: int,
        action_dim: int = 3,
        hidden_layers: Optional[List[int]] = None
    ) -> None:
        """
        Constructor for ActorNetwork.

        Args:
            request_state_dim (int): Dimensionality of the request state features.
            action_dim (int): Number of assignment actions (default: 3).
            hidden_layers (Optional[List[int]]): List of hidden layer sizes. Defaults to [128, 128].
        """
        super(ActorNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128]
        # Build MLP that outputs logits
        self.actor_mlp: nn.Sequential = build_mlp(
            in_features=request_state_dim,
            hidden_layers=hidden_layers,
            out_features=action_dim,
            activation_hidden=nn.ReLU,
            activation_output=None  # No activation; will apply softmax later
        )
        init_weights(self.actor_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action probabilities.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, request_state_dim).
        
        Returns:
            torch.Tensor: Probability distribution over actions with shape (batch_size, action_dim).
        """
        logits: torch.Tensor = self.actor_mlp(x)
        # Softmax to convert logits into probabilities.
        action_probs: torch.Tensor = F.softmax(logits, dim=-1)
        return action_probs


class CriticNetwork(nn.Module):
    """
    CriticNetwork implements the value function estimator for the LowerAgent (PPO).
    It takes as input the request state features and outputs a single scalar representing the state value.
    """
    def __init__(
        self,
        request_state_dim: int,
        hidden_layers: Optional[List[int]] = None
    ) -> None:
        """
        Constructor for CriticNetwork.

        Args:
            request_state_dim (int): Dimensionality of the request state features.
            hidden_layers (Optional[List[int]]): List of hidden layer sizes. Defaults to [128, 128].
        """
        super(CriticNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128]
        # The critic outputs a scalar value so out_features = 1
        self.critic_mlp: nn.Sequential = build_mlp(
            in_features=request_state_dim,
            hidden_layers=hidden_layers,
            out_features=1,
            activation_hidden=nn.ReLU,
            activation_output=None  # Linear output
        )
        init_weights(self.critic_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute state value.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, request_state_dim).
        
        Returns:
            torch.Tensor: State value tensor with shape (batch_size, 1).
        """
        value: torch.Tensor = self.critic_mlp(x)
        return value
