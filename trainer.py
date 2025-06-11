"""
trainer.py

This module implements the Trainer class, which orchestrates training of the hierarchical deep 
reinforcement learning (HDDRL) framework for the same©\day delivery dispatching and routing problem 
(SD3RVD). It integrates the environment, UpperAgent (DQN for dispatch decisions), LowerAgent (PPO for 
assignment decisions), and routing heuristics to simulate episodes (¡°days¡±), manage reward shaping, 
coordinate agent updates, and log evaluation metrics.

The Trainer class follows the design and plan specified by the paper. It uses configuration settings 
from config.yaml and employs helper methods to extract state features for both upper and lower agents.
"""

import random
import math
import numpy as np
import torch
from typing import Any, Dict, List, Tuple

# Import modules from project files
from utils import get_logger, compute_euclidean_distance
from env import SD3RVDEnv
from agent_upper import UpperAgent
from agent_lower import LowerAgent
from heuristics import Heuristics

class Trainer:
    """
    Trainer class that coordinates simulation, agent actions, reward shaping, and model updates.
    
    Attributes:
        env (SD3RVDEnv): The simulation environment.
        upper_agent (UpperAgent): The DQN-based agent for dispatch decisions.
        lower_agent (LowerAgent): The PPO-based agent for request assignments.
        heuristics (Any): Reference to the Heuristics module (static methods).
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        logger (Logger): Logger for logging metrics and progress.
        evaluation_frequency (int): How often (in epochs) to perform evaluation checkpoints.
        lower_state_dim (int): Dimensionality of lower agent state features.
    """
    
    def __init__(self, env: SD3RVDEnv, upper_agent: UpperAgent, lower_agent: LowerAgent,
                 heuristics: Any, config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer with the environment, agents, heuristics, and configuration.
        
        Args:
            env (SD3RVDEnv): Simulation environment instance.
            upper_agent (UpperAgent): Upper agent instance (DQN).
            lower_agent (LowerAgent): Lower agent instance (PPO).
            heuristics (Any): Heuristics module (with static methods).
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.env = env
        self.upper_agent = upper_agent
        self.lower_agent = lower_agent
        self.heuristics = heuristics
        self.config = config
        
        self.logger = get_logger("Trainer")
        
        # Evaluation frequency from training config, default to 1000 epochs if not specified.
        self.evaluation_frequency: int = int(
            config.get("training", {}).get("evaluation_frequency", 1000)
        )
        # For constructing lower state vectors, get lower state dimension; default to 10.
        self.lower_state_dim: int = int(config.get("state_dim_lower", 10))
        
        # (Optional) Initialize any additional shared data structures if needed.
        self.logger.info("Trainer initialized with evaluation_frequency=%d and lower_state_dim=%d",
                         self.evaluation_frequency, self.lower_state_dim)
    
    def _extract_upper_state(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Extract a compact feature vector (upper state) from the environment observation for the UpperAgent.

        The upper state vector includes:
          - current simulation time,
          - the minimum available time among vehicles,
          - the minimum available time among drones,
          - the number of pending requests in the request buffer.
        The vector is padded with zeros to match the state dimension required by the UpperAgent.
        
        Args:
            observation (Dict[str, Any]): The current environment observation.
        
        Returns:
            torch.Tensor: A 1D tensor of shape (state_dim_upper,) representing the extracted upper state.
        """
        # Get current simulation time
        current_time = float(observation.get("current_time", [0])[0])
        
        # Extract vehicles and drones list from observation.
        vehicles: List[Dict[str, Any]] = observation.get("vehicles", [])
        drones: List[Dict[str, Any]] = observation.get("drones", [])
        
        # Compute minimum available times among vehicles and drones.
        vehicle_times = [float(v.get("available_time", 0.0)) for v in vehicles] if vehicles else [0.0]
        drone_times = [float(d.get("available_time", 0.0)) for d in drones] if drones else [0.0]
        min_vehicle_time = min(vehicle_times)
        min_drone_time = min(drone_times)
        
        # Count number of pending requests in request buffer.
        request_buffer: List[Any] = observation.get("request_buffer", [])
        request_count = float(len(request_buffer))
        
        # Form feature vector (list of floats)
        features: List[float] = [current_time, min_vehicle_time, min_drone_time, request_count]
        
        # Get desired upper state dimension from upper_agent. Default is from config "state_dim", default 10.
        state_dim_upper: int = getattr(self.upper_agent, "state_dim", 10)
        
        # Pad with zeros if needed.
        if len(features) < state_dim_upper:
            features.extend([0.0] * (state_dim_upper - len(features)))
        else:
            features = features[:state_dim_upper]
        
        upper_state = torch.tensor(features, dtype=torch.float32)
        return upper_state

    def _extract_lower_state(self, request: Dict[str, Any], observation: Dict[str, Any]) -> torch.Tensor:
        """
        Extract a feature vector (lower state) for a given customer request for the LowerAgent.
        
        The lower state vector includes:
          - the request's arrival time,
          - the Euclidean distance from the depot (assumed at (0,0)) to the request's location,
          - the minimum available time among vehicles,
          - the minimum available time among drones.
        The vector is padded with zeros to match the lower agent's required state dimension.
        
        Args:
            request (Dict[str, Any]): The customer request dictionary.
            observation (Dict[str, Any]): The current environment observation (to extract fleet info).
        
        Returns:
            torch.Tensor: A 1D tensor of shape (lower_state_dim,) representing the extracted lower state.
        """
        # Request arrival time
        arrival_time = float(request.get("arrival_time", 0.0))
        # Compute Euclidean distance from depot (0,0) to the request location.
        location = request.get("location", (0.0, 0.0))
        distance = compute_euclidean_distance((0.0, 0.0), location)
        
        # Extract available times from vehicles and drones in the observation.
        vehicles: List[Dict[str, Any]] = observation.get("vehicles", [])
        drones: List[Dict[str, Any]] = observation.get("drones", [])
        vehicle_times = [float(v.get("available_time", 0.0)) for v in vehicles] if vehicles else [0.0]
        drone_times = [float(d.get("available_time", 0.0)) for d in drones] if drones else [0.0]
        min_vehicle_time = min(vehicle_times)
        min_drone_time = min(drone_times)
        
        features: List[float] = [arrival_time, distance, min_vehicle_time, min_drone_time]
        
        # Pad to lower_state_dim (from config "state_dim_lower", default 10).
        if len(features) < self.lower_state_dim:
            features.extend([0.0] * (self.lower_state_dim - len(features)))
        else:
            features = features[:self.lower_state_dim]
        
        lower_state = torch.tensor(features, dtype=torch.float32)
        return lower_state

    def train(self, num_epochs: int) -> None:
        """
        Train the HDDRL model for a specified number of epochs (episodes).
        
        For each episode:
          1. Reset the environment.
          2. At each decision epoch, extract the upper state for the UpperAgent and determine the
             dispatch action (wait = 0 or dispatch = 1).
          3. For each customer request in the request buffer, extract a lower state and use the LowerAgent
             to decide assignment (assign to drone, assign to vehicle, or reject).
          4. Aggregate these lower assignment decisions into an action dictionary.
          5. If the dispatch decision is to dispatch (action = 1), then aggregate a shaped reward equal 
             to the lower agent rewards; otherwise, assign a reward of 0 for the upper agent.
          6. Call env.step() with the action dictionary and obtain the next state, reward, done flag, and info.
          7. Store the UpperAgent's experience in its replay buffer.
          8. Call update() methods for both UpperAgent and LowerAgent.
          9. Continue until the episode ends, then log performance metrics.
         10. Every evaluation_frequency epochs, perform an evaluation checkpoint (logging).
         11. After all epochs, save both agent models.
        
        Args:
            num_epochs (int): Total number of training episodes (epochs).
        """
        self.logger.info("Starting training for %d epochs.", num_epochs)
        
        for epoch in range(1, num_epochs + 1):
            state = self.env.reset()
            done = False
            episode_upper_reward: float = 0.0
            
            while not done:
                # Extract upper state from environment observation.
                upper_state = self._extract_upper_state(state)
                
                # UpperAgent selects dispatch action (0 = wait, 1 = dispatch)
                upper_action: int = self.upper_agent.select_action(upper_state)
                
                # Process lower agent assignments for each request in the current request buffer.
                lower_assignment_dict: Dict[Any, int] = {}
                request_buffer: List[Dict[str, Any]] = state.get("request_buffer", [])
                
                for request in request_buffer:
                    lower_state = self._extract_lower_state(request, state)
                    assignment_action: int = self.lower_agent.select_action(lower_state)
                    lower_assignment_dict[request["id"]] = assignment_action
                    # Lower agent experiences will be stored within lower_agent.select_action.
                    # Reward for each request will be later assigned via env.step().
                
                # Formulate the action dictionary for the environment.
                action_dict: Dict[str, Any] = {
                    "dispatch_decision": upper_action,
                    "assignment_decisions": lower_assignment_dict
                }
                
                # Step the environment.
                next_state, env_reward, done, info = self.env.step(action_dict)
                
                # Reward shaping for upper agent:
                # If the dispatch decision was to dispatch (1), use the env_reward (aggregated lower rewards),
                # otherwise, assign a reward of 0.
                shaped_upper_reward: float = env_reward if upper_action == 1 else 0.0
                episode_upper_reward += shaped_upper_reward
                
                # Extract next upper state.
                next_upper_state = self._extract_upper_state(next_state)
                
                # Store UpperAgent experience.
                self.upper_agent.store_transition(
                    upper_state, upper_action, shaped_upper_reward, next_upper_state, done
                )
                
                # (LowerAgent experiences already stored during action selection.
                #  Optionally, lower_agent.store_reward(reward, done) could be invoked if per-request reward is available.)
                
                # Call updates for agents.
                # Update UpperAgent (DQN) using its replay buffer.
                self.upper_agent.update(epoch)
                # Update LowerAgent (PPO) using its collected trajectory.
                self.lower_agent.update()
                
                # Advance state.
                state = next_state
            
            # End-of-episode metrics from environment info
            total_served: int = info.get("total_requests_served", 0)
            total_generated: int = info.get("total_requests_generated", 1)
            service_rate: float = info.get("service_rate", total_served / total_generated)
            metrics: Dict[str, float] = {
                "Episode": epoch,
                "Upper_Reward": episode_upper_reward,
                "Served": float(total_served),
                "Service_Rate": service_rate
            }
            self.logger.info("Episode %d: Upper Reward=%.2f, Served=%d, Service Rate=%.4f",
                             epoch, episode_upper_reward, total_served, service_rate)
            
            # Evaluation checkpoint
            if epoch % self.evaluation_frequency == 0:
                self.logger.info("Evaluation checkpoint at epoch %d", epoch)
                # Additional evaluation routines could be inserted here.
        
        # After training is complete, save models
        upper_model_path: str = self.config.get("model_save", {}).get("upper_agent", "upper_agent.pth")
        lower_model_path: str = self.config.get("model_save", {}).get("lower_agent", "lower_agent.pth")
        self.upper_agent.save_model(upper_model_path)
        self.lower_agent.save_model(lower_model_path)
        self.logger.info("Training complete. Models saved to: %s, %s", upper_model_path, lower_model_path)

if __name__ == "__main__":
    # Main block for standalone training execution.
    import yaml
    # Import necessary modules from project files.
    from dataset_loader import DatasetLoader
    from env import SD3RVDEnv
    from agent_upper import UpperAgent
    from agent_lower import LowerAgent
    from heuristics import Heuristics

    # Load configuration from config.yaml.
    config_path: str = "config.yaml"
    try:
        with open(config_path, "r") as config_file:
            config: Dict[str, Any] = yaml.safe_load(config_file)
    except Exception as error:
        config = {}
    
    # Initialize logger for the main trainer.
    main_logger = get_logger("TrainerMain")
    main_logger.info("Loading configuration and initializing components.")

    # Create the environment.
    env = SD3RVDEnv(config)

    # Instantiate the UpperAgent and LowerAgent with configuration.
    upper_agent = UpperAgent(config)
    lower_agent = LowerAgent(config)
    
    # Heuristics is used as a class for static methods.
    heuristics = Heuristics

    # Create the Trainer instance.
    trainer = Trainer(env, upper_agent, lower_agent, heuristics, config)

    # Determine the number of epochs from configuration; default to 400000 if not provided.
    num_epochs: int = int(config.get("training", {}).get("epochs", 400000))
    main_logger.info("Starting training for %d epochs.", num_epochs)

    # Start training.
    trainer.train(num_epochs)
