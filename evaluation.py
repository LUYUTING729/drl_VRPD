"""
evaluation.py

This module defines the Evaluation class that conducts testing of the HDDRL framework
for the same©\day delivery dispatching and routing problem (SD3RVD).

The Evaluation class runs test episodes using the provided environment (SD3RVDEnv),
UpperAgent (DQN-based dispatch agent) and LowerAgent (PPO-based assignment agent) in a
deterministic inference mode and aggregates performance metrics including:
    - Average Served Number (SN)
    - Average Service Rate (SR)

The configuration parameters (e.g., number of test episodes) are obtained from the
config.yaml file via the configuration dictionary.
"""

import os
import random
import math
from typing import Any, Dict, List, Tuple

import torch
import numpy as np

from utils import get_logger, load_config, compute_euclidean_distance
from env import SD3RVDEnv
from agent_upper import UpperAgent
from agent_lower import LowerAgent

class Evaluation:
    """
    Evaluation conducts test episodes on the SD3RVD environment using the trained agents.
    
    Methods:
        __init__(env, upper, lower, config): Initializes with the environment, agents, and config.
        evaluate() -> Dict[str, Any]: Runs the test episodes and returns aggregated metrics.
    """
    
    def __init__(self, env: SD3RVDEnv, upper_agent: UpperAgent, lower_agent: LowerAgent,
                 config: Dict[str, Any]) -> None:
        """
        Initialize the Evaluation instance.
        
        Args:
            env (SD3RVDEnv): Simulation environment instance.
            upper_agent (UpperAgent): Upper agent (DQN-based) for dispatch decisions.
            lower_agent (LowerAgent): Lower agent (PPO-based) for request assignment.
            config (Dict[str, Any]): Configuration dictionary from config.yaml.
        """
        self.env = env
        self.upper_agent = upper_agent
        self.lower_agent = lower_agent
        self.config = config
        
        # Set number of test episodes per instance type; default to 100 if not specified.
        evaluation_conf: Dict[str, Any] = config.get("evaluation", {})
        self.num_test_episodes: int = int(evaluation_conf.get("test_instances_per_type", 100))
        
        # Determine state dimensions for upper and lower agents from config defaults.
        # Upper state dimension: use upper_agent.state_dim if available, else default to 10.
        self.upper_state_dim: int = getattr(self.upper_agent, "state_dim", 10)
        # Lower state dimension: from config "state_dim_lower", default to 10.
        self.lower_state_dim: int = int(config.get("state_dim_lower", 10))
        
        self.logger = get_logger("Evaluation")
        
        # Set the agents to evaluation mode (deterministic inference).
        self.upper_agent.q_network.eval()
        self.lower_agent.actor.eval()
        self.lower_agent.critic.eval()
        
        self.logger.info("Evaluation initialized: num_test_episodes=%d, upper_state_dim=%d, lower_state_dim=%d",
                         self.num_test_episodes, self.upper_state_dim, self.lower_state_dim)
    
    def _extract_upper_state(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Extract a compact upper state feature vector from the environment observation.
        
        The feature vector consists of:
            - current simulation time (in minutes)
            - minimum available time among vehicles
            - minimum available time among drones
            - number of pending requests in the request buffer
        
        The vector is padded with zeros to match self.upper_state_dim.
        
        Args:
            observation (Dict[str, Any]): The current state observation.
            
        Returns:
            torch.Tensor: 1D tensor of shape (upper_state_dim,)
        """
        current_time = float(observation.get("current_time", [0.0])[0])
        
        vehicles: List[Dict[str, Any]] = observation.get("vehicles", [])
        drones: List[Dict[str, Any]] = observation.get("drones", [])
        
        # Get available times; if empty, default to 0
        vehicle_times = [float(v.get("available_time", 0.0)) for v in vehicles] if vehicles else [0.0]
        drone_times   = [float(d.get("available_time", 0.0)) for d in drones] if drones else [0.0]
        min_vehicle_time = min(vehicle_times)
        min_drone_time   = min(drone_times)
        
        request_buffer: List[Any] = observation.get("request_buffer", [])
        request_count = float(len(request_buffer))
        
        features: List[float] = [current_time, min_vehicle_time, min_drone_time, request_count]
        
        # Pad the feature vector to the required upper state dimension.
        if len(features) < self.upper_state_dim:
            features.extend([0.0] * (self.upper_state_dim - len(features)))
        else:
            features = features[:self.upper_state_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_lower_state(self, request: Dict[str, Any], observation: Dict[str, Any]) -> torch.Tensor:
        """
        Extract a lower state feature vector from a customer request and observation.
        
        The feature vector consists of:
            - request arrival time
            - Euclidean distance from depot (assumed at (0,0)) to request location
            - minimum available time among vehicles
            - minimum available time among drones
        
        The vector is padded with zeros to match self.lower_state_dim.
        
        Args:
            request (Dict[str, Any]): Customer request dictionary.
            observation (Dict[str, Any]): Current environment observation (for fleet info).
            
        Returns:
            torch.Tensor: 1D tensor of shape (lower_state_dim,)
        """
        arrival_time = float(request.get("arrival_time", 0.0))
        location = request.get("location", (0.0, 0.0))
        distance = compute_euclidean_distance((0.0, 0.0), location)
        
        vehicles: List[Dict[str, Any]] = observation.get("vehicles", [])
        drones: List[Dict[str, Any]] = observation.get("drones", [])
        vehicle_times = [float(v.get("available_time", 0.0)) for v in vehicles] if vehicles else [0.0]
        drone_times   = [float(d.get("available_time", 0.0)) for d in drones] if drones else [0.0]
        min_vehicle_time = min(vehicle_times)
        min_drone_time   = min(drone_times)
        
        features: List[float] = [arrival_time, distance, min_vehicle_time, min_drone_time]
        if len(features) < self.lower_state_dim:
            features.extend([0.0] * (self.lower_state_dim - len(features)))
        else:
            features = features[:self.lower_state_dim]
        return torch.tensor(features, dtype=torch.float32)
    
    def _deterministic_upper_action(self, upper_state: torch.Tensor) -> int:
        """
        Select an upper agent action deterministically (greedy action) using the Q-network.
        
        Args:
            upper_state (torch.Tensor): Upper state tensor of shape (upper_state_dim,).
            
        Returns:
            int: Dispatch action (0 for wait, 1 for dispatch).
        """
        # Ensure no random sampling; use argmax of Q-values.
        with torch.no_grad():
            q_values = self.upper_agent.q_network(upper_state.unsqueeze(0))  # shape (1, num_actions)
        action = int(torch.argmax(q_values, dim=1).item())
        return action
    
    def _deterministic_lower_action(self, lower_state: torch.Tensor) -> int:
        """
        Select a lower agent assignment action deterministically using the actor network.
        
        Args:
            lower_state (torch.Tensor): Lower state tensor of shape (lower_state_dim,).
            
        Returns:
            int: Assignment action (0: assign to drone, 1: assign to vehicle, 2: reject).
        """
        with torch.no_grad():
            action_probs = self.lower_agent.actor(lower_state.unsqueeze(0)).squeeze(0)  # shape (action_dim,)
        # Deterministic choice: use argmax of the probabilities.
        action = int(torch.argmax(action_probs, dim=0).item())
        return action
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained HDDRL model on a set of test episodes.
        
        For each episode, the environment is reset and the simulation runs until termination.
        At each decision epoch, the upper agent (dispatch decision) and the lower agent (assignment
        decisions for pending requests) select actions deterministically. The chosen actions are
        combined into an action dictionary passed to the environment's step() method.
        
        The method aggregates metrics:
            - Total served requests per episode.
            - Total generated requests per episode.
            - Overall service rate (total served / total generated).
            - Average served number (SN) across episodes.
        
        Returns:
            Dict[str, Any]: Aggregated evaluation metrics.
                Example:
                {
                    "average_served_number": <value>,
                    "average_service_rate": <value>,
                    "episode_metrics": [
                        {"episode": 1, "served": X1, "total": Y1, "service_rate": Z1},
                        ...
                    ]
                }
        """
        self.logger.info("Starting evaluation over %d episodes.", self.num_test_episodes)
        
        episode_metrics: List[Dict[str, Any]] = []
        total_served = 0
        total_generated = 0
        
        for ep in range(1, self.num_test_episodes + 1):
            state = self.env.reset()
            done = False
            # Loop until episode terminates.
            while not done:
                # Extract upper state from observation.
                upper_state = self._extract_upper_state(state)
                # Use deterministic action: greedily choose dispatch action.
                dispatch_action = self._deterministic_upper_action(upper_state)
                
                # For each request in the request buffer, use lower agent to decide assignment.
                assignment_decisions: Dict[Any, int] = {}
                request_buffer: List[Dict[str, Any]] = state.get("request_buffer", [])
                for req in request_buffer:
                    lower_state = self._extract_lower_state(req, state)
                    action_assignment = self._deterministic_lower_action(lower_state)
                    assignment_decisions[req["id"]] = action_assignment
                
                # Form the action dictionary expected by the environment.
                action_dict: Dict[str, Any] = {
                    "dispatch_decision": dispatch_action,
                    "assignment_decisions": assignment_decisions
                }
                
                # Take a step in the environment using the deterministic action.
                next_state, reward, done, info = self.env.step(action_dict)
                # Update state for next step.
                state = next_state
            
            # End of episode: retrieve cumulative metrics from info.
            # The environment info contains "total_requests_served" and "total_requests_generated".
            ep_served: int = int(info.get("total_requests_served", 0))
            ep_total: int = int(info.get("total_requests_generated", 1))
            ep_service_rate: float = float(info.get("service_rate", ep_served / ep_total))
            
            episode_metrics.append({
                "episode": ep,
                "served": ep_served,
                "total": ep_total,
                "service_rate": ep_service_rate
            })
            total_served += ep_served
            total_generated += ep_total
            self.logger.info("Episode %d: Served=%d, Total=%d, Service Rate=%.4f",
                             ep, ep_served, ep_total, ep_service_rate)
        
        average_served_number = total_served / self.num_test_episodes if self.num_test_episodes > 0 else 0.0
        overall_service_rate = total_served / total_generated if total_generated > 0 else 0.0
        
        metrics: Dict[str, Any] = {
            "average_served_number": average_served_number,
            "average_service_rate": overall_service_rate,
            "episode_metrics": episode_metrics
        }
        
        self.logger.info("Evaluation complete: Avg Served Number = %.2f, Overall Service Rate = %.4f",
                         average_served_number, overall_service_rate)
        return metrics


if __name__ == "__main__":
    # Main block to perform evaluation.
    import yaml
    import sys

    # Load configuration from config.yaml
    config_path: str = "config.yaml"
    if not os.path.exists(config_path):
        sys.exit(f"Error: Configuration file {config_path} not found.")
    
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    
    # Initialize logger for evaluation main.
    main_logger = get_logger("EvaluationMain")
    main_logger.info("Loading configuration and initializing evaluation components.")
    
    # Instantiate environment
    env = SD3RVDEnv(config)
    # Initialize UpperAgent and LowerAgent using the configuration.
    upper_agent = UpperAgent(config)
    lower_agent = LowerAgent(config)
    
    # Create Evaluation instance.
    evaluator = Evaluation(env, upper_agent, lower_agent, config)
    
    # Run evaluation.
    metrics = evaluator.evaluate()
    
    main_logger.info("Evaluation Metrics: %s", metrics)
    
    # Optionally, plot metrics using matplotlib.
    # Example: Plot service rate across episodes.
    try:
        import matplotlib.pyplot as plt
        episodes = [m["episode"] for m in metrics["episode_metrics"]]
        service_rates = [m["service_rate"] for m in metrics["episode_metrics"]]
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, service_rates, marker='o', linestyle='-', color='b')
        plt.title("Service Rate per Evaluation Episode")
        plt.xlabel("Episode")
        plt.ylabel("Service Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        main_logger.warning("matplotlib not available; skipping plot.")
