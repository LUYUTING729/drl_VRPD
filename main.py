"""
main.py

Entry point for reproducing the HDDRL experiments for the same©\day delivery dispatching 
and routing problem with vehicles and drones (SD3RVD).

This file loads the configuration from config.yaml, sets up the dataset (via DatasetLoader), 
instantiates the simulation environment (SD3RVDEnv), creates the UpperAgent (DQN-based) and 
LowerAgent (PPO-based), and sets up the Heuristics module. It then instantiates the Trainer 
to orchestrate training for a configured number of epochs, and finally performs evaluation 
using the Evaluation module, reporting key metrics (SN, SR, etc.).

All default values and hyperparameters are read from the configuration file, with fallbacks 
to default values if necessary.
"""

import os
import sys
import yaml
import random
import numpy as np
import torch

from utils import load_config, set_seed, get_logger
from dataset_loader import DatasetLoader
from env import SD3RVDEnv
from agent_upper import UpperAgent
from agent_lower import LowerAgent
from heuristics import Heuristics
from trainer import Trainer
from evaluation import Evaluation

def main() -> None:
    # Load configuration from config.yaml
    config_path: str = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found. Exiting.")
        sys.exit(1)
    try:
        config = load_config(config_path)
    except Exception as error:
        print(f"Error loading configuration: {error}")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    seed_value: int = int(config.get("seed", 42))
    set_seed(seed_value)
    
    # Initialize logger
    logger = get_logger("Main")
    logger.info("Configuration loaded successfully from %s", config_path)
    
    # Data Preparation (optional): Generate simulated dataset via DatasetLoader.
    dataset_loader = DatasetLoader(config)
    dataset_train = dataset_loader.load_data(mode="train")
    logger.info("Training dataset generated with instance types: %s", list(dataset_train.keys()))
    
    # Environment Initialization
    env = SD3RVDEnv(config)
    logger.info("SD3RVD Environment initialized.")
    
    # Instantiate RL Agents
    upper_agent = UpperAgent(config)
    lower_agent = LowerAgent(config)
    logger.info("UpperAgent and LowerAgent initialized.")
    
    # Heuristics Module (static class, used directly)
    heuristics = Heuristics  # No instantiation needed; using static methods
    
    # Instantiate Trainer with environment, agents, heuristics, and config
    trainer = Trainer(env, upper_agent, lower_agent, heuristics, config)
    logger.info("Trainer initialized successfully.")
    
    # Extract training parameters from configuration
    training_config = config.get("training", {})
    num_epochs: int = int(training_config.get("epochs", 400000))
    logger.info("Starting training for %d epochs.", num_epochs)
    
    # Run training loop
    trainer.train(num_epochs)
    logger.info("Training completed.")
    
    # Evaluation Phase: instantiate Evaluation and run test episodes.
    evaluator = Evaluation(env, upper_agent, lower_agent, config)
    logger.info("Starting evaluation of the trained agents.")
    evaluation_metrics = evaluator.evaluate()
    
    logger.info("Evaluation Metrics: %s", evaluation_metrics)
    
    # Print final evaluation summary to console
    print("Evaluation Complete:")
    print(f"Average Served Number (SN): {evaluation_metrics.get('average_served_number', 0):.2f}")
    print(f"Overall Service Rate (SR): {evaluation_metrics.get('average_service_rate', 0):.4f}")

if __name__ == "__main__":
    main()
