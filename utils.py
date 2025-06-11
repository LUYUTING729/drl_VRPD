# utils.py

import os
import random
import logging
import pickle
import datetime
import math
from typing import Any, Optional, Dict, List, Tuple

import yaml  # PyYAML required for reading configuration files
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    Args:
        file_path (str): Path to the config.yaml file.
    
    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.
    
    Raises:
        ValueError: If required configuration keys are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
    with open(file_path, "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    
    # Validate required keys in configuration
    required_keys = ["training", "environment", "fleet", "data", "evaluation"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Parse working hours. E.g., "10h" -> 600 minutes.
    if "working_hours" in config["environment"]:
        working_hours_str: str = str(config["environment"]["working_hours"])
        config["environment"]["working_minutes"] = parse_time_string(working_hours_str)
    else:
        # Default to 600 minutes if not specified.
        config["environment"]["working_minutes"] = 600

    # Additional parsing for request arrival times if provided as strings could be done here.
    # For now, we assume request_arrival.start_time and end_time are numeric.
    
    return config


def parse_time_string(time_str: str) -> int:
    """
    Convert a string time representation (e.g., '10h', '600m') into minutes.
    
    Args:
        time_str (str): The time string to convert.
    
    Returns:
        int: Time in minutes.
    
    Raises:
        ValueError: If the time format is invalid.
    """
    time_str = time_str.strip().lower()
    if time_str.endswith("h"):
        try:
            hours = float(time_str[:-1])
            return int(hours * 60)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")
    elif time_str.endswith("m"):
        try:
            minutes = float(time_str[:-1])
            return int(minutes)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")
    else:
        # Assume the time is given in minutes if no suffix is provided
        try:
            return int(time_str)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility in random, numpy, and PyTorch.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger that writes to the console and optionally to a file.
    
    Args:
        name (str): The name of the logger.
        log_file (Optional[str]): Path to a file to log messages (if provided).
    
    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear previously added handlers (if any)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(metrics: Dict[str, Any], epoch: int) -> None:
    """
    Log training metrics with epoch information.
    
    Args:
        metrics (Dict[str, Any]): A dictionary of metric names and their numeric values.
        epoch (int): The current epoch number.
    """
    logger = logging.getLogger("MetricsLogger")
    # Format each metric; if float, format to 4 decimal places.
    metric_str = ", ".join(
        [f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
         for key, value in metrics.items()]
    )
    logger.info(f"Epoch {epoch} - {metric_str}")


def calculate_service_rate(served: int, total: int) -> float:
    """
    Calculate the service rate as the ratio of served requests to total requests.
    
    Args:
        served (int): Number of served requests.
        total (int): Total number of requests.
    
    Returns:
        float: Service rate, or 0.0 if total is zero.
    """
    if total == 0:
        return 0.0
    return served / total


def calculate_improvement(new_value: float, baseline_value: float) -> float:
    """
    Calculate the improvement percentage.
    
    Args:
        new_value (float): The new performance metric value.
        baseline_value (float): The baseline performance metric value.
    
    Returns:
        float: Improvement percentage as a fraction (e.g., 0.1 for 10%). Returns infinity if baseline is zero.
    """
    if baseline_value == 0:
        return float("inf")
    return (new_value - baseline_value) / baseline_value


def aggregate_episode_metrics(episode_data: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate episode metrics by computing the mean of each metric.
    
    Args:
        episode_data (List[Dict[str, float]]): List of dictionaries containing metrics from each episode.
    
    Returns:
        Dict[str, float]: Dictionary of aggregated metrics.
    """
    aggregated: Dict[str, List[float]] = {}
    for data in episode_data:
        for key, value in data.items():
            aggregated.setdefault(key, []).append(value)
    # Compute average for each metric
    aggregated_mean: Dict[str, float] = {key: float(np.mean(values)) for key, values in aggregated.items()}
    return aggregated_mean


def compute_euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two 2D points.
    
    Args:
        point1 (Tuple[float, float]): Coordinates (x, y) of the first point.
        point2 (Tuple[float, float]): Coordinates (x, y) of the second point.
    
    Returns:
        float: Euclidean distance.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def compute_scaled_distance(point1: Tuple[float, float], point2: Tuple[float, float], scale: float) -> float:
    """
    Compute the scaled distance between two 2D points by applying a scaling factor.
    
    Args:
        point1 (Tuple[float, float]): Coordinates (x, y) of the first point.
        point2 (Tuple[float, float]): Coordinates (x, y) of the second point.
        scale (float): Scaling factor (e.g., route_distance_scale from config).
    
    Returns:
        float: Scaled distance.
    """
    distance = compute_euclidean_distance(point1, point2)
    return distance * scale


def minutes_to_hours(minutes: int) -> float:
    """
    Convert minutes into hours.
    
    Args:
        minutes (int): Time in minutes.
    
    Returns:
        float: Time in hours.
    """
    return minutes / 60.0


def get_current_timestamp() -> str:
    """
    Get the current timestamp as a formatted string.
    
    Returns:
        str: Timestamp in the format YYYYMMDD_HHMMSS.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def seed_worker(worker_id: int) -> None:
    """
    Set the seed for a worker to ensure reproducibility in multi-worker settings.
    
    Args:
        worker_id (int): The worker's unique identifier.
    """
    # Generate a worker-specific seed
    worker_seed = (torch.initial_seed() % 2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_pickle(data: Any, path: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data (Any): Python object to pickle.
        path (str): File path to save the pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        path (str): File path to load the pickle from.
    
    Returns:
        Any: The unpickled Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_training_curve(metric_list: List[float], title: str, xlabel: str, ylabel: str) -> None:
    """
    Plot a training curve using matplotlib.
    
    Args:
        metric_list (List[float]): List of metric values over epochs.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metric_list, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
