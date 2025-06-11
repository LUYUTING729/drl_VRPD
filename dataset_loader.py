"""
dataset_loader.py

This module implements the DatasetLoader class which is responsible for generating
simulated customer request datasets for the same©\day delivery dispatching and routing
problem with vehicles and drones (SD3RVD). The dataset is constructed based on a
combination of temporal distributions (homogeneous Poisson process or heterogeneous with peaks)
and spatial distributions (normal or power-law clustering). Each instance represents a day¡¯s
worth of requests.
"""

import os
import random
import math
from typing import Any, List, Dict, Tuple

import numpy as np
import yaml

from utils import get_logger  # Logger from utils.py

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DatasetLoader with the provided configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.logger = get_logger("DatasetLoader")

        # Extract environment configuration parameters
        env_config: Dict[str, Any] = config.get("environment", {})
        request_arrival_config: Dict[str, Any] = env_config.get("request_arrival", {})
        self.start_time: float = float(request_arrival_config.get("start_time", 0))
        self.end_time: float = float(request_arrival_config.get("end_time", 420))
        # Request deadline (in minutes after arrival), default value from config provided
        self.request_deadline: float = float(env_config.get("request_deadline", 240))

        # Extract request rate parameters
        request_rate_config: Dict[str, Any] = config.get("request_rate", {})
        self.homo_rate: float = float(request_rate_config.get("homogeneous", 1.2))
        self.homo_low_rate: float = float(request_rate_config.get("homogeneous_low", 0.5))
        # For this loader, for homogeneous temporal type we use homo_rate by default.

        # Define peak intervals for heterogeneous temporal distribution
        # Peak intervals chosen as [90, 150] and [270, 330] minutes.
        self.peak_intervals: List[Tuple[float, float]] = [(90.0, 150.0), (270.0, 330.0)]

        # Spatial distribution parameters
        self.normal_std: float = 3.0  # Standard deviation for "Normal" spatial distribution (km)
        self.powerlaw_num_clusters: int = 3
        self.powerlaw_cluster_weights: List[float] = [0.7, 0.15, 0.15]
        self.powerlaw_cluster_std: float = 1.0  # Standard deviation for requests within a cluster (km)
        self.cluster_center_min: float = -10.0
        self.cluster_center_max: float = 10.0

        # Set instance generation counts
        self.num_train_instances: int = 1000  # Default number of training instances per type
        eval_config: Dict[str, Any] = config.get("evaluation", {})
        self.num_test_instances: int = int(eval_config.get("test_instances_per_type", 100))

        # Retrieve instance types from configuration; default types provided if missing.
        data_config: Dict[str, Any] = config.get("data", {})
        self.instance_types: List[Dict[str, str]] = data_config.get(
            "instance_types",
            [
                {"temporal": "Homo", "spatial": "Normal"},
                {"temporal": "Heter", "spatial": "Normal"},
                {"temporal": "Homo", "spatial": "Power-L"},
                {"temporal": "Heter", "spatial": "Power-L"},
            ]
        )

        # Set a random seed for reproducibility; default seed 42 if not provided.
        seed: int = int(config.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)

        self.logger.info("DatasetLoader initialized with configuration.")

    def load_data(self, mode: str = "train") -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Generate a dataset containing simulated customer request instances.
        
        Args:
            mode (str): 'train' or 'test'. Determines the number of instances to generate per type.
        
        Returns:
            Dict[str, List[List[Dict[str, Any]]]]:
                A dictionary where keys are strings representing instance type combinations
                (e.g., "Homo_Normal") and each value is a list of instances.
                Each instance is a list of request dictionaries.
        """
        self.logger.info(f"Loading data in mode: {mode}")
        dataset: Dict[str, List[List[Dict[str, Any]]]] = {}
        num_instances: int = self.num_train if mode.lower() == "train" else self.num_test

        # Loop through each instance type defined in the configuration
        for instance_type in self.instance_types:
            temporal_type: str = instance_type.get("temporal", "Homo")
            spatial_type: str = instance_type.get("spatial", "Normal")
            key: str = f"{temporal_type}_{spatial_type}"
            dataset[key] = []
            self.logger.info(f"Generating {num_instances} instances for type {key}")

            for instance_index in range(num_instances):
                # Generate arrival times based on temporal distribution type
                if temporal_type.strip().lower() == "homo":
                    arrival_times: List[float] = self._generate_homo_arrivals()
                elif temporal_type.strip().lower() == "heter":
                    arrival_times = self._generate_heter_arrivals()
                else:
                    self.logger.warning(f"Unrecognized temporal type '{temporal_type}'. Defaulting to homogeneous.")
                    arrival_times = self._generate_homo_arrivals()

                # Generate instance requests using the selected spatial distribution
                instance_requests: List[Dict[str, Any]] = self._generate_instance_requests(arrival_times, spatial_type)
                dataset[key].append(instance_requests)

            self.logger.info(f"Instance type '{key}': Generated {len(dataset[key])} instances.")
        return dataset

    def _generate_homo_arrivals(self) -> List[float]:
        """
        Generate arrival times using a homogeneous Poisson process.
        
        Returns:
            List[float]: Sorted list of arrival times (in minutes).
        """
        arrivals: List[float] = []
        current_time: float = self.start_time
        rate: float = self.homo_rate  # Using the homogeneous rate from config

        while current_time < self.end_time:
            # Sample interarrival time using exponential distribution (mean = 1/rate)
            interarrival: float = np.random.exponential(scale=1.0/rate)
            current_time += interarrival
            if current_time < self.end_time:
                arrivals.append(current_time)
        arrivals.sort()
        return arrivals

    def _generate_heter_arrivals(self) -> List[float]:
        """
        Generate arrival times using a heterogeneous process with two peak intervals.
        Outside the peaks, the base rate is used; during the peak intervals, the rate is doubled.
        
        Returns:
            List[float]: Sorted list of arrival times (in minutes).
        """
        segments: List[Tuple[float, float, float]] = []
        # Define peak interval boundaries
        peak1_start, peak1_end = self.peak_intervals[0]
        peak2_start, peak2_end = self.peak_intervals[1]

        # Define segments: (segment_start, segment_end, rate)
        segments.append((self.start_time, peak1_start, self.homo_rate))
        segments.append((peak1_start, peak1_end, 2 * self.homo_rate))
        segments.append((peak1_end, peak2_start, self.homo_rate))
        segments.append((peak2_start, peak2_end, 2 * self.homo_rate))
        segments.append((peak2_end, self.end_time, self.homo_rate))

        arrivals: List[float] = []
        for (seg_start, seg_end, rate) in segments:
            segment_length: float = seg_end - seg_start
            # Determine expected number of arrivals in this segment
            expected_arrivals: float = rate * segment_length
            # Sample actual count from a Poisson distribution
            count: int = np.random.poisson(expected_arrivals)
            if count > 0:
                # Sample arrival times uniformly within the segment
                seg_arrivals: List[float] = list(np.random.uniform(low=seg_start, high=seg_end, size=count))
                arrivals.extend(seg_arrivals)
        arrivals.sort()

        # Filter to ensure arrival times lie within [start_time, end_time)
        arrivals = [t for t in arrivals if self.start_time <= t < self.end_time]
        return arrivals

    def _generate_instance_requests(self, arrival_times: List[float], spatial_type: str) -> List[Dict[str, Any]]:
        """
        Generate request dictionaries for a single simulation instance.
        
        Each request dictionary contains:
            - "id": Unique integer identifier within the instance.
            - "arrival_time": Time (in minutes) when the request arrives.
            - "deadline": Arrival time plus the request_deadline.
            - "location": Tuple (x, y) representing customer location.
        
        Args:
            arrival_times (List[float]): List of arrival times.
            spatial_type (str): Spatial distribution type: "Normal" or "Power-L".
        
        Returns:
            List[Dict[str, Any]]: List of request dictionaries.
        """
        requests: List[Dict[str, Any]] = []

        # For power-law spatial distribution, generate cluster centers once per instance.
        clusters: List[Tuple[float, float]] = []
        if spatial_type.strip().lower() == "power-l":
            clusters = self._generate_powerlaw_clusters()

        for req_id, arrival_time in enumerate(arrival_times):
            deadline: float = arrival_time + self.request_deadline
            # Generate location based on spatial type
            if spatial_type.strip().lower() == "normal":
                location: Tuple[float, float] = self._generate_normal_location()
            elif spatial_type.strip().lower() == "power-l":
                location = self._generate_powerlaw_location(clusters)
            else:
                self.logger.warning(f"Unrecognized spatial type '{spatial_type}'. Defaulting to normal distribution.")
                location = self._generate_normal_location()

            request: Dict[str, Any] = {
                "id": req_id,
                "arrival_time": arrival_time,
                "deadline": deadline,
                "location": location
            }
            requests.append(request)
        return requests

    def _generate_normal_location(self) -> Tuple[float, float]:
        """
        Generate a customer location from a bivariate normal distribution centered at (0,0)
        with standard deviation given by self.normal_std.
        
        Returns:
            Tuple[float, float]: (x, y) location.
        """
        x: float = np.random.normal(loc=0.0, scale=self.normal_std)
        y: float = np.random.normal(loc=0.0, scale=self.normal_std)
        return (x, y)

    def _generate_powerlaw_clusters(self) -> List[Tuple[float, float]]:
        """
        Generate cluster centers for the power-law spatial distribution.
        Cluster centers are sampled uniformly from the range defined by cluster_center_min and cluster_center_max.
        
        Returns:
            List[Tuple[float, float]]: List of (x, y) cluster center coordinates.
        """
        clusters: List[Tuple[float, float]] = []
        for _ in range(self.powerlaw_num_clusters):
            x_center: float = np.random.uniform(low=self.cluster_center_min, high=self.cluster_center_max)
            y_center: float = np.random.uniform(low=self.cluster_center_min, high=self.cluster_center_max)
            clusters.append((x_center, y_center))
        return clusters

    def _generate_powerlaw_location(self, clusters: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Generate a customer location for a power-law spatial distribution.
        A cluster is chosen based on predefined weights, and the location is sampled
        from a bivariate normal distribution centered at the chosen cluster.
        
        Args:
            clusters (List[Tuple[float, float]]): Pre-generated list of cluster centers.
        
        Returns:
            Tuple[float, float]: (x, y) location.
        """
        cluster_index: int = int(np.random.choice(len(clusters), p=self.powerlaw_cluster_weights))
        center: Tuple[float, float] = clusters[cluster_index]
        x: float = np.random.normal(loc=center[0], scale=self.powerlaw_cluster_std)
        y: float = np.random.normal(loc=center[1], scale=self.powerlaw_cluster_std)
        return (x, y)


if __name__ == "__main__":
    # For testing purposes: Load configuration and generate sample dataset
    config_path: str = "config.yaml"
    try:
        with open(config_path, "r") as config_file:
            config_data: Dict[str, Any] = yaml.safe_load(config_file)
    except Exception as error:
        config_data = {}
    
    loader = DatasetLoader(config=config_data)
    # Generate training dataset
    dataset_train = loader.load_data(mode="train")
    # Log a summary of the generated dataset
    main_logger = get_logger("DatasetLoaderMain")
    for key, instances in dataset_train.items():
        if instances:
            instance_count: int = len(instances)
            first_instance_req_count: int = len(instances[0])
        else:
            instance_count = 0
            first_instance_req_count = 0
        main_logger.info(f"Type: {key}, Instances: {instance_count}, "
                         f"First instance request count: {first_instance_req_count}")
