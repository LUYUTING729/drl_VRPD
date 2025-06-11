"""
env.py

This module defines the SD3RVDEnv class that simulates the same©\day delivery
dispatching and routing problem with vehicles and drones (SD3RVD). It implements
the Gym environment API methods: reset(), step(action), and render().

The environment simulates:
  - Dynamic customer request arrivals (using homogeneous or heterogeneous temporal distributions).
  - Spatial request generation (¡°Normal¡± or ¡°Power-L¡± distributions).
  - A fleet of vehicles and drones starting at the depot.
  - Decision epochs triggered by new request arrivals or depot events.
  
When step(action) is called, the environment processes:
  - The upper agent decision (dispatch decision: wait or dispatch a vehicle).
  - The lower agent assignment decisions for each request in the request buffer
    (assign to drone, assign to vehicle, or reject).

It uses external heuristic functions (from heuristics.py) for:
  - Updating vehicle routes via the cheapest_insertion heuristic.
  - Scheduling drone assignments using a FIFO heuristic.

Configuration parameters are taken from the config.yaml file via the config dictionary.
"""

import gym
from gym import spaces
import numpy as np
import random
import math
from typing import Any, Dict, List, Tuple

# Import utilities for logging, parsing time strings, and distance computations.
from utils import get_logger, parse_time_string, compute_euclidean_distance, compute_scaled_distance

# Import Heuristics module (assumed to be implemented in heuristics.py)
from heuristics import Heuristics

# Constants for lower agent assignment decisions.
ASSIGN_DRONE: int = 0
ASSIGN_VEHICLE: int = 1
REJECT: int = 2

class SD3RVDEnv(gym.Env):
    """
    SD3RVDEnv simulates the same©\day delivery problem with heterogeneous fleets.
    It conforms to the Gym API.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SD3RVD environment using settings from the config.
        """
        super(SD3RVDEnv, self).__init__()
        self.config: Dict[str, Any] = config
        self.logger = get_logger("SD3RVDEnv")

        # Parse environment configuration parameters
        env_conf: Dict[str, Any] = config.get("environment", {})
        self.vehicle_speed: float = float(env_conf.get("vehicle_speed", 30))  # km/h
        self.drone_speed: float = float(env_conf.get("drone_speed", 40))  # km/h
        self.vehicle_loading_time: float = float(env_conf.get("vehicle_loading_time", 5))  # minutes
        self.drone_loading_time: float = float(env_conf.get("drone_loading_time", 5))  # minutes
        self.drop_off_time: float = float(env_conf.get("drop_off_time", 5))  # minutes
        self.drone_charging_time: float = float(env_conf.get("drone_charging_time", 20))  # minutes
        self.grid_block_time: float = float(env_conf.get("grid_block_time", 5))  # minutes per block
        self.route_distance_scale: float = float(env_conf.get("route_distance_scale", 1.5))
        self.request_deadline: float = float(env_conf.get("request_deadline", 240))  # minutes

        # Request arrival time window
        req_arrival_conf: Dict[str, Any] = env_conf.get("request_arrival", {})
        self.request_start_time: float = float(req_arrival_conf.get("start_time", 0))
        self.request_end_time: float = float(req_arrival_conf.get("end_time", 420))

        # Determine working hours in minutes
        if "working_minutes" in env_conf:
            self.working_minutes: float = float(env_conf["working_minutes"])
        else:
            self.working_minutes = float(parse_time_string(env_conf.get("working_hours", "10h")))  # default 600 min

        # Waiting interval ¦¤t; default to grid_block_time
        self.wait_interval: float = self.grid_block_time

        # Fleet configuration
        fleet_conf: Dict[str, Any] = config.get("fleet", {})
        self.num_vehicles: int = int(fleet_conf.get("vehicles", 2))
        self.num_drones: int = int(fleet_conf.get("drones", 3))

        # Data configuration: choose instance type (temporal & spatial) for event generation.
        data_conf: Dict[str, Any] = config.get("data", {})
        instance_types: List[Dict[str, str]] = data_conf.get("instance_types", [{"temporal": "Homo", "spatial": "Normal"}])
        self.temporal_type: str = instance_types[0].get("temporal", "Homo")
        self.spatial_type: str = instance_types[0].get("spatial", "Normal")

        # Request rate parameters
        req_rate_conf: Dict[str, Any] = config.get("request_rate", {})
        self.homo_rate: float = float(req_rate_conf.get("homogeneous", 1.2))
        self.homo_low_rate: float = float(req_rate_conf.get("homogeneous_low", 0.5))
        # Define peak intervals for heterogeneous temporal distribution (in minutes)
        self.peak_intervals: List[Tuple[float, float]] = [(90.0, 150.0), (270.0, 330.0)]

        # Spatial distribution parameters
        self.normal_std: float = 3.0  # km, for Normal spatial distribution
        self.powerlaw_num_clusters: int = 3
        self.powerlaw_cluster_weights: List[float] = [0.7, 0.15, 0.15]
        self.powerlaw_cluster_std: float = 1.0  # km for cluster spread
        self.cluster_center_min: float = -10.0
        self.cluster_center_max: float = 10.0

        # Initialize simulation clock and counters
        self.current_time: float = 0.0  # minutes, starting at 0 (8am)
        self.total_requests_generated: int = 0
        self.total_requests_served: int = 0

        # Initialize the fleet: vehicles and drones start at depot (position (0,0))
        self.vehicles: List[Dict[str, Any]] = []
        for vid in range(self.num_vehicles):
            vehicle: Dict[str, Any] = {
                "id": vid,
                "status": "at_depot",  # "at_depot" or "in_transit"
                "position": (0.0, 0.0),
                "route": [{"location": (0.0, 0.0), "time": 0.0}],
                "available_time": 0.0
            }
            self.vehicles.append(vehicle)

        self.drones: List[Dict[str, Any]] = []
        for did in range(self.num_drones):
            drone: Dict[str, Any] = {
                "id": did,
                "status": "at_depot",  # "at_depot" or "in_transit"
                "position": (0.0, 0.0),
                "route": [{"location": (0.0, 0.0), "time": 0.0}],
                "available_time": 0.0
            }
            self.drones.append(drone)

        # Request buffer collects customer requests that have arrived but not yet processed.
        self.request_buffer: List[Dict[str, Any]] = []

        # Event list: will hold the pre-generated customer requests for the day.
        self.event_list: List[Dict[str, Any]] = []

        # Define Gym action and observation spaces (placeholders for now)
        self.action_space = spaces.Dict({
            "dispatch_decision": spaces.Discrete(2),  # 0: wait, 1: dispatch
            "assignment_decisions": spaces.Dict({})   # A dict mapping request ids to decisions
        })
        self.observation_space = spaces.Dict({
            "current_time": spaces.Box(low=0, high=self.working_minutes, shape=(1,), dtype=np.float32),
            "vehicles": spaces.Dict({}),
            "drones": spaces.Dict({}),
            "request_buffer": spaces.Dict({})
        })

        self.logger.info("SD3RVDEnv initialized: working_minutes=%s, num_vehicles=%d, num_drones=%d",
                         self.working_minutes, self.num_vehicles, self.num_drones)

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Returns:
            Dict[str, Any]: The initial observation containing current_time, fleet status, and request buffer.
        """
        self.logger.info("Resetting environment.")
        self.current_time = 0.0

        # Reset vehicle fleet
        for vehicle in self.vehicles:
            vehicle["status"] = "at_depot"
            vehicle["position"] = (0.0, 0.0)
            vehicle["route"] = [{"location": (0.0, 0.0), "time": 0.0}]
            vehicle["available_time"] = 0.0

        # Reset drone fleet
        for drone in self.drones:
            drone["status"] = "at_depot"
            drone["position"] = (0.0, 0.0)
            drone["route"] = [{"location": (0.0, 0.0), "time": 0.0}]
            drone["available_time"] = 0.0

        # Clear request buffer and event list, then pre-generate event list for the day.
        self.request_buffer = []
        self.event_list = self._generate_request_events()
        self.total_requests_generated = len(self.event_list)
        self.total_requests_served = 0

        # Add any events that have already arrived (arrival_time <= current_time)
        while self.event_list and self.event_list[0]["arrival_time"] <= self.current_time:
            event = self.event_list.pop(0)
            self.request_buffer.append(event)

        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Advance the simulation by one decision epoch based on the provided action.
        
        Args:
            action (Dict[str, Any]): Action dictionary with keys:
                - "dispatch_decision": int (0 for wait, 1 for dispatch)
                - "assignment_decisions": Dict mapping request id to decision integer (0: drone, 1: vehicle, 2: reject)
        
        Returns:
            Tuple containing:
                - observation (Dict[str, Any]): Updated state.
                - reward (float): Aggregated reward from this step.
                - done (bool): True if episode terminated.
                - info (Dict[str, Any]): Additional info such as cumulative metrics.
        """
        reward: float = 0.0
        info: Dict[str, Any] = {}

        # Extract action details
        dispatch_decision: int = int(action.get("dispatch_decision", 0))
        assignment_decisions: Dict[Any, int] = action.get("assignment_decisions", {})

        # Process lower agent assignments for requests in the request buffer.
        vehicle_requests: List[Dict[str, Any]] = []
        buffer_requests_copy: List[Dict[str, Any]] = self.request_buffer.copy()
        for req in buffer_requests_copy:
            req_id = req.get("id")
            decision: int = assignment_decisions.get(req_id, REJECT)
            if decision == ASSIGN_DRONE:
                # Process drone assignment
                assigned: bool = False
                for drone in self.drones:
                    if drone["status"] == "at_depot" and self.current_time >= drone["available_time"]:
                        # Compute travel time from depot (0,0) to customer location.
                        distance: float = compute_euclidean_distance((0.0, 0.0), req["location"])
                        travel_time: float = (distance / self.drone_speed) * 60.0  # minutes
                        service_time: float = self.drone_loading_time + travel_time + self.drop_off_time + self.drone_charging_time
                        finish_time: float = self.current_time + service_time
                        if finish_time <= req["deadline"]:
                            reward += 1.0
                            self.total_requests_served += 1
                        else:
                            reward += -5.0
                        # Update drone assignment using FIFO heuristic
                        # Here we simulate the heuristic update.
                        updated_route = [
                            {"location": (0.0, 0.0), "time": self.current_time},
                            {"location": req["location"], "time": self.current_time + travel_time},
                            {"location": (0.0, 0.0), "time": finish_time}
                        ]
                        drone["route"] = updated_route
                        drone["status"] = "in_transit"
                        drone["available_time"] = finish_time
                        drone["position"] = req["location"]
                        assigned = True
                        break
                if not assigned:
                    # No drone available; penalize.
                    reward += -3.0
                # Remove the request from the buffer
                self.request_buffer.remove(req)
            elif decision == REJECT:
                reward += -3.0
                self.request_buffer.remove(req)
            elif decision == ASSIGN_VEHICLE:
                # For vehicle assignments, postpone scheduling until dispatch decision.
                vehicle_requests.append(req)
                # Do not remove from buffer yet.
            else:
                # Unrecognized decision; treat as reject.
                reward += -3.0
                self.request_buffer.remove(req)

        # Process upper agent (dispatch decision)
        if dispatch_decision == 1:
            # Dispatch: choose an available vehicle at depot.
            available_vehicle = None
            for vehicle in self.vehicles:
                if vehicle["status"] == "at_depot" and self.current_time >= vehicle["available_time"]:
                    available_vehicle = vehicle
                    break
            if available_vehicle is not None:
                # Process vehicle assignments using the cheapest insertion heuristic.
                for req in vehicle_requests:
                    # Call heuristic to insert request into vehicle's route
                    feasible, insertion_cost, new_route = Heuristics.cheapest_insertion(available_vehicle["route"], req, self.current_time)
                    if feasible:
                        reward += 1.0
                        self.total_requests_served += 1
                        available_vehicle["route"] = new_route
                    else:
                        reward += -5.0
                    # Remove processed request from buffer
                    if req in self.request_buffer:
                        self.request_buffer.remove(req)
                # Mark vehicle as dispatched (in_transit) and update its availability.
                available_vehicle["status"] = "in_transit"
                departure_time: float = self.current_time + self.vehicle_loading_time
                # For simplicity, assume the vehicle's route last time indicates finish time.
                if available_vehicle["route"]:
                    last_stop = available_vehicle["route"][-1]
                    available_vehicle["available_time"] = max(departure_time, last_stop.get("time", departure_time))
                else:
                    available_vehicle["available_time"] = departure_time
                # Advance simulation time to departure time.
                self.current_time = departure_time
            else:
                # No vehicle available: still advance time by vehicle loading time.
                self.current_time += self.vehicle_loading_time
        else:
            # Wait action: do not dispatch vehicle; simply increment time by waiting interval.
            self.current_time += self.wait_interval

        # Update fleet statuses: Check if any in-transit vehicle/drone has finished its route.
        for vehicle in self.vehicles:
            if vehicle["status"] == "in_transit" and self.current_time >= vehicle["available_time"]:
                vehicle["status"] = "at_depot"
                vehicle["position"] = (0.0, 0.0)
                vehicle["route"] = [{"location": (0.0, 0.0), "time": self.current_time}]
                vehicle["available_time"] = self.current_time
        for drone in self.drones:
            if drone["status"] == "in_transit" and self.current_time >= drone["available_time"]:
                drone["status"] = "at_depot"
                drone["position"] = (0.0, 0.0)
                drone["route"] = [{"location": (0.0, 0.0), "time": self.current_time}]
                drone["available_time"] = self.current_time

        # Add new customer requests from the event list whose arrival time <= current_time.
        while self.event_list and self.event_list[0]["arrival_time"] <= self.current_time:
            new_req = self.event_list.pop(0)
            self.request_buffer.append(new_req)

        # Determine if episode is done.
        done: bool = False
        if (self.current_time >= self.working_minutes and 
            not self.request_buffer and 
            not self.event_list and
            all(vehicle["status"] == "at_depot" for vehicle in self.vehicles) and 
            all(drone["status"] == "at_depot" for drone in self.drones)):
            done = True

        observation: Dict[str, Any] = self._get_observation()
        service_rate: float = (self.total_requests_served / self.total_requests_generated) if self.total_requests_generated > 0 else 0.0
        info["total_requests_served"] = self.total_requests_served
        info["total_requests_generated"] = self.total_requests_generated
        info["service_rate"] = service_rate

        return observation, reward, done, info

    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the simulation to the console.
        """
        print("----- Environment Rendering -----")
        print(f"Current Time: {self.current_time:.2f} minutes")
        print("Vehicles:")
        for vehicle in self.vehicles:
            print(f"  Vehicle {vehicle['id']}: Status: {vehicle['status']}, "
                  f"Available at: {vehicle['available_time']:.2f}, Route: {vehicle['route']}")
        print("Drones:")
        for drone in self.drones:
            print(f"  Drone {drone['id']}: Status: {drone['status']}, "
                  f"Available at: {drone['available_time']:.2f}, Route: {drone['route']}")
        if self.request_buffer:
            print("Request Buffer:")
            for req in self.request_buffer:
                print(f"  Request {req['id']}: Arrival: {req['arrival_time']:.2f}, "
                      f"Deadline: {req['deadline']:.2f}, Location: {req['location']}")
        else:
            print("Request Buffer: Empty")
        print("---------------------------------")

    def _get_observation(self) -> Dict[str, Any]:
        """
        Construct and return the current observation/state.
        Returns:
            Dict[str, Any]: Dictionary with current_time, fleet statuses, and request buffer.
        """
        observation: Dict[str, Any] = {
            "current_time": np.array([self.current_time], dtype=np.float32),
            "vehicles": self.vehicles,
            "drones": self.drones,
            "request_buffer": self.request_buffer,
            "pending_event_count": len(self.event_list)
        }
        return observation

    def _generate_request_events(self) -> List[Dict[str, Any]]:
        """
        Pre-generate a list of customer request events for the day.
        Each event is a dictionary containing id, arrival_time, deadline, and location.
        
        Returns:
            List[Dict[str, Any]]: Sorted list of request events.
        """
        events: List[Dict[str, Any]] = []
        arrival_times: List[float] = []
        if self.temporal_type.strip().lower() == "homo":
            arrival_times = self._generate_homo_arrivals()
        elif self.temporal_type.strip().lower() == "heter":
            arrival_times = self._generate_heter_arrivals()
        else:
            self.logger.warning(f"Unrecognized temporal type '{self.temporal_type}'. Defaulting to homogeneous.")
            arrival_times = self._generate_homo_arrivals()
        
        # For spatial generation, generate each request's location based on spatial_type.
        clusters: List[Tuple[float, float]] = []
        if self.spatial_type.strip().lower() == "power-l":
            clusters = self._generate_powerlaw_clusters()
        
        for req_id, arrival_time in enumerate(arrival_times):
            if self.spatial_type.strip().lower() == "normal":
                location: Tuple[float, float] = self._generate_normal_location()
            elif self.spatial_type.strip().lower() == "power-l":
                location = self._generate_powerlaw_location(clusters)
            else:
                self.logger.warning(f"Unrecognized spatial type '{self.spatial_type}'. Defaulting to normal.")
                location = self._generate_normal_location()
            request: Dict[str, Any] = {
                "id": req_id,
                "arrival_time": arrival_time,
                "deadline": arrival_time + self.request_deadline,
                "location": location
            }
            events.append(request)
        # Sort events by arrival time
        events.sort(key=lambda x: x["arrival_time"])
        return events

    def _generate_homo_arrivals(self) -> List[float]:
        """
        Generate arrival times using a homogeneous Poisson process.
        
        Returns:
            List[float]: Sorted list of arrival times (in minutes).
        """
        arrivals: List[float] = []
        current_time: float = self.request_start_time
        rate: float = self.homo_rate
        while current_time < self.request_end_time:
            # Exponential interarrival time with mean = 1/rate
            interarrival: float = np.random.exponential(scale=1.0/rate)
            current_time += interarrival
            if current_time < self.request_end_time:
                arrivals.append(current_time)
        arrivals.sort()
        return arrivals

    def _generate_heter_arrivals(self) -> List[float]:
        """
        Generate arrival times using a heterogeneous process with two peak intervals.
        
        Returns:
            List[float]: Sorted list of arrival times (in minutes).
        """
        segments: List[Tuple[float, float, float]] = []
        # Define peak intervals from self.peak_intervals
        peak1_start, peak1_end = self.peak_intervals[0]
        peak2_start, peak2_end = self.peak_intervals[1]
        segments.append((self.request_start_time, peak1_start, self.homo_rate))
        segments.append((peak1_start, peak1_end, 2 * self.homo_rate))
        segments.append((peak1_end, peak2_start, self.homo_rate))
        segments.append((peak2_start, peak2_end, 2 * self.homo_rate))
        segments.append((peak2_end, self.request_end_time, self.homo_rate))
        
        arrivals: List[float] = []
        for seg_start, seg_end, seg_rate in segments:
            seg_length: float = seg_end - seg_start
            expected_arrivals: float = seg_rate * seg_length
            count: int = np.random.poisson(expected_arrivals)
            if count > 0:
                seg_arrivals: List[float] = list(np.random.uniform(low=seg_start, high=seg_end, size=count))
                arrivals.extend(seg_arrivals)
        arrivals.sort()
        # Filter arrivals strictly within the [start_time, end_time)
        arrivals = [t for t in arrivals if self.request_start_time <= t < self.request_end_time]
        return arrivals

    def _generate_normal_location(self) -> Tuple[float, float]:
        """
        Generate a customer location from a bivariate normal distribution centered at (0,0).
        
        Returns:
            Tuple[float, float]: (x, y) location.
        """
        x: float = np.random.normal(loc=0.0, scale=self.normal_std)
        y: float = np.random.normal(loc=0.0, scale=self.normal_std)
        return (x, y)

    def _generate_powerlaw_clusters(self) -> List[Tuple[float, float]]:
        """
        Generate cluster centers uniformly in the specified range.
        
        Returns:
            List[Tuple[float, float]]: List of cluster centers.
        """
        clusters: List[Tuple[float, float]] = []
        for _ in range(self.powerlaw_num_clusters):
            x_center: float = np.random.uniform(low=self.cluster_center_min, high=self.cluster_center_max)
            y_center: float = np.random.uniform(low=self.cluster_center_min, high=self.cluster_center_max)
            clusters.append((x_center, y_center))
        return clusters

    def _generate_powerlaw_location(self, clusters: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Generate a location based on a power-law clustering process.
        
        Args:
            clusters (List[Tuple[float, float]]): Pre-generated cluster centers.
        
        Returns:
            Tuple[float, float]: (x, y) location.
        """
        cluster_index: int = int(np.random.choice(len(clusters), p=np.array(self.powerlaw_cluster_weights) / sum(self.powerlaw_cluster_weights)))
        center: Tuple[float, float] = clusters[cluster_index]
        x: float = np.random.normal(loc=center[0], scale=self.powerlaw_cluster_std)
        y: float = np.random.normal(loc=center[1], scale=self.powerlaw_cluster_std)
        return (x, y)
