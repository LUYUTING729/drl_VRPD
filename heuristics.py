"""
heuristics.py

This module encapsulates the routing heuristics (non©\learned) for the same©\day delivery dispatching
and routing problem with vehicles and drones (SD3RVD). It provides two core static methods:

  1. cheapest_insertion(route, request, current_time):
       Determines if a new customer request can be feasibly inserted into a vehicle¡¯s current planned route.
       It iterates over candidate insertion positions in the route, recomputes scheduled arrival times
       (factoring in fixed delays such as vehicle loading and drop-off times), checks deadline constraints,
       and computes the incremental cost of insertion based on the additional travel distance.
       Returns a tuple (feasible, insertion_cost, updated_route).

  2. FIFO_assignment(drone_queue, request):
       Implements a FIFO strategy for drone assignment. It processes the list of available drones
       in FIFO order, calculates the required total trip time (including loading, travel, drop-off,
       and charging), and checks if the selected drone can complete the trip before the customer¡¯s deadline.
       If feasible, it updates the drone¡¯s schedule and reorders the drone queue accordingly.
       Returns a tuple (feasible, updated_drone_queue).

All configuration parameters (speeds, times, scaling factors, deadlines, working time) are read from
the shared configuration file (config.yaml) via the utils.load_config() function. Default values are provided
if configuration is unavailable.
"""

import math
import numpy as np
from typing import Any, Dict, List, Tuple

from utils import (
    load_config,
    compute_euclidean_distance,
    compute_scaled_distance,
    get_logger
)

# Attempt to load the configuration from "config.yaml"; otherwise, use default values.
try:
    CONFIG = load_config("config.yaml")
except Exception as e:
    CONFIG = {
        "environment": {
            "vehicle_speed": 30,           # km/h
            "drone_speed": 40,             # km/h
            "vehicle_loading_time": 5,     # minutes
            "drone_loading_time": 5,       # minutes
            "drop_off_time": 5,            # minutes
            "drone_charging_time": 20,     # minutes
            "grid_block_time": 5,          # minutes per block (if applicable)
            "route_distance_scale": 1.5,   # multiplier for road-network distance conversion
            "request_deadline": 240,       # minutes after arrival
            "working_minutes": 600,        # Work period in minutes (e.g., 10h)
            "request_arrival": {
                "start_time": 0,
                "end_time": 420
            }
        },
        "fleet": {
            "vehicles": 2,
            "drones": 3
        }
    }

logger = get_logger("Heuristics")


class Heuristics:
    """
    Heuristics class encapsulates static methods for routing and assignment decisions.
    """

    @staticmethod
    def cheapest_insertion(
        route: List[Dict[str, Any]],
        request: Dict[str, Any],
        current_time: float
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Attempt to insert a new customer request into the vehicle¡¯s planned route using a cheapest insertion heuristic.

        Args:
            route (List[Dict[str, Any]]): Current planned route, a list of dictionaries where each dict has:
                - "location": Tuple[float, float] representing coordinates.
                - "time": Scheduled arrival time (in minutes).
                (The route is expected to start and end at the depot, assumed to be at (0.0, 0.0)).
            request (Dict[str, Any]): The customer request with keys:
                - "location": Tuple[float, float] of the customer.
                - "arrival_time": When the request arrived.
                - "deadline": Latest time (in minutes) by which the request must be served.
            current_time (float): The current simulation time in minutes.

        Returns:
            Tuple[bool, float, List[Dict[str, Any]]]:
                - A boolean flag indicating if the insertion is feasible.
                - The computed insertion cost (¦¤_cost) (lower is better) if inserted.
                - The updated route with recalculated scheduled times including the inserted request.
                  If insertion is not feasible, returns the original route and a cost of infinity.
        """
        env_conf: Dict[str, Any] = CONFIG.get("environment", {})
        vehicle_speed: float = float(env_conf.get("vehicle_speed", 30))           # km/h
        vehicle_loading_time: float = float(env_conf.get("vehicle_loading_time", 5))  # minutes
        drop_off_time: float = float(env_conf.get("drop_off_time", 5))              # minutes
        route_distance_scale: float = float(env_conf.get("route_distance_scale", 1.5))
        working_minutes: float = float(env_conf.get("working_minutes", 600))
        default_request_deadline: float = float(env_conf.get("request_deadline", 240))
        
        # Use the request's deadline if provided; otherwise, set deadline = arrival_time + default_request_deadline.
        req_deadline: float = float(request.get("deadline", request.get("arrival_time", 0) + default_request_deadline))
        req_location: Tuple[float, float] = request.get("location", (0.0, 0.0))

        best_cost: float = float("inf")
        best_route: List[Dict[str, Any]] = route.copy()
        insertion_feasible: bool = False

        # If the current route is empty or only has one stop, create a simple route: depot -> request -> depot.
        if len(route) < 2:
            depot: Dict[str, Any] = {"location": (0.0, 0.0), "time": None}
            new_route: List[Dict[str, Any]] = [
                depot,
                {"location": req_location, "time": None, "deadline": req_deadline},
                depot.copy()
            ]
            success, scheduled_route = Heuristics._recompute_schedule(
                new_route,
                current_time,
                vehicle_loading_time,
                vehicle_speed,
                drop_off_time,
                route_distance_scale,
                working_minutes
            )
            if success:
                dist_depot_req = compute_euclidean_distance((0.0, 0.0), req_location) * route_distance_scale
                dist_req_depot = compute_euclidean_distance(req_location, (0.0, 0.0)) * route_distance_scale
                insertion_cost = dist_depot_req + dist_req_depot
                return (True, insertion_cost, scheduled_route)
            else:
                return (False, float("inf"), route)

        # Iterate over candidate insertion positions (between consecutive stops).
        for i in range(len(route) - 1):
            # Create a candidate route by inserting the new request between route[i] and route[i+1].
            candidate_route = route.copy()
            inserted_stop = {
                "location": req_location,
                "time": None,
                "deadline": req_deadline
            }
            candidate_route.insert(i + 1, inserted_stop)

            # Recompute the schedule for the candidate route.
            success, updated_candidate_route = Heuristics._recompute_schedule(
                candidate_route,
                current_time,
                vehicle_loading_time,
                vehicle_speed,
                drop_off_time,
                route_distance_scale,
                working_minutes
            )
            if not success:
                continue  # Infeasible insertion at this candidate position.

            # Compute incremental cost:
            # Cost = (d(route[i] -> request) + d(request -> route[i+1]) - d(route[i] -> route[i+1]))
            loc_before: Tuple[float, float] = route[i].get("location", (0.0, 0.0))
            loc_after: Tuple[float, float] = route[i + 1].get("location", (0.0, 0.0))
            cost_before = compute_euclidean_distance(loc_before, loc_after) * route_distance_scale
            cost_candidate = (compute_euclidean_distance(loc_before, req_location) * route_distance_scale +
                              compute_euclidean_distance(req_location, loc_after) * route_distance_scale)
            incremental_cost = cost_candidate - cost_before

            if incremental_cost < best_cost:
                best_cost = incremental_cost
                best_route = updated_candidate_route
                insertion_feasible = True

        if insertion_feasible:
            return (True, best_cost, best_route)
        else:
            return (False, float("inf"), route)

    @staticmethod
    def _recompute_schedule(
        route: List[Dict[str, Any]],
        current_time: float,
        vehicle_loading_time: float,
        vehicle_speed: float,
        drop_off_time: float,
        route_distance_scale: float,
        working_minutes: float
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Recompute the scheduled arrival times for all stops in the route based on current_time and fixed delays.

        The scheduling works as follows:
          - Start at the depot with time = current_time + vehicle_loading_time.
          - For each subsequent stop, compute the travel time from the previous stop:
                travel_time (minutes) = (scaled Euclidean distance between stops / vehicle_speed) * 60
          - For request stops (non-depot, i.e. location != (0.0, 0.0)), add drop_off_time.
          - For each request stop that specifies a "deadline", check if the computed arrival time is within the deadline.
          - Ensure the final depot arrival time is within working_minutes.

        Args:
            route (List[Dict[str, Any]]): Candidate route (list of stops as dictionaries).
            current_time (float): The current simulation time in minutes.
            vehicle_loading_time (float): Time (minutes) at the depot for loading.
            vehicle_speed (float): Vehicle speed in km/h.
            drop_off_time (float): Fixed drop-off time at each request stop.
            route_distance_scale (float): Multiplier to convert Euclidean distance to road-network distance.
            working_minutes (float): Latest allowed completion time (e.g., end of working day).

        Returns:
            Tuple[bool, List[Dict[str, Any]]]:
                - A boolean indicating if the route is feasible (all deadlines met, depot return within working time).
                - The updated route with recalculated "time" values.
        """
        updated_route: List[Dict[str, Any]] = []
        t_current: float = current_time + vehicle_loading_time  # Start time after loading
        # Process the first stop (assumed to be the depot)
        if len(route) == 0:
            return (False, route)
        first_stop = route[0].copy()
        first_stop["time"] = t_current
        updated_route.append(first_stop)
        prev_location: Tuple[float, float] = first_stop.get("location", (0.0, 0.0))

        # Process subsequent stops
        for idx in range(1, len(route)):
            current_stop = route[idx]
            current_location: Tuple[float, float] = current_stop.get("location", (0.0, 0.0))
            # Compute travel distance and then travel time.
            distance: float = compute_euclidean_distance(prev_location, current_location) * route_distance_scale
            travel_time: float = (distance / vehicle_speed) * 60.0  # Convert hours to minutes.
            arrival_time: float = t_current + travel_time

            # If current stop is a request (assumed if location != depot), add drop-off time.
            if current_location != (0.0, 0.0):
                arrival_time += drop_off_time
                # Check deadline if available.
                if "deadline" in current_stop:
                    deadline_value = float(current_stop["deadline"])
                    if arrival_time > deadline_value:
                        # This candidate route is infeasible; deadline missed.
                        return (False, route)
            # Update the stop's scheduled time.
            updated_stop = current_stop.copy()
            updated_stop["time"] = arrival_time
            updated_route.append(updated_stop)

            # Update time for next leg and previous location.
            t_current = arrival_time
            prev_location = current_location

        # Check if final stop (depot) is reached within working_minutes.
        final_stop = updated_route[-1]
        if final_stop.get("location", (0.0, 0.0)) == (0.0, 0.0) and final_stop["time"] > working_minutes:
            return (False, route)

        return (True, updated_route)

    @staticmethod
    def FIFO_assignment(
        drone_queue: List[Dict[str, Any]],
        request: Dict[str, Any]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Assign a customer request to a drone using a FIFO (first-in-first-out) strategy.

        The method iterates through the drone_queue in order and selects the first drone that can complete
        the delivery of the request within its deadline. It computes the travel time from the drone's current
        position to the customer using the drone's speed, and accounts for fixed time delays (loading,
        drop-off, and charging). If a drone is found to feasibly serve the request, its schedule is updated
        (its available_time is advanced, and its route is updated) and the drone is removed from its current
        position in the queue and appended at the end to adhere to FIFO policy.

        Args:
            drone_queue (List[Dict[str, Any]]): List of drones in FIFO order. Each drone dict should contain:
                - "id": Drone identifier.
                - "status": "at_depot" or "in_transit".
                - "position": Current position as tuple (x, y). For drones at depot, typically (0.0, 0.0).
                - "available_time": The earliest time (in minutes) the drone is available.
                - "route": The current planned route.
            request (Dict[str, Any]): Customer request dictionary with:
                - "location": Tuple[float, float] of customer's location.
                - "arrival_time": Time of request arrival (in minutes).
                - "deadline": The latest acceptable delivery time (in minutes).

        Returns:
            Tuple[bool, List[Dict[str, Any]]]:
                - A boolean flag: True if a drone is feasibly assigned to the request, False otherwise.
                - The updated drone_queue with the selected drone's schedule updated if assigned;
                  otherwise, the original drone_queue is returned.
        """
        env_conf: Dict[str, Any] = CONFIG.get("environment", {})
        drone_speed: float = float(env_conf.get("drone_speed", 40))            # km/h
        drone_loading_time: float = float(env_conf.get("drone_loading_time", 5))   # minutes
        drop_off_time: float = float(env_conf.get("drop_off_time", 5))             # minutes
        drone_charging_time: float = float(env_conf.get("drone_charging_time", 20))# minutes

        req_location: Tuple[float, float] = request.get("location", (0.0, 0.0))
        req_arrival: float = float(request.get("arrival_time", 0.0))
        req_deadline: float = float(request.get("deadline", req_arrival + 240))

        # Iterate over the drone queue in FIFO order.
        for index, drone in enumerate(drone_queue):
            drone_position: Tuple[float, float] = drone.get("position", (0.0, 0.0))
            drone_available: float = float(drone.get("available_time", 0.0))

            # Compute the travel distance from the drone's current position (or depot) to request location.
            distance: float = compute_euclidean_distance(drone_position, req_location)
            # For drones, use Euclidean distance directly (no scaling factor).
            travel_time: float = (distance / drone_speed) * 60.0  # in minutes

            # Determine the starting time for the drone (it cannot depart before it is available or before the request arrives).
            start_time: float = max(drone_available, req_arrival)
            # Total trip time includes loading, travel, drop-off, and charging.
            total_trip_time: float = drone_loading_time + travel_time + drop_off_time + drone_charging_time
            finish_time: float = start_time + total_trip_time

            # Check if the drone can complete the trip within the request's deadline.
            if finish_time <= req_deadline:
                # Feasible assignment found.
                # Update the drone's schedule.
                updated_drone = drone.copy()
                # For simplicity, construct a new trip route: current position -> request -> depot.
                new_route: List[Dict[str, Any]] = [
                    {"location": drone_position, "time": start_time},
                    {"location": req_location, "time": start_time + travel_time},
                    {"location": (0.0, 0.0), "time": finish_time}
                ]
                updated_drone["route"] = new_route
                updated_drone["available_time"] = finish_time
                # After completing the trip, the drone returns to the depot.
                updated_drone["position"] = (0.0, 0.0)
                updated_drone["status"] = "in_transit"

                # Update the drone queue: remove the assigned drone and append it at the end to maintain FIFO.
                updated_queue: List[Dict[str, Any]] = drone_queue.copy()
                assigned_drone = updated_queue.pop(index)
                # Replace with updated schedule.
                assigned_drone = updated_drone
                updated_queue.append(assigned_drone)
                logger.info(
                    "Drone ID %s assigned to request at location %s. Finish time: %.2f, Deadline: %.2f",
                    assigned_drone.get("id", "Unknown"), str(req_location), finish_time, req_deadline
                )
                return (True, updated_queue)
            else:
                # This drone cannot complete the assignment in time; try next.
                continue

        # If no drone in the queue is feasible, return False.
        logger.info("No feasible drone in the queue for request at location %s.", str(req_location))
        return (False, drone_queue)
