import sys
from datetime import datetime, timedelta
from time import time as get_time
from queue import PriorityQueue
import random
from typing import Dict, Tuple, List, Optional, Callable, Set

import joblib
import pandas as pd

from pathfinding_algorithms.dijkstra import create_road_list, create_road_list_version_b, \
    create_road_list_version_b_with_nodes_number, create_road_list_with_number_of_nodes
from .additional_functions import connection_graph, GraphTransition, travel_between, manhattan_distance, \
    tram_speed, difference_between_datetimes, SolutionRow, get_random_stops, generate_random_time
from pathfinding_algorithms.a_star_by_changes_number import a_star_solution_by_changes
from read_data.node_class import Node


mean_absolute_error = 8281.76614264802
use_regression_model: int = 2
model_file_path = 'linear_regression_time_model.pkl'
model = joblib.load(model_file_path)


def get_time_prediction(node: Node, end_node: Node) -> timedelta:
    """
    Try to predict the time of travel by distance from node to end_node
    :param node: Node (starting node)
    :param end_node: Node (final node)
    :param path_chooser: Dict[Node, Tuple[datetime, datetime, timedelta]]
    :return: timedelta (in seconds)
    """
    if use_regression_model == 1:
        return timedelta(seconds=manhattan_distance(node, end_node) / tram_speed)
    elif use_regression_model == 2:
        return timedelta(seconds=max(model.predict([[manhattan_distance(node, end_node)]])[0][0]-mean_absolute_error, 0))
    else:
        return max(
            timedelta(seconds=max(model.predict([[manhattan_distance(node, end_node)]])[0][0]-mean_absolute_error, 0)),
            timedelta(seconds=manhattan_distance(node, end_node) / tram_speed)
        )


def a_star_solution(start_stop: str, end_stop: str, time: datetime) \
        -> Tuple[GraphTransition, Optional[timedelta], Node]:
    """
    algorithm for finding the shortest path from start stop to end stop using the
     A* algorithm based on the time criterion
    :param start_stop: str (start stop name)
    :param end_stop: str (end stop name)
    :param time: datetime (means the time when the journey begins)
    :param cost_function: Callable[[Node, Node], float] (specifies what the solution criterion is)
    :return: Tuple[GraphTransition, timedelta, Node]
    """
    travel_total_time: Optional[datetime] = None
    start_node: Node = connection_graph[start_stop]

    if not start_node.neighbours:
        raise ValueError(f"The {start_stop} has no neighbours")

    path_chooser: Dict[Node, Tuple[datetime, timedelta, timedelta]] = {
        start_node: (timedelta(seconds=0), timedelta(seconds=0), timedelta(seconds=0))}
    graph_transition: GraphTransition = {start_node: []}
    node_queue: PriorityQueue = PriorityQueue()
    closed: Set[Node] = set()

    node_queue.put((0.0, start_node))
    end_node: Optional[Node] = None
    while not node_queue.empty():
        node = node_queue.get()[1]
        print(f"NODE FROM QUEUE: {node.stop}")

        if node.stop == end_stop:
            print("STOP")
            print(f"Nonetype: {graph_transition[node]}")
            travel_total_time = difference_between_datetimes(time, graph_transition[node][-1][2])
            end_node = node
            print(f"ITS DONE: {end_stop}")
            break

        closed.add(node)
        for next_node_stop in node.neighbours:
            print(f"     NEIGHBOUR {next_node_stop}")
            next_node = connection_graph[next_node_stop]
            travel_between_nodes = travel_between(node, next_node,
                                                  graph_transition[node][-1][1] if graph_transition[node] else time)
            g_next_node = difference_between_datetimes(time, travel_between_nodes[1])
            print(f"    G = {g_next_node}")
            if next_node not in path_chooser and next_node not in closed:
                h_next_node = get_time_prediction(next_node, connection_graph[end_stop])
                print(f"    h = {h_next_node}")
                f_next_node = g_next_node + h_next_node
                print(f"    f = {f_next_node}")
                path_chooser[next_node] = (g_next_node, h_next_node, f_next_node)
                graph_transition[next_node] = \
                    graph_transition[node] + [(next_node, travel_between_nodes[0], travel_between_nodes[1], travel_between_nodes[2])]
                    #node, travel_between_nodes[0], travel_between_nodes[1], travel_between_nodes[2]
                node_queue.put((f_next_node, next_node))
            elif path_chooser[next_node][0] > g_next_node:
                graph_transition[next_node] = \
                    graph_transition[node] + [
                        (next_node, travel_between_nodes[0], travel_between_nodes[1], travel_between_nodes[2])]
                   # node, travel_between_nodes[0], travel_between_nodes[1], travel_between_nodes[2]
                f_next_node = g_next_node + path_chooser[next_node][1]
                print(f"    h = {h_next_node}")
                print(f"    f = {f_next_node}")
                path_chooser[next_node] = (g_next_node, path_chooser[next_node][1], f_next_node)
                if next_node in closed:
                    node_queue.put((f_next_node, next_node))
                    closed.remove(next_node)

    if end_node:
        return graph_transition, travel_total_time, end_node

    raise ValueError(f"There is no solution for such stops {start_stop}, {end_stop}")


def show_a_star_time_solution(start_stop: str, end_stop: str, time: datetime) -> None:
    """
    Show result of A* for given start_stop, end_stop and time.
    The roadmap is going to be printed on standard output and the total time will be printed on standard error output
    :param start_stop: str
    :param end_stop: str
    :param time: datetime
    :return: None
    """
    graph_transition, travel_total_time, end_stop = a_star_solution(start_stop, end_stop, time)
    print(f"Result: {travel_total_time}", file=sys.stderr)
    for road in create_road_list(graph_transition, end_stop):
        print(road)


def a_star_time_save_to_csv(records_number: int, file_path: str) -> None:
    """
    Generates a records_number of random request using dijkstra algorithm and at the end save results to csv of file_path
    :param records_number: int
    :param file_path: str
    :return: None
    """
    results = []
    stop_iterator = get_random_stops()
    for _ in range(records_number):
        print("----------------------------------------------------")
        start_stop, end_stop = next(stop_iterator)

        print(f"START STOP: {start_stop}")
        print(f"END STOP: {end_stop}")

        time = generate_random_time()

        try:
            initial_time = get_time()
            graph_transition, travel_total_time, end = a_star_solution(start_stop, end_stop, time)
            final_time = get_time()
            distance = manhattan_distance(connection_graph[start_stop], connection_graph[end_stop])
            solution, stops_number = create_road_list_with_number_of_nodes(graph_transition, end)
            changes = len(solution)
            total_time = travel_total_time.total_seconds() if travel_total_time else None
            results.append([distance, changes, total_time, time, (final_time-initial_time)*1000, stops_number])
        except ValueError:
            pass

    pd.DataFrame(results, columns=['Distance', 'Change number', 'Travel time', 'Starting hour', 'Executing time', 'Stops number'])\
        .to_csv(file_path, index=False)