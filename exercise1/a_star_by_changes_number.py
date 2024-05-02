import sys
from datetime import datetime, timedelta
from queue import PriorityQueue
from time import time as get_time

from typing import Dict, Tuple, List, Optional, Callable

import joblib
import pandas as pd

from exercise1.dijkstra import create_road_list, create_road_list_version_b, create_road_list_with_number_of_nodes
from .exercise1_additional_functions import connection_graph, GraphTransition, travel_between, manhattan_distance, \
    tram_speed, assert_travel_possible, difference_between_datetimes, distance_between_stops, get_random_stops, \
    generate_random_time
from read_data.node_class import Node


mean_absolute_error = 9.999262935027797
model_file_path = 'linear_regression_change_model.pkl'
model = joblib.load(model_file_path)


def get_change_number_prediction(node: Node, end_node: Node) -> float:
    """
    Try to predict the number of changes upon the travel based on distance from node to end_node
    :param node: Node (starting node)
    :param end_node: Node (final node)
    :return: float (estimation of number of changes)
    """
    return max(model.predict([[manhattan_distance(node, end_node)]])[0][0]-mean_absolute_error, 0)


def a_star_solution_by_changes(start_stop: str, end_stop: str, time: datetime) \
        -> Tuple[GraphTransition, Optional[timedelta], Node, int]:
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

    path_chooser: Dict[Node, Tuple[int, float, float, str]] = {
        start_node: (0, 0, 0, None)}
    graph_transition: GraphTransition = {start_node: []}
    if not start_node.neighbours:
        raise ValueError(f"The {start_stop} has no neighbour")
    opened = PriorityQueue()
    closed = set()

    opened.put((path_chooser[start_node][2], start_node))
    node: Node = start_node
    while not opened.empty():
        node = opened.get()[1]
        print(f"FROM QUEUE: {node.stop}")

        if node.stop == end_stop:
            print(f"FINITO")
            final_time = graph_transition[node][-1][2] if graph_transition[node] else time
            travel_total_time = difference_between_datetimes(time, final_time)
            break

        closed.add(node)
        for next_node_stop in node.neighbours:
            print(f"    Neighbour: {next_node_stop}")
            next_node = connection_graph[next_node_stop]
            travel_between_nodes = assert_travel_possible(node,
                                                          next_node,
                                                          end_stop,
                                                          graph_transition[node][-1][2]
                                                          if graph_transition[node]
                                                          else time)
            g_next_node = (path_chooser[node][0] + (1 if path_chooser[node][3] != travel_between_nodes[2] else 0)) \
                if travel_between_nodes \
                else sys.maxsize
            print(f"    g_next_node: {g_next_node}")
            if next_node not in path_chooser and next_node not in closed:
                h_next_node = get_change_number_prediction(next_node, connection_graph[end_stop])
                f_next_node = g_next_node + h_next_node
                opened.put((f_next_node, next_node))
                print(f"    H = {h_next_node}")
                print(f"    F = {f_next_node}")
                path_chooser[next_node] = (g_next_node, h_next_node, f_next_node, travel_between_nodes[2])
                graph_transition[next_node] = \
                    graph_transition[node] + [(node, travel_between_nodes[0], travel_between_nodes[1], travel_between_nodes[2])]
            elif path_chooser[next_node][0] > g_next_node:
                graph_transition[next_node] = \
                    graph_transition[node] + [(node, travel_between_nodes[0], travel_between_nodes[1],
                                              travel_between_nodes[2])]
                f_next_node = g_next_node + path_chooser[next_node][1]
                print(f"    H = {path_chooser[next_node][1]}")
                print(f"    F = {f_next_node}")
                path_chooser[next_node] = (g_next_node, path_chooser[next_node][1], f_next_node, travel_between_nodes[2])
                if next_node in closed:
                    opened.put((f_next_node, next_node))
                    closed.remove(next_node)

    return graph_transition, travel_total_time, node, path_chooser[node][0]


def show_a_star_change_solution(start_stop: str, end_stop: str, time: datetime) -> None:
    """
    Show result of A* for given start_stop, end_stop and time.
    The roadmap is going to be printed on standard output and the total time will be printed on standard error output
    :param start_stop: str
    :param end_stop: str
    :param time: datetime
    :return: None
    """
    graph_transition, travel_total_time, end_stop, changes = a_star_solution_by_changes(start_stop, end_stop, time)
    print(f"Result: {changes}", file=sys.stderr)
    for road in create_road_list(graph_transition, end_stop):
        print(road)


def a_star_change_save_to_csv(records_number: int, file_path: str) -> None:
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
            graph_transition, travel_total_time, end, changes = a_star_solution_by_changes(start_stop, end_stop, time)
            final_time = get_time()
            distance = manhattan_distance(connection_graph[start_stop], connection_graph[end_stop])
            solution, stops_number = create_road_list_with_number_of_nodes(graph_transition, end)
            print(f"Solution {solution}")
            print(f"stops_number {stops_number}")
            total_time = travel_total_time.total_seconds() if travel_total_time else None
            #print(f"-{distance, changes, total_time, time, (final_time-initial_time)*1000, stops_number}----------------------------------")
            results.append([distance, changes, total_time, time, (final_time-initial_time)*1000, stops_number])
        except ValueError:
            pass

    pd.DataFrame(results, columns=['Distance',
                                   'Change number',
                                   'Travel time',
                                   'Starting hour',
                                   'Executing time',
                                   'Stops number'])\
        .to_csv(file_path, index=False)