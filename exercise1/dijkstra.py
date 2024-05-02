import random
from itertools import groupby

import pandas as pd
import sys
from datetime import datetime, timedelta
from time import time as get_time
from queue import PriorityQueue
from typing import Dict, Tuple, List, Optional
from read_data.read_connection_graph import Node
from .exercise1_additional_functions import connection_graph, travel_between, calculate_travel_time, GraphTransition, \
    SolutionRow, difference_between_datetimes, manhattan_distance, generate_random_time, get_random_stops


def create_road_list_version_b(graph_transition: GraphTransition, end_stop: Node) -> List[SolutionRow]:
    """
    Converts a dictionary symbolising a node transition into a sequentially ordered list of nodes/stops
    :param graph_transition: Dict[Node, Tuple[Node, datetime, datetime, str]]
    :param end_stop: Node
    :return: List[SolutionRow] (line name, departure time, start_stop name, arrival time, end stop name)
    """
    travel_to_end_node = graph_transition[end_stop]
    result = []
    while travel_to_end_node:
        result.insert(0, (travel_to_end_node[3],
                          travel_to_end_node[1],
                          travel_to_end_node[0].stop,
                          travel_to_end_node[2],
                          end_stop.stop))
        end_stop = travel_to_end_node[0]
        travel_to_end_node = graph_transition[end_stop]

    condensed_records = []
    current_tram_line = None

    for record in result:
        tram_line, _, _, _, _ = record

        if tram_line != current_tram_line:
            condensed_records.append(record)
            current_tram_line = tram_line

    condensed_records[-1] = condensed_records[-1][0], condensed_records[-1][1], condensed_records[-1][2], result[-1][
        -2], result[-1][-1]
    return condensed_records



def create_road_list_version_b_with_nodes_number(graph_transition: GraphTransition, end_stop: Node) -> List[SolutionRow]:
    """
    Converts a dictionary symbolising a node transition into a sequentially ordered list of nodes/stops
    :param graph_transition: Dict[Node, Tuple[Node, datetime, datetime, str]]
    :param end_stop: Node
    :return: List[SolutionRow] (line name, departure time, start_stop name, arrival time, end stop name)
    """
    travel_to_end_node = graph_transition[end_stop]
    result = []
    while travel_to_end_node:
        result.insert(0, (travel_to_end_node[3],
                          travel_to_end_node[1],
                          travel_to_end_node[0].stop,
                          travel_to_end_node[2],
                          end_stop.stop))
        end_stop = travel_to_end_node[0]
        travel_to_end_node = graph_transition[end_stop]

    condensed_records = []
    current_tram_line = None

    for record in result:
        tram_line, _, _, _, _ = record

        if tram_line != current_tram_line:
            condensed_records.append(record)
            current_tram_line = tram_line

    condensed_records[-1] = condensed_records[-1][0], condensed_records[-1][1], condensed_records[-1][2], result[-1][
        -2], result[-1][-1]
    return condensed_records, len(result)


def create_road_list(graph_transition: GraphTransition, end_stop: Node) -> List[SolutionRow]:
    """
    Converts a dictionary symbolising a node transition into a sequentially ordered list of nodes/stops
    :param graph_transition: Dict[Node, List[Tuple[Node, datetime, datetime, str]]]
    :param end_stop: Node
    :return: List[SolutionRow] (line name, departure time, start_stop name, arrival time, end stop name)
    """
    travel_to_end_node = graph_transition[end_stop]
    print(str(travel_to_end_node[0]))
    result = [
        [travel_to_end_node[0][3],
         travel_to_end_node[0][1],
         travel_to_end_node[0][0].stop,
         None,
         None]
    ]
    prev_line = travel_to_end_node[0][3]
    prev_arrival = travel_to_end_node[0][2]
    for road in travel_to_end_node:
        if prev_line != road[3]:
            result[-1][4] = road[0].stop
            result[-1][3] = prev_arrival
            result.append([road[3], road[1], road[0].stop, road[2], None])
            prev_line = road[3]
            prev_arrival = road[2]

    result[-1][4] = end_stop.stop
    result[-1][3] = prev_arrival
    return result


def create_road_list_with_number_of_nodes(graph_transition: GraphTransition, end_stop: Node)\
    -> Tuple[List[SolutionRow], int]:
    """
    Decorates functions "create_road_list" further adding the number of stops on the way
    """
    return create_road_list(graph_transition, end_stop), len(graph_transition[end_stop])


def dijkstra_solution(start_stop: str, end_stop: str, time: datetime) \
        -> Tuple[GraphTransition, timedelta, Node]:
    """
    algorithm for finding the shortest path from start stop to end stop using the
    Dijkstra's algorithm based on the time criterion
    :param start_stop: str (start stop name)
    :param end_stop: str (end stop name)
    :param time: datetime (means the time when the journey begins)
    :return: Tuple[Dict[Node, Optional[Tuple[Node, datetime, datetime, str]]], timedelta, Node]
    """
    travel_total_time: Optional[datetime] = None

    if start_stop not in connection_graph:
        raise ValueError(f"The {start_stop} stop does not exist")
    start_node: Node = connection_graph[start_stop]

    if not start_node.neighbours:
        raise ValueError(f"The {start_stop} stop does not allow departure")

    the_shortest_path_cost: Dict[Node, datetime] = {start_node: 0}
    graph_transition: GraphTransition = {start_node: []}
    nodes_queue: PriorityQueue = PriorityQueue()

    nodes_queue.put((0.0, start_node))
    while not nodes_queue.empty():
        next_node: Node = nodes_queue.get()[1]
        print(f"FROM QUEUE: {next_node.stop}")

        if next_node.stop == end_stop:
            print(f"    FIND: {end_stop}")
            travel_total_time = timedelta(seconds=the_shortest_path_cost[
                next_node])  # difference_between_datetimes(time, the_shortest_path_cost[next_node])
            break

        print("Not stop not yet")
        for neighbour_stop in next_node.neighbours:
            neighbour = connection_graph[neighbour_stop]
            print(f"    NEIGHBOUR : {neighbour_stop}")
            travel_tuple: Tuple[datetime, datetime, str] = travel_between(next_node,
                                                                          neighbour,
                                                                          time + timedelta(
                                                                              seconds=the_shortest_path_cost[
                                                                                  next_node]))

            print(f"    POTENTIAL TIME OF ARRIVAL : {travel_tuple[1]}")
            cost = difference_between_datetimes(time, travel_tuple[1]).total_seconds()
            if neighbour not in the_shortest_path_cost \
                    or the_shortest_path_cost[neighbour] > cost:  # travel_tuple[1]:
                print(f"    YES IT NEW VALUE")
                the_shortest_path_cost[neighbour] = cost  # travel_tuple[1]
                graph_transition[neighbour] = \
                    graph_transition[next_node] + [(next_node, travel_tuple[0], travel_tuple[1], travel_tuple[2])]
                nodes_queue.put((cost, neighbour))

    return graph_transition, travel_total_time, next_node


def show_dijkstra_solution(start_stop: str, end_stop: str, time: datetime) -> None:
    """
    Show result of dijkstra_solution for given start_stop, end_stop and time.
    The roadmap is going to be printed on standard output and the total time will be printed on standard error output
    :param start_stop: str
    :param end_stop: str
    :param time: datetime
    :return: None
    """
    initial_time = get_time()
    graph_transition, travel_total_time, end_stop = dijkstra_solution(start_stop, end_stop, time)
    ending_time = get_time()
    print(f"Result: {travel_total_time}", file=sys.stderr)
    print(f"Time: {ending_time - initial_time}", file=sys.stderr)
    for road in create_road_list(graph_transition, end_stop):
        print(road)


def save_to_csv(records_number: int, file_path: str) -> None:
    """
    Generates a records_number of random request using dijkstra algorithm and at the end save results to csv of file_path
    :param records_number: int
    :param file_path: str
    :return: None
    """
    results = []
    iterator = get_random_stops()
    for _ in range(records_number):
        print("----------------------------------------------------")
        start_stop, end_stop = next(iterator)

        print(f"START STOP: {start_stop}")
        print(f"END STOP: {end_stop}")

        time = generate_random_time()

        try:
            initial_time = get_time()
            graph_transition, travel_total_time, end = dijkstra_solution(start_stop, end_stop, time)
            final_time = get_time()
            distance = manhattan_distance(connection_graph[start_stop], connection_graph[end_stop])
            solution, stops_number = create_road_list_with_number_of_nodes(graph_transition, end)
            changes = len(solution)
            if not travel_total_time:
                print(end.stop)
                print(graph_transition[end])
            total_time = travel_total_time.total_seconds() if travel_total_time else None
            results.append([distance, changes, total_time, time, (final_time-initial_time)*1000, stops_number])
        except ValueError:
            pass

    pd.DataFrame(results, columns=['Distance', 'Change number', 'Travel time', 'Starting hour', 'Executing time', 'Stops number'])\
        .to_csv(file_path, index=False)
