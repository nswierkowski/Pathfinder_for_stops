from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple
from .node_class import Node
import pickle
# pd.set_option('display.max_columns', None)


def __read_connection_graph() -> pd.DataFrame:
    connection_graph = pd.read_csv("./data/fixed_connection_graph.csv", usecols=lambda x: x != 'Unnamed: 0',
                                   dtype={'line': str},
                                   converters={'departure_time': lambda x: pd.to_datetime(x, format='%H:%M:%S'),
                                               'arrival_time': lambda x: pd.to_datetime(x, format='%H:%M:%S')})
    return connection_graph


def sort_all_neighbours(stop_to_node_dict: Dict[str, Node]) -> None:
    for stop, node in stop_to_node_dict.items():
        node.sort_neighbours()
        node.sort_lines()


def serialize_nodes(node_dict: Dict[str, Node], path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(node_dict, f)


def read_serialized_object(file_path: str) -> Dict[str, Node]:
    try:
        print("TRY OPEN")
        with open(file_path, 'rb') as f:
            serialized_object = pickle.load(f)
        return serialized_object
    except FileNotFoundError:
        print("FILE NOT FOUND")
        nodes = get_nodes_set()
        serialize_nodes(nodes, file_path)
        return nodes
    except pickle.UnpicklingError:
        nodes = get_nodes_set()
        serialize_nodes(nodes, file_path)
        return nodes


def get_nodes_set() -> Dict[str, Node]:
    stop_to_node_dict: Dict[str, Node] = {}
    for index, row in __read_connection_graph().iterrows():

        start_stop = row['start_stop']
        if start_stop not in stop_to_node_dict:
            stop_to_node_dict[start_stop] = Node(start_stop, row['start_stop_lat'], row['start_stop_lon'])

        end_stop = row['end_stop']
        if end_stop not in stop_to_node_dict:
            stop_to_node_dict[end_stop] = Node(end_stop, row['end_stop_lat'], row['end_stop_lon'])

        print(f"{start_stop} add neighbour {end_stop}")
        stop_to_node_dict[start_stop].add_neighbour(stop_to_node_dict[end_stop].stop, row['departure_time'],
                                                    row['arrival_time'], line=row['line'])
    sort_all_neighbours(stop_to_node_dict)
    print("----------------------------------------")
    return stop_to_node_dict
