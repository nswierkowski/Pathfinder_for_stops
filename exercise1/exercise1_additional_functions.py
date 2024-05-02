from datetime import datetime, timedelta
import random
from typing import Dict, Tuple, List, NewType, Optional, Set  # , TypeAlias
from read_data.read_connection_graph import get_nodes_set, read_serialized_object
from read_data.node_class import Node


# Function for constants
def calculate_medium_speed() -> float:
    total_speed = 0
    count = 0

    for node in connection_graph.values():
        for neighbour, schedules in node.neighbours.items():
            for schedule in schedules:
                departure_time, arrival_time, _ = schedule
                time_diff = (arrival_time - departure_time).total_seconds()
                distance = abs(node.lat - neighbour.lat) + abs(node.lon - neighbour.lon)
                speed = distance / time_diff
                total_speed += speed
                count += 1

    if count == 0:
        return 0

    return total_speed / count


def calculate_maximum_speed() -> float:
    """
    Calculates the highest possible speed at which a tram can travel between two stops.
    :return: float
    """
    maximum_speed = 0

    for node in connection_graph.values():
        for neighbour_stop, schedules in node.neighbours.items():
            neighbour = connection_graph[neighbour_stop]
            for schedule in schedules:
                departure_time, arrival_time, _ = schedule
                time_diff = max((arrival_time - departure_time).total_seconds(), 1.0)
                distance = abs(node.lat - neighbour.lat) + abs(node.lon - neighbour.lon)
                maximum_speed = max(distance / time_diff,
                                    maximum_speed)

    return maximum_speed


def calculate_minimum_distance_between_stops() -> float:
    """
    :return: float
    """
    minimum_distance = 1

    for node in connection_graph.values():
        for neighbour_stop, schedules in node.neighbours.items():
            neighbour = connection_graph[neighbour_stop]
            new_distance = abs(node.lat - neighbour.lat) + abs(node.lon - neighbour.lon)
            if new_distance == 0:
                continue
            minimum_distance = min(new_distance,
                                   minimum_distance)

    return minimum_distance


def calculate_line_to_stops_dict() -> Dict[str, Set[str]]:
    """
    :return: Dict[str, Set[str]]
    """
    result: Dict[str, Set[str]] = {}

    for node in connection_graph.values():
        for neighbour_stop, schedules in node.neighbours.items():
            for schedule in schedules:
                _, _, line = schedule
                if line in result:
                    result[line].add(neighbour_stop)
                else:
                    result[line] = {neighbour_stop}
                result[line].add(node.stop)

    return result


# Constants

path_to_serialized_object = 'node_dict.pkl'  # 'tmp_dict.pkl'
connection_graph: Dict[str, Node] = read_serialized_object(path_to_serialized_object)
tram_speed: float = calculate_maximum_speed()
distance_between_stops: float = calculate_minimum_distance_between_stops()
lines: Dict[str, str] = calculate_line_to_stops_dict()

# counted by function model_mean_error in ../additional_scripts/count_coefficent_correlation
regression_model_changes_maximum_error = 63.90568056237911
regression_model_changes_mean_abs_error = 7.221355444043101

print(f"ASSUME MEDIUM SPEED IS: {tram_speed}")
#print(f"ASSUME MEDIUM CHANGES NUMBER IS: {distance_between_stops}")

# aliases
GraphTransition = NewType("GraphTransition", Dict[Node, List[Tuple[Node, datetime, datetime, str]]])
SolutionRow = NewType("SolutionRow", Tuple[str, datetime, str, datetime, str])


def calculate_travel_time(travel: Tuple[datetime, datetime, str]) -> float:
    """
    Counts the time of the travel in seconds
    :param travel: Tuple[datetime, datetime, str]
    :return: float
    """
    return (travel[1] - travel[0]).total_seconds()


def manhattan_distance(first_node: Node, second_node: Node) -> float:
    """
    Counts the manhattan distance between first_node and second_node based on geographic coordinates
    :param first_node: Node
    :param second_node: Node
    :return:
    """
    return abs(first_node.lat - second_node.lat) + abs(first_node.lon - second_node.lon)


def manhattan_distance_by_stops(first_node: str, second_node: str) -> float:
    """
    Counts the manhattan distance between first_node and second_node based on geographic coordinates
    :param first_node: Node
    :param second_node: Node
    :return:
    """
    return manhattan_distance(connection_graph[first_node], connection_graph[second_node])


def travel_between(start: Node, end: Node, time: datetime) -> Tuple[datetime, datetime, str]:
    """
    Find the fastest possible travel between given two nodes
    :param start: Node (starting node)
    :param end: Node (ending node)
    :param time: datetime (current time)
    :return: Tuple[datetime, datetime, str]
    """
    departures: List[Tuple[datetime, datetime]] = start.neighbours[end.stop]
    left, right, closest_bigger = 0, len(departures) - 1, None

    while left <= right:
        mid = (left + right) // 2

        if departures[mid][0] <= time:
            left = mid + 1
        else:
            closest_bigger = departures[mid]
            right = mid - 1

    # If there are no departures after the given time,
    # it means that we need to take the earliest departure (the first one).
    return closest_bigger if closest_bigger else departures[0]


def travel_between_by_line(start: Node, end: Node, destination_node_stop: str, time: datetime) -> Tuple[datetime, datetime, str]:
    """
    Find the line which will reach out destination node or fastest possible travel between given two nodes
    :param destination_node_stop: str
    :param start: Node (starting node)
    :param end: Node (ending node)
    :param time: datetime (current time)
    :return: Tuple[datetime, datetime, str]
    """

    line_to_destination = None
    for line in start.lines.keys():
        if destination_node_stop in lines[line] and end.stop in lines[line]:
            line_to_destination = line
            break

    if not line_to_destination:
        return travel_between(start, end, time)

    left, right, closest_bigger = 0, len(start.lines[line_to_destination]) - 1, None

    while left <= right:
        mid = (left + right) // 2

        if start.lines[line_to_destination][mid][0] <= time:
            left = mid + 1
        else:
            closest_bigger = start.lines[line_to_destination][mid]
            right = mid - 1

    if closest_bigger:
        return closest_bigger[0], closest_bigger[1], line_to_destination
    # If there are no departures after the given time,
    # it means that we need to take the earliest departure (the first one).
    return start.lines[line_to_destination][0][0], start.lines[line_to_destination][0][1], line_to_destination,


def assert_travel_possible(start: Node, end: Node, destination_node_stop: str, time: datetime) -> Optional[Tuple[datetime, datetime, str]]:
    """
    Find whether the travel between given nodes is possible without changes
    :param start: Node (starting node)
    :param end: Node (ending node)
    :param time: datetime
    :return: Optional[Tuple[datetime, datetime, str]]
    """
    return None if end.stop not in start.neighbours else travel_between_by_line(start, end, destination_node_stop, time)


def difference_between_datetimes(start_time: datetime, final_time: datetime) -> timedelta:
    """
    Counts the difference between given datetimes
    :param start_time: datetime
    :param final_time: datetime
    :return: timedelta
    """
    return (final_time - start_time) \
        if final_time > start_time \
        else (final_time - start_time + timedelta(hours=24))


def generate_random_time() -> datetime:
    seconds = f"0{s}" if (s := random.randrange(60)) < 10 else str(s)
    minutes = f"0{m}" if (m := random.randrange(60)) < 10 else str(m)
    hours = f"0{h}" if (h := random.randrange(24)) < 10 else str(h)
    return datetime.strptime(f"{hours}:{minutes}:{seconds}", "%H:%M:%S")


def get_random_stops() -> Tuple[str, str]:
    """
    Return two different stops from graph
    :return: Tuple[str, str]
    """
    stops = list(connection_graph.keys())
    while True:
        start_stop = random.choice(stops)
        end_stop = random.choice(stops)
        while end_stop == start_stop:
            end_stop = random.choice(stops)

        yield start_stop, end_stop
