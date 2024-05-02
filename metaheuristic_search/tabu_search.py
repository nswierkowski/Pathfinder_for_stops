import math
import random
import sys
from datetime import datetime, timedelta
from time import time as get_time
from itertools import permutations
from typing import List, Dict, Tuple, Set

from ex1_main import get_a_star_solution
from pathfinding_algorithms.a_star_by_time import a_star_solution
from pathfinding_algorithms.a_star_by_changes_number import a_star_solution_by_changes
from pathfinding_algorithms.additional_functions import manhattan_distance_by_stops

tabu_roads_time: Dict[str, Dict[str, Tuple[float, timedelta]]] = {}
tabu_roads_change: Dict[str, Dict[str, Tuple[int, timedelta]]] = {}
DEFAULT_DAYS_VALUE = 24
infinite_time: timedelta = timedelta(days=24)
distance_matrix: Dict[str, Dict[str, float]] = {}
tabu: Set[Tuple[str]] = set()
improve_mode: bool = False


def next_lexicographic_permutation(stops: List[str]) -> None:
    """
    Generate next lexicographic permutation of a list.
    :param stops:
    :return: None
    """
    print("---next permutation")
    n: int = len(stops)
    i: int = n - 2
    if i < 0:
        return

    j: int = n - 1
    while j >= 0 and stops[j] <= stops[i]:
        j -= 1

    stops[i], stops[j] = stops[j], stops[i]
    stops[i + 1:] = reversed(stops[i + 1:])


def get_travel_changes(start: str, end: str, time: datetime) -> Tuple[int, timedelta]:
    if start in tabu_roads_change and end in tabu_roads_change[start]:
        return tabu_roads_change[start][end]
    else:
        _, travel_time, _, changes = a_star_solution_by_changes(start, end, time)
        if not travel_time:
            tabu_roads_change[start][end] = sys.maxsize, infinite_time
            tabu_roads_change[end][start] = sys.maxsize, infinite_time
            return sys.maxsize, infinite_time
        return changes, travel_time


def estimate_road_changes(start: str, stops: List[str], time: datetime) -> float:
    """
    Estimate the number of changes of the whole travel
    :param start:
    :param stops:
    :param time:
    :return: float (total time of all travel)
    """
    previous_node: str = start
    result: float = 0.0
    for stop in stops:
        print(f"ESTIMATE {stops}")
        print(f"from {previous_node} to {stop}")
        changes_number, travel_time = get_travel_changes(previous_node, stop, time)
        result += changes_number
        previous_node = stop

    return result + get_travel_changes(previous_node, start, time)[0]


def get_travel_time(start: str, end: str, time: datetime) -> Tuple[float, timedelta]:
    if start in tabu_roads_time and end in tabu_roads_time[start]:
        return tabu_roads_time[start][end]
    else:
        _, travel_time, _ = a_star_solution(start, end, time)
        if not travel_time:
            tabu_roads_time[start][end] = sys.maxsize, infinite_time
            tabu_roads_time[end][start] = sys.maxsize, infinite_time
            return sys.maxsize, infinite_time
        return travel_time.total_seconds(), travel_time


def estimate_road_time(start: str, stops: List[str], time: datetime) -> float:
    """
    Estimate the time of whole travel
    :param start:
    :param stops:
    :param time:
    :return: float (total time of all travel)
    """
    previous_node: str = start
    result: float = 0.0
    for stop in stops:
        print(f"ESTIMATE {stops}")
        print(f"from {previous_node} to {stop}")
        time_in_seconds, travel_time = get_travel_time(previous_node, stop, time)
        result += time_in_seconds
        previous_node = stop

    return result + get_travel_time(previous_node, start, time)[0]


def count_distance(start_stop: str, road: List[str]) -> float:
    prev_stop = start_stop
    result = 0
    for stop in road:
        result += manhattan_distance_by_stops(prev_stop, stop)
        prev_stop = stop
    result += manhattan_distance_by_stops(prev_stop, start_stop)
    return result


def assert_better_solution(main_list: List[str], alternative_lists: List[List[str]], start_stop: str):
    main_distance = count_distance(start_stop, main_list)
    minimum_distance = sys.maxsize
    for i, alternative_list in enumerate(alternative_lists):
        minimum_distance = min(minimum_distance, count_distance(start_stop, alternative_list))
    return main_list if main_distance < minimum_distance else alternative_lists[i]


def generate_next_main_permutation(main_list: List[str]):
    next_lexicographic_permutation(main_list)


def generate_next_alternative_permutation(alternative_lists: List[List[str]]):
    def __generate(given_alternative_list: List[str]):
        i = random.randint(0, len(given_alternative_list) - 1)
        j = random.randint(0, len(given_alternative_list) - 1)
        while i == j:
            j = random.randint(0, len(given_alternative_list) - 1)
        tmp = given_alternative_list[i]
        given_alternative_list[i] = given_alternative_list[j]
        given_alternative_list[j] = tmp

    for alternative_list in alternative_lists:
        __generate(alternative_list)


def tabu_search_optimized(start: str, stops: List[str], time: datetime, T: int = sys.maxsize, time_cost: bool = True,
                          population_size: int = 2):
    alternative_lists = [stops.copy() * population_size]
    main_list = stops.copy()
    main_list.sort()
    iteration_number = min(T, math.factorial(len(stops)))
    the_best_travel_cost = sys.maxsize
    for i in range(iteration_number):
        print(f"NEW ITERATION: {i}")

        best_fitness_solution = assert_better_solution(main_list, alternative_lists, start)

        new_travel_cost: float = estimate_road_time(start, best_fitness_solution, time) \
            if time_cost \
            else estimate_road_changes(start, best_fitness_solution, time)
        tabu.add(tuple(best_fitness_solution))

        if new_travel_cost < the_best_travel_cost:
            the_best_travel_cost = new_travel_cost
            the_best_road = main_list

        if len(tabu) >= iteration_number:
            break
        generate_next_alternative_permutation(alternative_lists)
        generate_next_main_permutation(main_list)

    result = []
    prev = start
    current_time = time
    total_number_of_changes = 0
    for stop in the_best_road:
        travel_result, travel_cost, changes_number = get_a_star_solution(prev, stop, current_time, time_cost)
        current_time += travel_cost
        total_number_of_changes += changes_number
        prev = stop
        result.extend(travel_result)
    travel_result, travel_time, changes_number = get_a_star_solution(prev, start, current_time, time_cost)
    result.extend(travel_result)
    return result, (current_time+travel_time-time).total_seconds() if time_cost else total_number_of_changes + changes_number


def tabu_search_naive(start: str, stops: List[str], time: datetime, T: int = sys.maxsize, time_cost: bool = True):
    all_permutations = [list(perm) for perm in permutations(stops)]
    the_best_travel_cost: float = estimate_road_time(start, stops, time) if time_cost else estimate_road_changes(start,
                                                                                                                 stops,
                                                                                                                 time)
    the_best_road = stops

    for road in all_permutations[:min(T, len(all_permutations))]:
        print(f"NEW ROAD: {road}")
        new_travel_cost: float = estimate_road_time(start, road, time) \
            if time_cost \
            else estimate_road_changes(start, road, time)
        if new_travel_cost < the_best_travel_cost:
            the_best_travel_cost = new_travel_cost
            the_best_road = road

    result = []
    prev = start
    current_time = time
    total_number_of_changes = 0
    for stop in the_best_road:
        travel_result, travel_cost, changes_number = get_a_star_solution(prev, stop, current_time, time_cost)
        current_time += travel_cost
        total_number_of_changes += changes_number
        prev = stop
        result.extend(travel_result)
    travel_result, travel_cost, changes_number = get_a_star_solution(prev, start, current_time, time_cost)
    result.extend(travel_result)
    return result, (current_time+travel_cost-time).total_seconds() if time_cost else total_number_of_changes + changes_number


def show_tabu_search(start: str, stops: List[str], time: datetime, T: int = sys.maxsize, time_cost: bool = True):
    initial_time = get_time()
    result, cost = tabu_search_naive(start, stops, time, T, time_cost) \
        if not improve_mode \
        else tabu_search_optimized(start, stops, time, T, time_cost)
    ending_time = get_time()
    print(f"Result: {timedelta(seconds=cost) if time_cost else int(cost)}", file=sys.stderr)
    print(f"Time: {ending_time - initial_time}")
    for road in result:
        print(road)


def handle_user_input() -> None:
    start_stop = input("Enter start stop: ")
    stops = []
    while next_stop := input("Enter the name of the stop you want to pass through (or enter Q to break): ") != 'Q':
        stops.append(next_stop)
    start_time_input = input("Enter start time [%H:%M:%S]: ")
    start_time = None
    while not start_time:
        try:
            start_time = datetime.strptime(start_time_input, "%H:%M:%S")
        except ValueError:
            print("The time should be in format '%H:%M:%S'")
            start_time_input = input("Enter start time [%H:%M:%S]: ")

    cost_type_input = input("Enter whether you'd like cost to be time or number of changes [t/c]: ")
    while cost_type_input not in {'c', 't'}:
        print("ERROR: Wrong input. press 't' if time or 'c' if changes")
        cost_type_input = input("Enter whether you'd like cost to be time or number of changes [t/c]: ")

    t = int(input("Enter maximum number of iteration: "))

    show_tabu_search(start_stop, stops, start_time, t, cost_type_input == 't')