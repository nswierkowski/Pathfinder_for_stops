import sys
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from time import time

from exercise1.a_star_by_changes_number import a_star_solution_by_changes
from exercise1.a_star_by_time import a_star_solution
from exercise1.dijkstra import create_road_list_version_b, create_road_list


def get_a_star_solution(start_stop: str, end_stop: str, time: datetime, time_cost: bool) -> Tuple[
    List[Tuple[str, datetime, str, datetime, str]], Optional[timedelta], int]:
    """
    Return a result of A* for given start_stop, end_stop and time.
    :param time_cost:
    :param start_stop: str
    :param end_stop: str
    :param time: datetime
    :return: None
    """
    if time_cost:
        graph_transition, travel_total_time, end_stop = a_star_solution(start_stop, end_stop, time)
        solution = create_road_list(graph_transition, end_stop)
        return solution, travel_total_time, len(solution)
    else:
        graph_transition, travel_total_time, end_stop, change_number = a_star_solution_by_changes(start_stop, end_stop, time)
        return create_road_list(graph_transition, end_stop), travel_total_time, change_number


def handle_user_input() -> Tuple[str, str, str, str]:
    start_stop = input("Enter start stop: ")
    end_stop = input("Enter end stop: ")
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

    return start_stop, end_stop, start_time, cost_type_input == 't'


def main():
    start_stop, end_stop, start_time, cost_type = handle_user_input()
    initial_time = time()
    solution, travel_time, changes_number = get_a_star_solution(start_stop, end_stop, start_time, cost_type)
    result_time = time()
    print(f"Result: {travel_time if cost_type else changes_number}", file=sys.stderr)
    print(f"Time: {result_time - initial_time}", file=sys.stderr)
    for road in solution:
        print(road)



# if __name__=="__main__":
#     main()
