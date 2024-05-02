from __future__ import annotations
from datetime import datetime
from typing import Tuple, List, Dict


class Node:

    def __init__(self, stop: str, lat: float, lon: float) -> None:
        self.neighbours: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
        self.lines: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
        self.stop: str = stop
        self.lat: float = lat
        self.lon: float = lon

    def add_neighbour(self, node_stop: str, departure_time: datetime, arrival_time: datetime, line: str) -> None:
        if node_stop in self.neighbours:
            self.neighbours[node_stop].append((departure_time, arrival_time, line))
        else:
            self.neighbours[node_stop] = [(departure_time, arrival_time, line)]

        if line in self.lines:
            self.lines[line].append((departure_time, arrival_time, node_stop))
        else:
            self.lines[line] = [(departure_time, arrival_time, node_stop)]

    def sort_neighbours(self):
        for stop, departure in self.neighbours.items():
            departure.sort(key=lambda x: x[0])

    def sort_lines(self):
        for stop, departure in self.lines.items():
            departure.sort(key=lambda x: x[0])

    def __lt__(self, other):
        return len(self.neighbours) < len(other.neighbours)

    def __gt__(self, other):
        return len(self.neighbours) > len(other.neighbours)
