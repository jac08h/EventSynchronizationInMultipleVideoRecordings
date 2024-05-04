# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import heapq
from collections import defaultdict
from copy import deepcopy
from math import inf
from typing import List, Optional, Set, Tuple

import numpy as np

from video_snapping.best_path.best_path import BestPath
from video_snapping.best_path.utils import path_cost, reconstruct_path
from video_snapping.types import Position


class DijkstraBestPath(BestPath):
    """Class implementing modified version of Dijkstra algorithm for finding synchronization between two videos.

    Args:
        allow_partial_overlap: Allow only partial overlap.
        min_path_length: Minimum path length if partial overlap is allowed.
        path_length_for_slope_estimation: How many steps to go back to fo find start node for slope calculation.
        min_slope: Minimum allowed slope.
        max_slope: Maximum allowed slope.
        penalty: Penalty for nodes that would break the allowed slope boundaries.
    """

    def __init__(self,
                 allow_partial_overlap: bool,
                 min_path_length: int,
                 use_warping_constraints: bool,
                 path_length_for_slope_estimation: int,
                 min_slope: float,
                 max_slope: float,
                 penalty: float
                 ):
        super().__init__()
        self.allow_partial_overlap = allow_partial_overlap
        self.min_path_length = min_path_length
        self.use_warping_constraints = use_warping_constraints
        self.path_length_for_slope_estimation = path_length_for_slope_estimation
        self.sigma_min = min_slope
        self.sigma_max = max_slope
        self.penalty = penalty

    def best_path(self,
                  cost_matrix: np.ndarray,
                  ) -> Optional[List[Position]]:
        """Find the best path through cost matrix using modified dijkstra algorithm described in paper.

        Args:
            cost_matrix: Cost matrix.

        Returns:
            Optional[List[Position]]: Best path if at least one path was found, None otherwise.
        """
        h, w = cost_matrix.shape
        possible_ends = {(h - 1, w - 1)}
        if not self.allow_partial_overlap:
            all_paths = self.dijkstra(cost_matrix, (0, 0), possible_ends)

        else:
            for y in range(0, h):
                possible_ends.add((y, w - 1))
            for x in range(0, w):
                possible_ends.add((h - 1, x))

            all_paths = self.dijkstra(cost_matrix, (-1, -1), possible_ends)
            all_paths = [p for p in all_paths if len(p) > self.min_path_length]
            all_paths.sort(key=lambda p: path_cost(cost_matrix, p))
        if len(all_paths) == 0:
            return None
        best_path = all_paths[0]
        if self.allow_partial_overlap:
            best_path = self.pad_partial_path(best_path, h, w)
        return best_path

    def dijkstra(self,
                 cost_matrix: np.ndarray,
                 start: np.array,
                 possible_ends: Set[Position],
                 ) -> List[List[Position]]:
        """Helper function for dijkstra algorithm.

        Args:
            cost_matrix: Cost matrix.
            start: Start position.
            possible_ends: Possible end positions.

        Returns:
            List[List[Position]]: List of possible paths.
        """
        costs = defaultdict(lambda: inf)
        costs[start] = 0
        previous = defaultdict(lambda: (-1, -1))
        queue: List[Tuple[float, Position]] = [(0, start)]
        visited = set()
        paths = []

        height, width = cost_matrix.shape
        while len(queue) > 0:
            priority, current = heapq.heappop(queue)
            if current in visited:
                continue

            local_path_start = None
            if self.use_warping_constraints:
                local_path = reconstruct_path(previous, start, current)[-self.path_length_for_slope_estimation + 1:]
                if len(local_path) > 0:
                    local_path_start = local_path[0]

            for neighbor in self.get_successors(current, height, width):
                new_cost = costs[current] + cost_matrix[neighbor]
                if local_path_start:
                    y_start, x_start = local_path_start
                    y_end, x_end = neighbor
                    if x_end - x_start == 0:
                        new_cost += self.penalty
                    else:
                        slope = (y_end - y_start) / (x_end - x_start)
                        if slope < self.sigma_min or slope > self.sigma_max:
                            new_cost += self.penalty

                if costs[neighbor] == inf or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor))
                    previous[neighbor] = current

            visited.add(current)
            if current in possible_ends:
                paths.append(reconstruct_path(previous, start, current))
        return paths

    @staticmethod
    def pad_partial_path(path: List[Position], height: int, width: int) -> List[Position]:
        """Pad partial path to start at top left corner of the cost matrix and end in the bottom right corner.
        Args:
            path: Partial path.
            height: Height of the cost matrix.
            width: Width of the cost matrix.
        Returns:
            List[Position]: Padded path.
        """
        padded_path = deepcopy(path)
        if path[0] != (0, 0):
            n_1, m_1 = path[0]
            if n_1 != 0:
                path_prefix = [(n, 0) for n in range(n_1)]
            else:
                path_prefix = [(0, m) for m in range(m_1)]
            padded_path = path_prefix + padded_path

        if path[-1] != (height - 1, width - 1):
            n_L, m_L = path[-1]
            if n_L != height - 1:
                path_suffix = [(n, m_L) for n in range(n_L + 1, height)]
            else:
                path_suffix = [(n_L, m) for m in range(m_L + 1, width)]
            padded_path = padded_path + path_suffix
        return padded_path
