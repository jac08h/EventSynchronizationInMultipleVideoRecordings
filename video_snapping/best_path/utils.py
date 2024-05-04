# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Dict, List

import numpy as np

from video_snapping.types import Position


def path_cost(cost_matrix: np.ndarray, path: List[Position]) -> float:
    """Calculate weighted path cost.

    Args:
        cost_matrix: Cost matrix.
        path: Path through cost matrix.

    Returns:
        float: Cost of the path.
    """
    if len(path) == 0:
        return 0
    return sum(cost_matrix[i[0]][i[1]] for i in path) / len(path)


def reconstruct_path(previous: Dict[Position, Position],
                     start: Position,
                     end: Position
                     ) -> List[Position]:
    """Reconstruct path from dictionary containing info about previous node for each node.

    Args:
        previous: Dictionary containing (node: previous node in path) mappings.
        start: Path start.
        end: Path end.

    Returns:
        List[Position]: Reconstructed path.
    """
    if end not in previous:
        return []
    path = [end]
    current = end
    while current != start:
        path.append(previous[current])
        current = previous[current]
    path.reverse()
    if start == (-1, -1):
        path = path[1:]
    return path
