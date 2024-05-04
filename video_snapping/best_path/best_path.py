# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Iterator

from video_snapping.types import Position


class BestPath:
    def __init__(self):
        self.predecessor_offsets = [(0, -1), (-1, 0), (-1, -1)]
        self.successors_offsets = [(0, 1), (1, 0), (1, 1)]

    def get_successors(self,
                       position: Position,
                       height: int,
                       width: int,
                       ) -> Iterator[Position]:
        """Get possible successors of a node.

        Args:
            position: Current position.
            height: Cost matrix height.
            width: Cost matrix width.

        Returns:
            Iterator[Position]: Successors of a node.
        """
        # (-1, -1) represents super-source node,
        # which is connected to all first frames of first and second video
        # (see Figure 3 in VideoSnapping paper).
        if position == (-1, -1):
            for x in range(width):
                yield 0, x
            for y in range(1, height):
                yield y, 0
        else:
            y, x = position
            for y_offset, x_offset in self.successors_offsets:
                yy = y + y_offset
                xx = x + x_offset
                if yy < height and xx < width:
                    yield yy, xx

    def get_predecessors(self, position: Position) -> Iterator[Position]:
        """Get possible predecessors of a node.

        Args:
            position: Current position.

        Returns:
            Iterator[Position]: Predecessors of a node.
        """
        y, x = position
        for y_offset, x_offset in self.predecessor_offsets:
            yy = y + y_offset
            xx = x + x_offset
            if yy >= 0 and xx >= 0:
                yield yy, xx
