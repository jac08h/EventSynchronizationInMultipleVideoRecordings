# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import numpy as np


@dataclass
class VideoFeatures:
    """Class for holding information about features detected in a video.

    Args:
        frame_indexes: Array of shape (N): Index of frame where the feature was detected.
        coords: Array of shape (N, 2): Image-space coordinates of the features.
        descriptors: Array: (N, D): Descriptors of the features. D is dimension of the descriptor used.
    """
    frame_indexes: np.ndarray
    coords: np.ndarray
    descriptors: np.ndarray


class Visualization(Enum):
    NONE = auto()
    PLAIN = auto()
    WITH_KEYPOINTS = auto()


class TestMetric(Enum):
    """See `tests/config_documentation.md` for further description of each metric."""
    DIFFERENCES = auto()
    CORRECT_FRAMES = auto()


class ErrorRateUnit(Enum):
    FRAME = auto()
    SECOND = auto()


class HistogramMethod(Enum):
    """See `tests/config_documentation.md` for further description of each method."""
    PAPER = auto()
    SIMPLIFIED = auto()


Position = Tuple[int, int]
# Matches[j][k] = List of matched features from frame j to frame k, where j is from video 0 and k is from video 1.
Matches = List[List[List[Position]]]
