# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Tuple

import torch

from video_snapping.cost_matrix.utils import nearest_neighbors
from video_snapping.types import Matches, VideoFeatures


def match_features(features: Tuple[VideoFeatures, VideoFeatures],
                   frame_count: Tuple[int, int],
                   device: str,
                   ) -> Matches:
    """Build a set of matches between features from two videos using method specified in VideoSnapping paper.

    Args:
        features: Features from the two videos.
        frame_count: Number of frames in the two videos.
        device: Device to use for distance calculation.

    Returns:
        Matches: Matches between frames.
    """
    matches: Matches = [[list() for _ in range(frame_count[1])] for _ in range(frame_count[0])]
    descriptors_0 = torch.tensor(features[0].descriptors, device=device)
    descriptors_1 = torch.tensor(features[1].descriptors, device=device)
    for a, b in nearest_neighbors(descriptors_0, descriptors_1):
        j = features[0].frame_indexes[a]
        k = features[1].frame_indexes[b]
        matches[j][k].append((a, b))
    for b, a in nearest_neighbors(descriptors_1, descriptors_0):
        j = features[0].frame_indexes[a]
        k = features[1].frame_indexes[b]
        matches[j][k].append((a, b))
    return matches
