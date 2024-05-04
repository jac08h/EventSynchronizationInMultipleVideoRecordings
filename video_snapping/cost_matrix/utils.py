# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Iterator, List, Tuple

import cv2
import numpy as np
import torch

from video_snapping.types import VideoFeatures


def decay(weight: float, vector: np.ndarray) -> np.ndarray:
    """Apply decaying weighting function to a vector.

    More details can be seen in the paper, see Equation (2).

    Args:
        weight: Decay control rate.
        vector: 1D vector.

    Returns:
        np.ndarray: Decayed vector.
    """
    return np.exp(-weight * vector)


def nearest_neighbors(queries: torch.Tensor,
                      database: torch.Tensor,
                      queries_split_size: int = 1000
                      ) -> Iterator[Tuple[int, int]]:
    """For each element in `queries` find its nearest neighbor in `database`.

    Args:
        queries: Tensor in (N, dimension) shape.
        database: Tensor in (M, dimension) shape.
        queries_split_size: Part size for working on parts of queries tensor to avoid memory error.

    Returns:
        Iterator[Tuple[int, int]]: Index of query element and index of its nearest neighbor in database.
    """
    queries_parts = torch.split(queries, queries_split_size)
    for part_index, queries_part in enumerate(queries_parts):
        distance_matrix = torch.cdist(queries_part, database)
        query_indexes = torch.arange(distance_matrix.shape[0], dtype=torch.int) + (part_index * queries_split_size)
        database_indexes = distance_matrix.argmin(dim=1)
        for a, b in zip(query_indexes.tolist(), database_indexes.tolist()):
            yield a, b


def opencv_keypoints_and_matches(features: Tuple[VideoFeatures, VideoFeatures],
                                 matches: List[Tuple[int, int]]
                                 ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    """Create keypoints and matches in OpenCV format.

    Args:
        features: Features from two videos.
        matches: Matches between the two frames.

    Returns:
        Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]: Keypoints from first frame, keypoints from
            second frame, matches.
    """
    keypoints_0 = []
    keypoints_1 = []
    cv_matches = []
    for match_index, (feature_i_0, feature_i_1) in enumerate(matches):
        x_0, y_0 = features[0].coords[feature_i_0]
        x_1, y_1 = features[1].coords[feature_i_1]
        keypoints_0.append(cv2.KeyPoint(x_0, y_0, size=1))
        keypoints_1.append(cv2.KeyPoint(x_1, y_1, size=1))
        cv_matches.append(cv2.DMatch(match_index, match_index, _distance=1))
    return keypoints_0, keypoints_1, cv_matches
