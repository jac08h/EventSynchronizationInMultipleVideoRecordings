# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Optional, Tuple

import numpy as np

from video_snapping.cost_matrix.utils import decay
from video_snapping.types import Matches, VideoFeatures


def paper_histogram(matches: Matches,
                    features: Tuple[VideoFeatures, VideoFeatures],
                    frame_count: Tuple[int, int],
                    d_decay: Optional[float],
                    s_decay: Optional[float],
                    ) -> np.ndarray:
    """Build histogram from matching features between frames from two videos. For more info see section 3.1
        in the paper, up to equation (2).

    Args:
        matches: Matches between frames.
        features: Features from the two videos.
        frame_count: Number of frames in the two videos.
        d_decay: Decay rate for descriptor differences. Do not use descriptor differences if set to None.
        s_decay: Decay rate for spatial differences. Do not use spatial differences if set to None.

    Returns:
        np.ndarray: Matrix in shape [A, B] where A, B is number of frames in the first and second video,
            respectively. Value at coordinates [a, b] stands for how good is the match between frame `a`
            from the first video and frame `b` from the second video.

    Raises:
        ValueError: d_decay and s_decay set to None.
    """
    if d_decay is None and s_decay is None:
        raise ValueError("d_decay and s_decay set to None. At least one has to be used.")
    histogram = np.empty(frame_count)
    for j in range(frame_count[0]):
        for k in range(frame_count[1]):
            a_indexes = [i[0] for i in matches[j][k]]
            b_indexes = [i[1] for i in matches[j][k]]

            descriptor_difference = np.linalg.norm(
                features[0].descriptors[a_indexes] - features[1].descriptors[b_indexes],
                ord=1,
                axis=1
            )
            coordinates_difference = np.linalg.norm(features[0].coords[a_indexes] - features[1].coords[b_indexes],
                                                    ord=1,
                                                    axis=1
                                                    )
            # Equation (1) in VideoSnapping paper.
            if d_decay is not None and s_decay is not None:
                histogram[j, k] = (decay(d_decay, descriptor_difference) *
                                   decay(s_decay, coordinates_difference)
                                   ).sum()
            elif d_decay is not None:
                histogram[j, k] = decay(d_decay, descriptor_difference).sum()
            else:
                histogram[j, k] = decay(s_decay, descriptor_difference).sum()
    return histogram


def simplified_histogram(matches: Matches, frame_count: Tuple[int, int]) -> np.ndarray:
    """Build histogram only by counting feature matches between frames.

    Args:
        matches: Matches between frames.
        frame_count: Number of frames in the two videos.

    Returns:
        np.ndarray: Matrix in shape [A, B] where A, B is number of frames in the first and second video,
            respectively. Value at coordinates [a, b] stands for how many matches are there between frame `a` from
            the first video and frame `b` from the second video.
    """
    histogram = np.empty(frame_count)
    for j in range(frame_count[0]):
        for k in range(frame_count[1]):
            histogram[j][k] = len(matches[j][k])
    return histogram
