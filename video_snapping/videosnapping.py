# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from video_snapping.best_path.dijkstra import DijkstraBestPath
from video_snapping.cost_matrix.cost_matrix import CostMatrix
from video_snapping.types import Position
from video_snapping.utils import query_identifier, resize_clip


def synchronize_clips(cost_matrix: CostMatrix,
                      clip_0: Path,
                      clip_1: Path,
                      max_size: Optional[Tuple[int, int]],
                      fps: int,
                      dijkstra_best_path: DijkstraBestPath,
                      cache_dir: Optional[Path] = None,
                      ) -> Tuple[List[Position], np.ndarray]:
    """Synchronize two videos.

    Args:
        cost_matrix: Cost matrix to use for synchronization.
        clip_0: First clip for synchronization.
        clip_1: Second clip for synchronization.
        max_size: Maximum image size: (height, width). Do not resize if it is set to None.
        fps: Fps to sample video.
        dijkstra_best_path: Path finding object.
        cache_dir: Cache directory. If not provided, caching will not be used.

    Returns:
        Tuple[List[Position], np.ndarray]: Synchronization path and cost matrix.
    """
    query_id = query_identifier(clip_0, clip_1, fps, cost_matrix.get_feature_name())
    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / f"{query_id}.pkl"
    else:
        cache_file = None
    clip_0_full_size = VideoFileClip(str(clip_0))
    clip_1_full_size = VideoFileClip(str(clip_1))
    if cache_file is not None and cache_file.exists():
        with open(cache_file, "rb") as fp:
            cost_matrix = pickle.load(fp)
    else:
        clip_0 = clip_0_full_size
        clip_1 = clip_1_full_size
        if max_size:
            clip_0 = resize_clip(clip_0_full_size, max_size)
            clip_1 = resize_clip(clip_1_full_size, max_size)

        cost_matrix = cost_matrix.get_cost_matrix(clip_0, clip_1)
        total_frames = round(clip_0.duration * clip_0.fps), \
                       round(clip_1.duration * clip_1.fps)
        cost_matrix = cv2.resize(cost_matrix, total_frames[::-1], cv2.INTER_LINEAR)
        if cache_file is not None:
            with open(cache_file, "wb") as fp:
                pickle.dump(cost_matrix, fp)

    best_path = dijkstra_best_path.best_path(cost_matrix)
    return best_path, cost_matrix
