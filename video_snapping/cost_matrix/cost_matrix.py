# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Dict, List, Union

import cv2
import numpy as np
from moviepy.video.io import VideoFileClip

from descriptors.Descriptor import Descriptor
from video_snapping.cost_matrix.histogram import paper_histogram, simplified_histogram
from video_snapping.cost_matrix.matching import match_features
from video_snapping.cost_matrix.utils import opencv_keypoints_and_matches
from video_snapping.types import HistogramMethod, Position, VideoFeatures
from video_snapping.utils import trim_image


class CostMatrix:
    """Cost matrix calculation following description in the VideoSnapping paper.
     Wang, Oliver, et al. "Videosnapping: Interactive synchronization of multiple videos."
     ACM Transactions on Graphics (TOG) 33.4 (2014): 1-10.

    Args:
        device: Device for nearest neighbor computation.
        descriptor: Feature descriptor.
        fps: FPS for sampling the videos.
        ignore_top: Top fraction of image to ignore when finding keypoints.
        ignore_bottom: Bottom fraction of image to ignore when finding keypoints.
        histogram_method: Method to use for histogram calculation.
        histogram_parameters: Parameters for histogram calculation.
        cost_matrix_parameters: Parameters for cost matrix calculation.
    """

    def __init__(self,
                 device: str,
                 descriptor: Descriptor,
                 fps: int,
                 ignore_top: float,
                 ignore_bottom: float,
                 histogram_method: HistogramMethod,
                 histogram_parameters: Dict[str, Union[bool, float]],
                 cost_matrix_parameters: Dict[str, float],
                 ):
        self.device = device
        self.descriptor = descriptor
        self.fps = fps
        self.ignore_top = ignore_top
        self.ignore_bottom = ignore_bottom
        self.histogram_method = histogram_method
        self.histogram_parameters = histogram_parameters
        self.cost_matrix_parameters = cost_matrix_parameters

    def get_feature_name(self) -> str:
        """Get name of feature that is extracted.

        Returns:
            str: Feature name.
        """
        return str(self.descriptor)

    def get_cost_matrix(self, clip_0: VideoFileClip, clip_1: VideoFileClip) -> np.ndarray:
        """Get cost matrix based on feature matches between frame pairs from two clips.

        Args:
            clip_0: First clip.
            clip_1: Second clip.

        Returns:
            np.ndarray: Cost matrix.
        """
        clip_0_frames = list(clip_0.iter_frames(fps=self.fps))
        clip_1_frames = list(clip_1.iter_frames(fps=self.fps))
        features = self.get_features_from_frames(clip_0_frames), self.get_features_from_frames(clip_1_frames)
        frame_count = len(clip_0_frames), len(clip_1_frames)
        matches = match_features(features, frame_count, self.device)

        if self.histogram_method == HistogramMethod.PAPER:
            histogram = paper_histogram(matches,
                                        features,
                                        frame_count,
                                        self.histogram_parameters["d_decay"],
                                        self.histogram_parameters["s_decay"],
                                        )
        else:
            histogram = simplified_histogram(matches, frame_count)

        return (1 - (histogram / np.max(histogram))) ** self.cost_matrix_parameters["alpha"]

    def get_features_from_frames(self,
                                 frames: List[np.ndarray],
                                 normalize_descriptors: bool = True,
                                 use_relative_coordinates: bool = True
                                 ) -> VideoFeatures:
        """Get features from frames.

        Args:
            frames: Video frames in RGB.
            normalize_descriptors: Normalize descriptors by their L2-norm.
            use_relative_coordinates: Use relative coordinates instead of absolute.

        Returns:
            VideoFeatures: Features in video.
        """
        all_descriptors = []
        all_coords = []
        width = frames[0].shape[1]
        height = frames[0].shape[0]
        for frame in frames:
            frame = trim_image(frame, self.ignore_top, self.ignore_bottom)
            frame_keypoints, frame_descriptors = self.descriptor.get_features(frame)
            all_descriptors.append(frame_descriptors)
            all_coords.append(frame_keypoints)

        feature_count = sum(len(i) for i in all_coords)
        frame_indexes = np.empty(feature_count, dtype=int)
        coords = np.empty((feature_count, 2))
        feature_dimension = len(all_descriptors[0][0])
        descriptors = np.empty((feature_count, feature_dimension))
        start_i = 0
        for frame_index, (frame_keypoints, frame_descriptors) in enumerate(zip(all_coords, all_descriptors)):
            end_i = start_i + len(frame_keypoints)
            frame_indexes[start_i: end_i] = frame_index
            coords[start_i: end_i] = frame_keypoints
            descriptors[start_i: end_i] = frame_descriptors
            start_i = end_i
        if normalize_descriptors:
            descriptors /= np.linalg.norm(descriptors, ord=2, axis=1, keepdims=True)
        if use_relative_coordinates:
            coords /= (width, height)
        return VideoFeatures(frame_indexes, coords, descriptors)

    def visualize_matches_in_clips(self,
                                   clip_0: VideoFileClip,
                                   clip_1: VideoFileClip,
                                   best_path: List[Position],
                                   ) -> List[np.ndarray]:
        """Visualize feature matches between frames in synchronized clips.

        Args:
            clip_0: First clip.
            clip_1: Second clip.
            best_path: Synchronization between two clips.

        Returns:
            List[np.ndarray]: Frames with visualized matches.
        """
        clip_0_frames = list(clip_0.iter_frames(self.fps))
        clip_1_frames = list(clip_1.iter_frames(self.fps))
        frames_with_keypoints = []
        features = self.get_features_from_frames(clip_0_frames, use_relative_coordinates=False), \
                   self.get_features_from_frames(clip_1_frames, use_relative_coordinates=False)
        frame_count = len(clip_0_frames), len(clip_1_frames)
        matches = match_features(features, frame_count, self.device)

        for i0, i1 in best_path:
            keypoints_0, keypoints_1, cv_matches = opencv_keypoints_and_matches(features, matches[i0][i1])
            joined_frame = cv2.drawMatches(img1=trim_image(clip_0_frames[i0], self.ignore_top, self.ignore_bottom),
                                           keypoints1=keypoints_0,
                                           img2=trim_image(clip_1_frames[i1], self.ignore_top, self.ignore_bottom),
                                           keypoints2=keypoints_1,
                                           matches1to2=cv_matches,
                                           outImg=None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                           )
            frames_with_keypoints.append(cv2.cvtColor(joined_frame, cv2.COLOR_BGR2RGB))
        return frames_with_keypoints
