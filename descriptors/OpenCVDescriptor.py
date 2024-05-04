# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Tuple

import cv2
import numpy as np

from descriptors.Descriptor import Descriptor


class OpenCVDescriptor(Descriptor):
    """Class for using descriptors from OpenCV.

    Args:
        keypoint_detector: Detect keypoints on the image.
        feature_extractor: Describe image at detected keypoints.
    """

    def __init__(self, keypoint_detector: cv2.Feature2D, feature_extractor: cv2.Feature2D):
        self.keypoint_detector = keypoint_detector
        self.feature_extractor = feature_extractor

    def get_features(self, image: np.array) -> Tuple[np.array, np.array]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints = self.keypoint_detector.detect(gray)
        keypoints, descriptors = self.feature_extractor.compute(gray, keypoints)
        keypoints = np.array([kp.pt for kp in keypoints])
        return keypoints, descriptors

    def __str__(self) -> str:
        return type(self.feature_extractor).__name__
