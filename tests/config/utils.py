# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Any, Dict, Optional

import cv2
import kornia.feature as KF
import torch.nn as nn

from descriptors.D2NetDescriptor import D2NetDescriptor
from descriptors.Descriptor import Descriptor
from descriptors.R2D2Descriptor import R2D2Descriptor
from descriptors.SuperPointDescriptor import SuperPointDescriptor
from video_snapping.types import ErrorRateUnit, TestMetric, Visualization


def initialize_opencv_descriptor(descriptor: str) -> cv2.Feature2D:
    """Initialize feature descriptor from OpenCV library.

    Args:
        descriptor: Descriptor name.

    Returns:
        cv2.Feature2D: Feature descriptor object.
    """
    if descriptor == "SIFT":
        return cv2.SIFT_create()
    elif descriptor == "ORB":
        return cv2.ORB_create()
    elif descriptor == "KAZE":
        return cv2.KAZE_create()
    elif descriptor == "BRISK":
        return cv2.BRISK_create()
    elif descriptor == "BoostDesc":
        return cv2.xfeatures2d.BoostDesc_create()
    elif descriptor == "DAISY":
        return cv2.xfeatures2d.DAISY_create()
    elif descriptor == "FREAK":
        return cv2.xfeatures2d.FREAK_create()
    elif descriptor == "LATCH":
        return cv2.xfeatures2d.LATCH_create()
    elif descriptor == "VGG":
        return cv2.xfeatures2d.VGG_create()


def has_keypoint_detector(descriptor: str) -> bool:
    """Determine if a feature descriptor also provides keypoint detector.

    Args:
        descriptor: Descriptor name.

    Returns:
        bool: True if descriptor provides keypoint detector.
    """
    return descriptor in ["SIFT", "ORB", "KAZE"]


def initialize_third_party_descriptor(descriptor: str, device: str, model_config: Dict[str, Any]) -> Descriptor:
    """Initialize neural descriptor using image-matching-toolbox library.

    Args:
        descriptor: Descriptor name.
        device: Device to use for computation.
        model_config: Descriptor configuration.

    Returns:
        Descriptor: Initialized descriptor.

    Raises:
        ValueError: Invalid descriptor name.
    """
    if descriptor == "D2Net":
        return D2NetDescriptor(device, model_config)
    elif descriptor == "SuperPoint":
        return SuperPointDescriptor(device, model_config)
    elif descriptor == "R2D2":
        return R2D2Descriptor(device, model_config)
    else:
        raise ValueError(f"'{descriptor}') descriptor not supported.")


def initialize_kornia_descriptor(descriptor: str) -> nn.Module:
    """Initialize kornia descriptor.

    Args:
        descriptor: Descriptor name.

    Returns:
        nn.Module: Initialized descriptor from kornia library.

    Raises:
        ValueError: Invalid descriptor name.
    """
    if descriptor == "MKDDescriptor":
        return KF.MKDDescriptor()
    elif descriptor == "TFeat":
        return KF.TFeat(True)
    elif descriptor == "HardNet":
        return KF.HardNet(True)
    elif descriptor == "HardNet8":
        return KF.HardNet8(True)
    elif descriptor == "HyNet":
        return KF.HyNet(True)
    elif descriptor == "SOSNet":
        return KF.SOSNet(True)
    else:
        raise ValueError(f"'{descriptor}') descriptor not supported.")


def parse_visualization(visualization: Optional[str]) -> Visualization:
    """Parse user-defined visualizaiton.

    Args:
        visualization: Visualization type.

    Returns:
        Visualization: Parsed visualization.

    Raises:
        ValueError: Invalid visualization.
    """
    if visualization is None:
        return Visualization.NONE
    elif visualization == "plain":
        return Visualization.PLAIN
    elif visualization == "keypoints":
        return Visualization.WITH_KEYPOINTS
    else:
        raise ValueError(f"Visualization type '{visualization}' not supported.")


def parse_test_metric(metric: str) -> TestMetric:
    """Parse user-defined test metric.

    Args:
        metric: Test metric type.

    Returns:
        Visualization: Parsed test metric.

    Raises:
        ValueError: Invalid metric.
    """
    if metric == "differences":
        return TestMetric.DIFFERENCES
    elif metric == "correct_frames":
        return TestMetric.CORRECT_FRAMES
    else:
        raise ValueError(f"Test metric '{metric}' not supported.")


def parse_error_rate_unit(error_rate_unit: str) -> ErrorRateUnit:
    """Parse user-defined error rate unit.

    Args:
        error_rate_unit: Error rate unit.

    Returns:
        Visualization: Parsed error rate unit.

    Raises:
        ValueError: Invalid unit.
    """
    if error_rate_unit == "second":
        return ErrorRateUnit.SECOND
    elif error_rate_unit == "frame":
        return ErrorRateUnit.FRAME
    else:
        raise ValueError(f"Error rate unit '{error_rate_unit}' not supported.")
