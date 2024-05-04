# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from typing import Iterator

import cv2
import yaml

from descriptors.KorniaDescriptor import KorniaDescriptor
from descriptors.OpenCVDescriptor import OpenCVDescriptor
from test_dataset.data_loaders.AnnotatedClipLoader import AnnotatedClipLoader
from test_dataset.data_loaders.AugmentedClipLoader import AugmentedClipLoader
from test_dataset.data_loaders.ClipLoader import ClipLoader
from tests.config.utils import has_keypoint_detector, initialize_opencv_descriptor, initialize_third_party_descriptor, \
    parse_error_rate_unit, parse_test_metric, parse_visualization
from tests.config.utils import initialize_kornia_descriptor
from video_snapping.best_path.dijkstra import DijkstraBestPath
from video_snapping.cost_matrix.cost_matrix import CostMatrix
from video_snapping.types import ErrorRateUnit, TestMetric, Visualization
from video_snapping.types import HistogramMethod


def check_value(value: str, allowed_values: List[str]) -> str:
    """Return value only if it is in list of allowed values.

    Args:
        value: Value to check.
        allowed_values: Allowed values.

    Returns:
        str: Value if it is allowed.

    Raises:
        ValueError: Value is not allowed.
    """
    if value in allowed_values:
        return value
    raise ValueError(f"Invalid configuration value: {value}. Allowed values: {allowed_values}.")


def check_value_with_none(value: Optional[str], allowed_values: List[str]) -> Optional[str]:
    """Return value if it is None or it is in list of allowed values.

    Args:
        value: Value to check.
        allowed_values: Allowed values.

    Returns:
        str: Value if it is allowed.

    Raises:
        ValueError: Value is not None and is not one of allowed values.
    """
    if value is None:
        return None
    return check_value(value, allowed_values)


class ConfigWrapper:
    """Wrapper to extract configuration parameters.

    Args:
        config_file: Path to configuration file.
    """

    def __init__(self, config_file: Path):
        with open(config_file) as fp:
            self.config = yaml.safe_load(fp)
        self.output_directory = self.date_specified_output_directory()

    def cost_matrices_config(self) -> Dict:
        return self.config["cost_matrices"]

    def opencv_config(self) -> Dict:
        return self.cost_matrices_config()["opencv"]

    def third_party_config(self) -> Dict:
        return self.cost_matrices_config()["third_party"]

    def kornia_config(self) -> Dict:
        return self.cost_matrices_config()["kornia"]

    def results_config(self) -> Dict:
        return self.config["results"]

    def augmentations_config(self) -> Dict:
        return self.config["augmentations"]

    def clip_loaders_config(self) -> Dict:
        return self.tests_config()["clip_loaders"]

    def augmented_clip_loaders_config(self) -> Dict:
        return self.clip_loaders_config()["augmented"]

    def annotated_clip_loaders_config(self) -> Dict:
        return self.clip_loaders_config()["annotated"]

    def path_finding_config(self) -> Dict:
        return self.config["path_finding"]

    def partial_overlap_config(self) -> Dict:
        return self.path_finding_config()["partial_overlap"]

    def warping_constraints_config(self) -> Dict:
        return self.path_finding_config()["warping_constraints"]

    def tests_config(self) -> Dict:
        return self.config["tests"]

    def correct_frames_metric_config(self) -> Dict:
        return self.tests_config()["correct_frames"]

    def distances_metric_config(self) -> Dict:
        return self.tests_config()["distances"]

    def demo_config(self) -> Dict:
        return self.config["demo"]

    def histogram_method(self) -> HistogramMethod:
        if self.cost_matrices_config()["histogram_method"] == "paper":
            return HistogramMethod.PAPER
        if self.cost_matrices_config()["histogram_method"] == "simplified":
            return HistogramMethod.SIMPLIFIED

    def histogram_parameters(self) -> Dict[str, Union[bool, float]]:
        return self.cost_matrices_config()["histogram_parameters"]

    def cost_matrix_parameters(self) -> Dict[str, float]:
        return self.cost_matrices_config()["cost_matrix_parameters"]

    def clip_0(self) -> Path:
        return Path(self.demo_config()["clip_0"])

    def clip_1(self) -> Path:
        return Path(self.demo_config()["clip_1"])

    def device(self) -> str:
        return check_value(self.cost_matrices_config()["device"],
                           ["cpu", "cuda"])

    def fps(self) -> int:
        return self.cost_matrices_config()["fps"]

    def use_cache(self) -> bool:
        return self.cost_matrices_config()["use_cache"]

    def cache_directory(self) -> Path:
        return Path(self.cost_matrices_config()["cache_directory"])

    def use_partial_overlap(self) -> bool:
        return self.partial_overlap_config()["use"]

    def min_path_length(self) -> int:
        return self.partial_overlap_config()["min_path_length"]

    def use_warping_constraints(self) -> bool:
        return self.warping_constraints_config()["use"]

    def path_length_for_slope_estimation(self) -> int:
        return self.warping_constraints_config()["path_length_for_slope_estimation"]

    def min_slope(self):
        return self.warping_constraints_config()["min_slope"]

    def max_slope(self):
        return self.warping_constraints_config()["max_slope"]

    def penalty(self):
        return self.warping_constraints_config()["penalty"]

    def max_size(self) -> Optional[Tuple[int, int]]:
        return self.augmentations_config().get("max_size")

    def save_plots(self) -> bool:
        return self.results_config()["save_plots"]

    def clip_ext(self) -> str:
        return self.results_config()["clip_ext"]

    def plot_ext(self) -> str:
        return self.results_config()["plot_ext"]

    def colormap(self) -> str:
        return self.results_config()["colormap"]

    def clip_loader_names(self) -> List[str]:
        return self.clip_loaders_config()["use"]

    def default_keypoint_detector(self) -> str:
        return self.opencv_config()["default_keypoint_detector"]

    def opencv_features(self) -> List[str]:
        return self.opencv_config()["use"]

    def third_party_features(self) -> List[str]:
        return self.third_party_config()["use"]

    def kornia_features(self) -> List[str]:
        return self.kornia_config()["use"]

    def kornia_max_sift_features(self) -> Optional[int]:
        return self.kornia_config()["max_sift_features"]

    def use_affnet(self) -> bool:
        return self.kornia_config()["use_affnet"]

    def visualization(self) -> Visualization:
        return parse_visualization(self.results_config()["visualization"])

    def test_metric(self) -> TestMetric:
        return parse_test_metric(self.tests_config()["metric"])

    def allowed_error_rates(self) -> List[int]:
        return self.correct_frames_metric_config()["allowed_error_rates"]

    def error_rate_unit(self) -> ErrorRateUnit:
        return parse_error_rate_unit(self.correct_frames_metric_config()["unit"])

    def normalize_distances(self) -> bool:
        return self.distances_metric_config()["normalize"]

    def save_elapsed_time(self) -> bool:
        return self.tests_config()["save_elapsed_time"]

    def clip_loaders(self) -> Iterator[ClipLoader]:
        for clip_loader_name in self.clip_loader_names():
            yield self.initialize_clip_loader(clip_loader_name)

    def initialize_clip_loader(self, clip_loader_name: str) -> ClipLoader:
        if clip_loader_name not in self.clip_loaders_config():
            raise ValueError(f"{clip_loader_name} dataset not supported.")

        clip_loader_class = AugmentedClipLoader if clip_loader_name == "augmented" else AnnotatedClipLoader
        return clip_loader_class(clip_directory=Path(self.clip_loaders_config()[clip_loader_name]["clip_directory"]),
                                 annotations_file=Path(
                                     self.clip_loaders_config()[clip_loader_name]["annotations_file"]),
                                 name=clip_loader_name
                                 )

    def initialize_dijkstra(self) -> DijkstraBestPath:
        return DijkstraBestPath(self.use_partial_overlap(),
                                self.min_path_length(),
                                self.use_warping_constraints(),
                                self.path_length_for_slope_estimation(),
                                self.min_slope(),
                                self.max_slope(),
                                self.penalty()
                                )

    def initialize_keypoint_detector_and_feature_extractor(self, feature: str) -> Tuple[cv2.Feature2D, cv2.Feature2D]:
        """Initialize keypoint detector and feature extractor from OpenCV. They are the same thing if the extractor
            is also able to detect keypoints, otherwise use default keypoint detector.

        Args:
            feature: Name of the feature.

        Returns:
            Tuple[cv2.Feature2D, cv2.Feature2D]: Detector and extractor.
        """
        feature_extractor = initialize_opencv_descriptor(feature)
        if has_keypoint_detector(feature):
            keypoint_detector = feature_extractor
        else:
            keypoint_detector = initialize_opencv_descriptor(self.default_keypoint_detector())
        return keypoint_detector, feature_extractor

    def cost_matrices(self) -> Iterator[CostMatrix]:
        """Initialize cost matrices specified in configuration file.

        Returns:
            Iterator[CostMatrix]: Initialized costr matrices.
        """
        descriptors = []
        for feature in self.opencv_features():
            keypoint_detector, feature_extractor = self.initialize_keypoint_detector_and_feature_extractor(feature)
            descriptors.append(OpenCVDescriptor(keypoint_detector, feature_extractor))

        for descriptor_name in self.third_party_features():
            descriptor_config = self.third_party_config()[descriptor_name]
            descriptors.append(initialize_third_party_descriptor(descriptor_name, self.device(), descriptor_config))

        for descriptor_name in self.kornia_features():
            descriptors.append(KorniaDescriptor(initialize_kornia_descriptor(descriptor_name),
                                                self.use_affnet(),
                                                self.device(),
                                                self.kornia_max_sift_features()
                                                ))

        for descriptor in descriptors:
            yield CostMatrix(self.device(),
                             descriptor,
                             self.fps(),
                             self.augmentations_config()["ignore_top"],
                             self.augmentations_config()["ignore_bottom"],
                             self.histogram_method(),
                             self.histogram_parameters(),
                             self.cost_matrix_parameters()
                             )

    def date_specified_output_directory(self, format_string: str = "%y-%m-%d_%H-%M-%S"):
        return Path(self.results_config()["output_directory"]) / f"{datetime.now().strftime(format_string)}"
