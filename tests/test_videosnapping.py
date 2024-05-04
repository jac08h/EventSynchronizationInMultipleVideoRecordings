# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import pickle
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from test_dataset.data_loaders.ClipLoader import ClipLoader
from tests.config.config_wrapper import ConfigWrapper
from tests.test_utils import evaluate_predictions, plot_synchronization, visualize_synchronized_clips
from video_snapping.cost_matrix.cost_matrix import CostMatrix
from video_snapping.types import ErrorRateUnit, Position, TestMetric
from video_snapping.utils import query_identifier
from video_snapping.videosnapping import synchronize_clips


class TestVideosnapping:
    """Test videosnapping performance with different configurations on ground truth data.

    Args:
        config_path: Path to configuration file.
    """

    def __init__(self, config_path: Path):
        self.config_wrapper = ConfigWrapper(config_path)

    def evaluate(self, results_filename: str = "results") -> pd.DataFrame:
        """Evaluate videosnapping on both datasets.

        Args:
            results_filename: Filename for saving results.

        Returns:
            pd.DataFrame: Evaluation results.
        """
        results: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        results_df = pd.DataFrame()
        for cost_matrix in self.config_wrapper.cost_matrices():
            cost_matrix_name = cost_matrix.get_feature_name()
            for clip_loader in self.config_wrapper.clip_loaders():
                print(f"{cost_matrix.get_feature_name()} - {clip_loader.name}")
                output_dir = self.config_wrapper.output_directory / clip_loader.name / cost_matrix_name
                dataset_results = self.evaluate_dataset(cost_matrix, clip_loader, output_dir)
                results[clip_loader.name][cost_matrix_name] = dataset_results

            results_df = pd.concat([pd.DataFrame.from_dict(clip_loader_results, orient="index")
                                    for clip_loader_results in results.values()],
                                   axis=1,
                                   keys=[name for name in results.keys()]
                                   )

            print(results_df)
            with open(self.config_wrapper.output_directory / f"{results_filename}.pkl", "wb") as fp:
                pickle.dump(results_df, fp)
            with open(self.config_wrapper.output_directory / f"{results_filename}.txt", "w") as fp:
                fp.write(results_df.to_string())
        with open(self.config_wrapper.output_directory / "config.yaml", "w") as fp:
            yaml.dump(self.config_wrapper.config, fp, default_flow_style=False)
        return results_df

    def evaluate_dataset(self,
                         cost_matrix: CostMatrix,
                         clip_loader: ClipLoader,
                         output_dir: Path
                         ) -> Dict[str, float]:
        """Evaluate performance on clip dataset.

        Args:
            cost_matrix: Cost matrix to use for synchronization.
            clip_loader: Class for loading clips with ground truth.
            output_dir: Directory for saving results.

        Returns:
            Dict[str, float]: Dataset results. The format depends on the used metric.
                differences: `mean` and `median` of the differences between prediction and ground truth.
                correct_frames: Raito of frames synchronized correctly for each allowed error rate.
                    If the error rates are [1, 3, 5] the dictionary will contain keys `1`, `3`, and `5`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        differences = []
        elapsed_times = []
        metric = self.config_wrapper.test_metric()
        for clip_0, clip_1, ground_truth in tqdm(clip_loader.get_synced_clips()):
            clip_differences, elapsed_time = self.evaluate_clip_pair(cost_matrix,
                                                                     clip_0,
                                                                     clip_1,
                                                                     ground_truth,
                                                                     output_dir)

            if self.config_wrapper.save_elapsed_time():
                elapsed_times.append(elapsed_time)

            if metric == TestMetric.DIFFERENCES:
                max_dist = ground_truth[-1][1]
                clip_differences /= max_dist
            else:
                if self.config_wrapper.error_rate_unit() == ErrorRateUnit.SECOND:
                    clip_differences /= VideoFileClip(str(clip_1)).fps
            differences.append(clip_differences)

        results = {}
        differences_np = np.concatenate(differences)
        if metric == TestMetric.DIFFERENCES:
            results["mean"] = np.mean(differences_np)
            results["median"] = np.median(differences_np)
        else:
            for allowed_error_rate in self.config_wrapper.allowed_error_rates():
                correct_frames = (differences_np <= allowed_error_rate).sum()
                results[str(allowed_error_rate)] = correct_frames / len(differences_np)

        if self.config_wrapper.save_elapsed_time():
            results["elapsed"] = sum(elapsed_times) / len(elapsed_times)
        return results

    def evaluate_clip_pair(self,
                           cost_matrix: CostMatrix,
                           clip_0: Path,
                           clip_1: Path,
                           ground_truth: List[Position],
                           output_dir: Path,
                           ) -> Tuple[np.array, float]:
        """Try to synchronize two clips and evaluate results against ground truth.

        Args:
            cost_matrix: Cost matrix to use for synchronization.
            clip_0: Path to first clip.
            clip_1: Path to second clip.
            ground_truth: Ground truth synchronization.
            output_dir: Directory for saving results.

        Returns:
            Tuple[np.array, float]: Differences for each frame between prediciton and ground truth and elapsed time.
        """
        start = time()
        cache_dir = self.config_wrapper.cache_directory() if self.config_wrapper.use_cache() else None
        predicted, calculated_cost_matrix = synchronize_clips(cost_matrix=cost_matrix,
                                                              clip_0=clip_0,
                                                              clip_1=clip_1,
                                                              max_size=self.config_wrapper.max_size(),
                                                              fps=self.config_wrapper.fps(),
                                                              dijkstra_best_path= \
                                                                  self.config_wrapper.initialize_dijkstra(),
                                                              cache_dir=cache_dir
                                                              )
        elapsed_time = time() - start
        differences = evaluate_predictions(ground_truth, predicted)
        identifier = query_identifier(clip_0, clip_1, self.config_wrapper.fps(), cost_matrix.get_feature_name())

        if self.config_wrapper.save_plots():
            graph_path = output_dir / f"{identifier}.{self.config_wrapper.plot_ext()}"
            plot_synchronization(ground_truth,
                                 predicted,
                                 calculated_cost_matrix,
                                 graph_path,
                                 colormap=self.config_wrapper.colormap()
                                 )
        visualize_synchronized_clips(cost_matrix,
                                     self.config_wrapper.visualization(),
                                     identifier,
                                     output_dir,
                                     clip_0,
                                     clip_1,
                                     predicted,
                                     self.config_wrapper.max_size(),
                                     self.config_wrapper.clip_ext())

        return differences, elapsed_time
