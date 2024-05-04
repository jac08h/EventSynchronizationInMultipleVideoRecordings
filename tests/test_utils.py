# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

from video_snapping.cost_matrix.cost_matrix import CostMatrix
from video_snapping.types import Position, Visualization
from video_snapping.utils import frames_to_video, join_matching_frames, resize_clip


def evaluate_predictions(ground_truth: List[Position], predicted: List[Position]) -> np.array:
    """Evaluate predicted synchronization against ground truth.

    Args:
        ground_truth: Ground truth synchronization.
        predicted: Predicted synchronization.

    Returns:
        np.array: Smallest difference between ground truth and prediction for each frame from first video.
    """
    ground_truth_lookup = defaultdict(set)
    for x, y in ground_truth:
        ground_truth_lookup[x].add(y)

    predicted_lookup = defaultdict(set)
    for x in range(0, predicted[0][0]):
        predicted_lookup[x].add(0)
    for x in range(predicted[-1][0], ground_truth[-1][0] + 1):
        predicted_lookup[x].add(ground_truth[-1][1])
    for x, y in predicted:
        predicted_lookup[x].add(y)

    differences = []
    last_frame_x, _ = ground_truth[-1]
    for x in range(last_frame_x + 1):
        ground_truth_ys = ground_truth_lookup[x]
        predicted_ys = predicted_lookup[x]
        dist = min((abs(gt_y - pr_y)) for (gt_y, pr_y) in product(ground_truth_ys, predicted_ys))
        differences.append(dist)

    return np.array(differences, dtype=float)


def plot_synchronization(ground_truth: List[Position],
                         predicted: List[Position],
                         cost_matrix: Optional[np.ndarray],
                         output_path: Path,
                         predicted_label: str = "predicted",
                         ground_truth_label: str = "ground truth",
                         predicted_color: str = "blue",
                         ground_truth_color: str = "green",
                         legend_location: str = "upper right",
                         colormap: str = "hot",
                         ) -> matplotlib.figure.Figure:
    """Plot predicted synchronization against ground truth as a path through frame matrix.

    Args:
        ground_truth: Ground truth synchronization.
        predicted: Predicted synchronization.
        cost_matrix: Cost matrix object to use for synchronization.
        output_path: Path to output file of the plot.
        predicted_label: Label for predicted data.
        ground_truth_label: Label for ground truth data.
        predicted_color: Color of predicted path.
        ground_truth_color: Color of ground truth path.
        legend_location: Legend location in graph.
        colormap: Cost matrix colormap.
    """
    plt.clf()
    plt.gca().invert_yaxis()
    plt.plot(*zip(*predicted), label=predicted_label, color=predicted_color)
    plt.plot(*zip(*ground_truth), label=ground_truth_label, color=ground_truth_color)
    if cost_matrix is not None:
        plt.imshow(cost_matrix.transpose((1, 0)), cmap=colormap)
        color_bar = plt.colorbar()
        color_bar.set_label("Node cost")

    plt.legend(loc=legend_location)
    plt.xlabel("First video frames")
    plt.ylabel("Second video frames")
    plt.savefig(str(output_path))


def plot_evaluation_results(results: pd.DataFrame,
                            metric: str = "mean",
                            plot_annotated: bool = True,
                            plot_augmented: bool = True,
                            results_axis_range: Tuple[float, float] = (0, 0.5),
                            plot_elapsed_time: bool = True,
                            time_axis_range: Tuple[int, int] = (0, 100)
                            ) -> matplotlib.figure.Figure:
    """Plot results of evaluation of different descriptors.

    Args:
        results: Evaluation results.
        metric: Which metric to plot: mean or median.
        plot_annotated: Plot results on annotated dataset.
        plot_augmented: Plot results on augmented dataset.
        results_axis_range: Axis range for plotting results.
        plot_elapsed_time: Plot mean elapsed time per video.
        time_axis_range: Axis range for plotting elapsed time.

    Returns:
        matplotlib.figure.Figure: Figure with plotted results.

    Raises:
        ValueError: Incorrect metric argument.
    """
    if metric != "mean" and metric != "median":
        raise ValueError(f"Incorrect metric {metric}. Options: mean, median.")
    if not (plot_annotated or plot_augmented):
        raise ValueError(f"No data to plot, pick at least one from annotated/augmented.")
    fig, ax = plt.subplots()
    ax.set_ylim(results_axis_range)
    if plot_augmented:
        plt.scatter(results.index, results["augmented"][metric], label="augmented", color="red")
    if plot_annotated:
        plt.scatter(results.index, results["annotated"][metric], label="annotated", color="orange")
    ax.legend(loc="upper right")
    ax.set_ylabel(f"{metric.capitalize()} error")

    if plot_elapsed_time:
        ax2 = ax.twinx()
        ax2.set_ylim(time_axis_range)
        ax2.scatter(results.index, results["annotated"]["elapsed"], label="time spent", color="blue", alpha=0.15)
        ax2.legend(loc="upper left")
        ax2.set_ylabel("Time spent (s)")

    return fig


def visualize_synchronized_clips(cost_matrix: CostMatrix,
                                 visualization: Visualization,
                                 clip_filename: str,
                                 output_dir: Path,
                                 clip_0: Path,
                                 clip_1: Path,
                                 predicted: List[Position],
                                 max_size: Tuple[int, int],
                                 clip_ext: str = "mp4",
                                 ) -> None:
    """Visualize synchronized clips, optionally also with matched keypoints.

    Args:
        cost_matrix: Cost matrix to use for synchronization.
        visualization: How to visualize synchronized videos.
        clip_filename: Filename of the clip.
        output_dir: Directory for saving results.
        clip_0: Path to first clip.
        clip_1: Path to second clip.
        predicted: Predicted synchronization.
        max_size: Maximum image size: (height, width).
        clip_ext: Output clip extension.
    """
    if visualization == Visualization.NONE:
        return
    clip_visualization_path = output_dir / f"{clip_filename}.{clip_ext}"
    clip_0 = VideoFileClip(str(clip_0))
    clip_1 = VideoFileClip(str(clip_1))

    if visualization == Visualization.PLAIN:
        frames = join_matching_frames(clip_0, clip_1, predicted)
    else:
        clip_0_resized = resize_clip(clip_0, max_size)
        clip_1_resized = resize_clip(clip_1, max_size)
        frames = cost_matrix.visualize_matches_in_clips(clip_0_resized, clip_1_resized, predicted)
    frames_to_video(frames, clip_visualization_path)
