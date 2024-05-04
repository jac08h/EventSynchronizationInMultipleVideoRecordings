# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Any, Dict, List

import numpy as np
from scipy.interpolate import interp1d

from video_snapping.types import Position


def get_speed_changes(clip_info: Dict[str, Any],
                      original_duration: float,
                      augmented_duration: float,
                      fps: int
                      ) -> List[Position]:
    """Calculate on which frames in original and augmented video speed changes.

    Args:
        clip_info: Augmentation information about the clip.
        original_duration: Duration of original clip.
        augmented_duration: Duration of augmented clip.
        fps: FPS of clips.

    Returns:
        List[Position]: Positions with speed changes.
    """
    speed_changes_timestamps = [(0, 0), (clip_info["start_shift"], 0)]
    for speed_aug in clip_info["speed_augmentations"]:
        last_time_in_orig, last_time_in_aug = speed_changes_timestamps[-1]
        duration_in_orig = speed_aug["end"] - speed_aug["start"]
        speed_changes_timestamps.append((last_time_in_orig + duration_in_orig,
                                         last_time_in_aug + (duration_in_orig / speed_aug["speed_change"]))
                                        )
    speed_changes_timestamps.append((original_duration, augmented_duration))
    speed_changes_frames = [(round(speed_change[0] * fps), round(speed_change[1] * fps))
                            for speed_change in speed_changes_timestamps]
    last_x, last_y = speed_changes_frames[-1]
    speed_changes_frames[-1] = (last_x - 1, last_y - 1)
    return speed_changes_frames


def sync_from_speed_changes(speed_changes: List[Position]) -> List[Position]:
    """Synchronize original and augmented video utilizing information
        about where there is a speed change in the augmented video.

    Args:
        speed_changes: Positions with speed changes.

    Returns:
        List[Position]: Frame-by-frame video synchronization.
    """
    xs = [i[0] for i in speed_changes]
    ys = [i[1] for i in speed_changes]
    interpolation = interp1d(xs, ys)
    return [(x, round(interpolation(x).item())) for x in range(xs[-1] + 1)]


def sync_from_corresponding_frames(synchronized_frames: List[Position],
                                   action_frames: int,
                                   replay_frames: int
                                   ) -> List[Position]:
    """Synchronize action and replay video utilizing manual annotations of a couple of corresponding frames.
        This is achieved by fitting a linear function to the annotated data by least-squares optimization.
        Padding is added so that each synchronization sequence starts in first frames of clips and
        ends in their last frames.

    Args:
        synchronized_frames: Frames from first and second video that display the same event.
        action_frames: Number of frames in action clip.
        replay_frames: Number of frames in replay clip.

    Returns:
        List[Position]: Frame-by-frame video synchronization.
    """
    xs = np.array([p[0] for p in synchronized_frames])
    ys = np.array([p[1] for p in synchronized_frames])
    fitted = np.linalg.lstsq(np.vstack([xs, np.ones(len(xs))]).T, ys, rcond=None)
    slope, intercept = fitted[0]

    gt_path = []

    for y in range(0, round(intercept)):
        gt_path.append((0, y))

    for x in range(action_frames):
        y = round(intercept + slope * x)
        gt_path.append((x, min(max(0, y), replay_frames - 1)))

    for y in range(gt_path[-1][1], replay_frames):
        gt_path.append((action_frames - 1, y))

    return gt_path
