# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import json
from pathlib import Path
from random import randint, uniform
from shutil import copy2
from typing import Any, Dict
from typing import Tuple

from moviepy.editor import VideoFileClip

from test_dataset.data_preparation.random_clip import get_random_clip
from test_dataset.data_preparation.utils import extend_annotations, augment_clip


class ClipAugmentations:
    """Augment clips by shifting the start, end, and modifying the speed in each part of the clip.

    Args:
        clip_dir: Directory containing action and replay clips.
        output_dir: Directory for saving generated dataset.
        max_shift: Maximum time shift in seconds.
        min_subclips: Minimum number of subclips. Speed is changed in each subclip.
        max_subclips: Maximum number of subclips. Speed is changed in each subclip.
        min_speed_change: Minimum speed change.
        max_speed_change: Maximum speed change.
        video_ext: Clip video extension.
        annotations_filename: Filename for information about used augmentations.
        augmented_clip_suffix: String to add to filename of augmented clips.
    """

    def __init__(self,
                 clip_dir: Path,
                 output_dir: Path,
                 max_shift: float,
                 min_subclips: int,
                 max_subclips: int,
                 min_speed_change: float,
                 max_speed_change: float,
                 video_ext="mp4",
                 annotations_filename: str = "annotations.json",
                 augmented_clip_suffix: str = "augmented"
                 ):
        self.clip_dir = clip_dir
        self.output_dir = output_dir
        self.max_shift = max_shift
        self.min_subclips = min_subclips
        self.max_subclips = max_subclips
        self.min_speed_change = min_speed_change
        self.max_speed_change = max_speed_change
        self.video_ext = video_ext
        self.annotations_filename = annotations_filename
        self.augmented_clip_suffix = augmented_clip_suffix

    def generate_dataset(self, size: int) -> None:
        """Generate dataset of augmented clips.

        Args:
            size: How many clips to augment.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        annotation_file = self.output_dir / self.annotations_filename
        if annotation_file.exists():
            with open(annotation_file) as fp:
                annotations = json.load(fp)
        else:
            annotations = {}

        for _ in range(size):
            self.generate_augmented_clip(annotations)

        with open(annotation_file, "w") as fp:
            json.dump(annotations, fp, indent=4)

    def generate_augmented_clip(self, annotations: Dict[str, Any]) -> None:
        """Pick a random clip, augment it and save information about the augmentation to annotations.

        Args:
            annotations: Annotations for augmented clips.
        """
        input_clip_path = get_random_clip(self.clip_dir)
        league, season, match = input_clip_path.parts[2:5]
        output_clip_dir = self.output_dir / league / season / match
        while output_clip_dir.exists():
            input_clip_path = get_random_clip(self.clip_dir)
            league, season, match = input_clip_path.parts[2:5]
            output_clip_dir = self.output_dir / league / season / match

        extend_annotations(annotations, league, season, match, "dict")

        clip_id = input_clip_path.stem
        output_original_clip_path = output_clip_dir / f"{clip_id}.{self.video_ext}"
        output_clip_dir.mkdir(parents=True, exist_ok=True)
        output_augmented_clip_path = output_clip_dir / f"{clip_id}_{self.augmented_clip_suffix}.{self.video_ext}"

        augmented_clip, augmented_clip_info = self.augment_clip(input_clip_path)
        annotations[league][season][match][clip_id] = augmented_clip_info

        augmented_clip.write_videofile(str(output_augmented_clip_path))
        copy2(input_clip_path, output_original_clip_path)

    def augment_clip(self, clip_path: Path) -> Tuple[VideoFileClip, Dict[str, Any]]:
        """Augment clip by shifting the start, end, and modifying the speed in each part of the clip.

        Args:
            clip_path: Path to clip.

        Returns:
            Tuple[VideoFileClip, Dict[str, Any]]: Augmented video file clip and information about augmentations.
        """
        clip = VideoFileClip(str(clip_path))

        start_shift = uniform(0, self.max_shift / 2)
        end_shift = uniform(0, self.max_shift / 2)

        n_subclips = randint(self.min_subclips, self.max_subclips)
        step = clip.duration / n_subclips
        start = 0
        end = step
        speed_augmentations = []
        for _ in range(n_subclips):
            speed_change_factor = uniform(self.min_speed_change, self.max_speed_change)
            speed_augmentations.append({"start": start + start_shift,
                                        "speed_change": speed_change_factor,
                                        "end": end + start_shift}
                                       )
            start = end
            end += step
        augmentation_info = {"start_shift": start_shift,
                             "end_shift": end_shift,
                             "speed_augmentations": speed_augmentations
                             }

        augmented_clip = augment_clip(clip, augmentation_info)
        return augmented_clip, augmentation_info
