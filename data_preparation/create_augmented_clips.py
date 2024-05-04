# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import argparse
import json
from pathlib import Path
from shutil import copy2

from moviepy.video.io.VideoFileClip import VideoFileClip

from test_dataset.data_preparation.utils import augment_clip


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("augmentations_annotations_file", help="Path to augmentation annotations.")
    parser.add_argument("clips_dir", help="Path to directory containing action and replay clips.")
    parser.add_argument("augmented_clips_dir", help="Path to directory to store augmented and copied clips.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(Path(args.augmentations_annotations_file)) as fp:
        augmentation_info = json.load(fp)
    augmented_clips_dir = Path(args.augmented_clips_dir)
    clips_dir = Path(args.clips_dir)

    for competition_name, seasons in augmentation_info.items():
        for season_name, matches in seasons.items():
            for match_name, events in matches.items():
                output_dir = augmented_clips_dir / competition_name / season_name / match_name
                output_dir.mkdir(parents=True, exist_ok=True)
                for event_id, augmentation_info in events.items():
                    clip_path = clips_dir / competition_name / season_name / match_name / f"{event_id}.mp4"
                    clip = VideoFileClip(str(clip_path))
                    augmented_clip = augment_clip(clip, augmentation_info)
                    augmented_clip.write_videofile(str(output_dir / f"{event_id}_augmented.mp4"))
                    copy2(clip_path, output_dir / f"{event_id}.mp4")
