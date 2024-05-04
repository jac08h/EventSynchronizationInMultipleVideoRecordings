# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import argparse
from pathlib import Path

from cut_clips.clips_cutter import ClipCutter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("soccernet_dir", help="Path to soccernet directory.")
    parser.add_argument("clips_dir", help="Path to directory to store action and replay videos.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    soccernet_dir = Path(args.soccernet_dir)

    clips_cutter = ClipCutter(
        v3_annotations_dir=soccernet_dir,
        v2_annotations_dir=soccernet_dir,
        videos_dir=soccernet_dir,
        output_dir=Path(args.clips_dir),
        max_length_before_event=5000,
        max_length_after_event=3000,
        minimal_length=2000,
        camera_switch_buffer=500,
        video_resolution="224p"
    )
    clips_cutter.cut_all_clips_for_all_videos()
