# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from pathlib import Path
from typing import Iterator, List, Tuple

from moviepy.editor import VideoFileClip

from test_dataset.data_loaders.ClipLoader import ClipLoader
from test_dataset.data_loaders.utils import get_speed_changes, sync_from_speed_changes
from video_snapping.types import Position


class AugmentedClipLoader(ClipLoader):
    """Load synchronization information of clip and the same clip with augmentations.

    Args:
        clip_directory: Directory containing clips.
        annotations_file: File containing information about clip augmentations.
        name: Name of the dataset.
        fps: FPS in clips.
        clip_ext: Clip extension.
        augmented_clip_suffix: Suffix added to a clip to distinguish it from the original clip.
    """

    def __init__(self,
                 clip_directory: Path,
                 annotations_file: Path,
                 name: str = "augmented",
                 fps: int = 25,
                 clip_ext: str = "mp4",
                 augmented_clip_suffix: str = "augmented"):
        super().__init__(clip_directory, annotations_file, name, fps, clip_ext)
        self.augmented_clip_suffix = augmented_clip_suffix

    def get_synced_clips(self) -> Iterator[Tuple[Path, Path, List[Position]]]:
        for league_name, league in self.annotations.items():
            for season_name, season in league.items():
                for match_name, match in season.items():
                    for clip_id, clip_info in match.items():
                        match_directory_path = self.clip_directory / league_name / season_name / match_name
                        original_clip_path = match_directory_path / f"{clip_id}.{self.clip_ext}"
                        augmented_clip_path = match_directory_path / f"{clip_id}_{self.augmented_clip_suffix}.{self.clip_ext}"

                        original_duration = VideoFileClip(str(original_clip_path)).duration
                        augmented_duration = VideoFileClip(str(augmented_clip_path)).duration

                        speed_changes = get_speed_changes(clip_info, original_duration, augmented_duration, self.fps)
                        synchronized_frames = sync_from_speed_changes(speed_changes)
                        yield original_clip_path, augmented_clip_path, synchronized_frames
