# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from pathlib import Path
from typing import Iterator, List, Tuple

from moviepy.editor import VideoFileClip

from test_dataset.data_loaders.ClipLoader import ClipLoader
from test_dataset.data_loaders.utils import sync_from_corresponding_frames
from video_snapping.types import Position


class AnnotatedClipLoader(ClipLoader):
    """Load synchronization of action and replay clips using manual annotations.

    Args:
        clip_directory: Directory containing clips.
        annotations_file: File with manual annotations of corresponding frames.
        name: Name of the dataset.
        fps: FPS in clips.
        clip_ext: Clip extension.
    """

    def __init__(self,
                 clip_directory: Path,
                 annotations_file: Path,
                 name: str = "annotated",
                 fps: int = 25,
                 clip_ext: str = "mp4"):
        super().__init__(clip_directory, annotations_file, name, fps, clip_ext)

    def get_synced_clips(self) -> Iterator[Tuple[Path, Path, List[Position]]]:
        for league_name, league in self.annotations.items():
            for season_name, season in league.items():
                for match_name, match in season.items():
                    for clips_info in match:
                        match_dir = self.clip_directory / league_name / season_name / match_name
                        action_path = match_dir / f"{clips_info['action']}.{self.clip_ext}"
                        replay_path = match_dir / f"{clips_info['replay']}.{self.clip_ext}"

                        action_frames = round(VideoFileClip(str(action_path)).duration * self.fps)
                        replay_frames = round(VideoFileClip(str(replay_path)).duration * self.fps)

                        synchronized_frames = [(round(first_ts * self.fps), round(second_ts * self.fps))
                                               for first_ts, second_ts in clips_info["corresponding_timestamps"]]
                        synchronized_frames = sync_from_corresponding_frames(synchronized_frames,
                                                                             action_frames,
                                                                             replay_frames
                                                                             )
                        yield action_path, replay_path, synchronized_frames
