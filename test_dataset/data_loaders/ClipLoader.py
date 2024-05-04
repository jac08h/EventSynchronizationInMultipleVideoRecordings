# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Tuple

from video_snapping.types import Position


class ClipLoader(ABC):
    """Base class for loading clips with ground truth data.

    Args:
        clip_directory: Directory containing clips.
        annotations_file: File with synchronization annotations.
        name: Name of the dataset.
        fps: FPS in clips.
        clip_ext: Clip extension.
    """

    def __init__(self, clip_directory: Path, annotations_file: Path, name: str, fps: int = 25, clip_ext: str = "mp4"):
        self.clip_directory = clip_directory
        self.name = name
        self.fps = fps
        self.clip_ext = clip_ext
        with open(annotations_file) as fp:
            self.annotations = json.load(fp)

    @abstractmethod
    def get_synced_clips(self) -> Iterator[Tuple[Path, Path, List[Position]]]:
        """Get synchronized clips.

        Returns:
            Iterator[Tuple[Path, Path, List[Position]]]: Original clip, augmented clip and their synchronization.
        """
        pass
