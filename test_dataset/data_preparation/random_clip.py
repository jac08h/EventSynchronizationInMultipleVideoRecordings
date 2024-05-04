# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import json
from pathlib import Path
from random import choice
from typing import Dict, Tuple


def get_random_clip(clips_dir: Path, video_ext: str = "mp4") -> Path:
    """Get a random clip from all clips of actions and replays.

    Args:
        clips_dir: Clip directory.
        video_ext: Extension of clips.

    Returns:
        Path: Random clip path.
    """
    match, annotations = get_random_match_with_annotations(clips_dir)
    clip_id = choice(list(annotations.keys()))
    return match / f"{clip_id}.{video_ext}"


def get_random_action_replay_pair(clips_dir: Path, video_ext: str = "mp4") -> Tuple[Path, Path]:
    """Get random action and replay pair from all clips.

    Args:
        clips_dir: Clip directory.
        video_ext: Extension of clips.

    Returns:
        Tuple[Path, Path]: Action and replay clip.
    """
    match, annotations = get_random_match_with_annotations(clips_dir)
    action_ids = [key for key in annotations.keys() if "_" not in key]
    action_id = choice(action_ids)
    replay_ids = [key for key in annotations.keys() if key.startswith(f"{action_id}_")]
    replay_id = choice(replay_ids)
    return match / f"{action_id}.{video_ext}", match / f"{replay_id}.{video_ext}"


def get_random_match_with_annotations(clips_dir: Path) -> Tuple[Path, Dict]:
    """Get random match from all clips directory.

    Args:
        clips_dir: Clip directory.

    Returns:
        Tuple[Path, Dict]: Match path and annotations.
    """
    competition = choice(list(clips_dir.iterdir()))
    season = choice(list(competition.iterdir()))
    match = choice(list(season.iterdir()))
    with open(match / "annotations.json") as fp:
        return match, json.load(fp)
