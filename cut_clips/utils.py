# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import json
from os import sep
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from SoccerNet.utils import getListGames
from moviepy.config import get_setting
from moviepy.tools import subprocess_call


def get_videos_with_fully_annotated_camera_switches(splits: List[str]) -> Set[str]:
    """Get videos where all camera switches are annotated in camera annotation file.

    Args:
        splits: Which splits to use for obtaining videos.

    Returns:
        Set[str]: Identifier of videos with fully annotated camera switches.
    """
    videos = set()
    for split in splits:
        game_info = [game.split(sep) for game in getListGames(task="camera-changes", split=split)]
        for competition, season, match in game_info:
            videos.add(match_identifier(competition, season, match))

    return videos


def match_identifier(competition: str, season: str, match: str) -> str:
    """Return match identifier.

    Args:
        competition: Competition.
        season: Season.
        match: Match.

    Returns:
        str: Match identifier.
    """
    return f"{competition} - {season} - {match}"


def get_events_annotations(event_annotation_file: Path) -> Dict[str, Any]:
    """Get action and replay annotations from SoccerNetv3 annotation file.

    Args:
        event_annotation_file: SoccerNetv3 annotation file.

    Returns:
        Dict[str, Any]: annotations.
    """
    with open(event_annotation_file) as fp:
        info = json.load(fp)
    clip_info = {}
    for action_frame, action_info in info["actions"].items():
        action_id = action_frame.split(".")[0]
        clip_info[action_id] = {
            "metadata": action_info["imageMetadata"],
            "linked_replays": [replay_filename.split(".")[0] for replay_filename in action_info["linked_replays"]]
        }
    for replay_frame, replay_info in info["replays"].items():
        replay_id = replay_frame.split(".")[0]
        clip_info[replay_id] = {
            "metadata": replay_info["imageMetadata"],
            "linked_action": replay_info["linked_action"].split(".")[0]
        }
    return clip_info


def get_video_path(annotation_file: Path) -> Path:
    """Get video path from SoccerNetv3 annotation file.

    Args:
        annotation_file: SoccerNetv3 annotation file.

    Returns:
        Path: Video path.
    """
    with open(annotation_file) as fp:
        info = json.load(fp)
    return Path(info["GameMetadata"]["UrlLocal"])


def get_camera_switch_timestamps(camera_annotation_file: Path) -> Dict[int, List[int]]:
    """Get positions of camera switches from SoccerNetv2 annotation file.

    Args:
        camera_annotation_file: SoccerNetv2 camera annotation file.

    Returns:
        Dict[int, List[int]]: Timestamps of camera switches divided by two match halves.
    """
    with open(camera_annotation_file) as fp:
        cameras = json.load(fp)
    camera_timestamps = {1: [], 2: []}
    for camera_info in cameras["annotations"]:
        half = int(camera_info["gameTime"].split(" - ")[0])
        camera_timestamps[half].append(int(camera_info["position"]))
    return camera_timestamps


def get_camera_switch_info(camera_annotation_file: Path) -> Dict[int, List[Dict[str, str]]]:
    """Get camera switch information from SoccerNetv2 annotation file.

    Args:
        camera_annotation_file: SoccerNetv2 camera annotation file.

    Returns:
        Dict[int, List[Dict[str, str]]]: Information about camera switches by half.
    """
    with open(camera_annotation_file) as fp:
        cameras = json.load(fp)
    camera_infos = {1: [], 2: []}
    for camera_info in cameras["annotations"]:
        half = int(camera_info["gameTime"].split(" - ")[0])
        camera_infos[half].append(camera_info)
    return camera_infos


def trim_clip_by_camera_switches(start: int,
                                 stop: int,
                                 half: int,
                                 camera_index: int,
                                 camera_switch_timestamps: Dict[int, List[int]],
                                 camera_switch_buffer: int,
                                 ) -> Tuple[int, int]:
    """Trim clip to contain view only from one camera using information about camera switches.

    Args:
        start: Clip start in milliseconds.
        stop: Clip stop in milliseconds.
        half: Match half.
        camera_index: Index of camera switch after the clip.
        camera_switch_timestamps: Timestamps of camera switches divided by two match halves.
        camera_switch_buffer: Buffer to use before/after camera switches to account for not exact annotations,
            in milliseconds.

    Returns:
        Tuple[int, int]: Start and stop timestamps of trimmed clip.
    """
    camera_switch_after_event = camera_switch_timestamps[half][camera_index] - camera_switch_buffer
    if camera_index != 0:
        camera_switch_before_event = camera_switch_timestamps[half][camera_index - 1] + camera_switch_buffer
    else:
        camera_switch_before_event = 0
    start = max(start, camera_switch_before_event)
    stop = min(stop, camera_switch_after_event)
    return start, stop


def cut_video(video: str, start: float, end: float, output: str, verbose: bool) -> None:
    """Cut video by timestamps.

    Args:
        video: Input video.
        start: Clip start in seconds.
        end: Clip end in seconds.
        output: Path to output file.
        verbose: Show logs.
    """
    cmd = [get_setting("FFMPEG_BINARY"), "-y",
           "-ss", "%0.2f" % start,
           "-to", "%0.2f" % end,
           "-i", video,
           "-map", "0:v",  # video only, avoid subtitle errors
           output]
    logger = "bar" if verbose else None
    subprocess_call(cmd, logger=logger)
