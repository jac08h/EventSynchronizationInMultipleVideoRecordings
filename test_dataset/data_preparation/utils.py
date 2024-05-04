# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Dict, Any

from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx


def extend_annotations(annotations: Dict[str, Any],
                       league: str,
                       season: str,
                       match: str,
                       container_type: str = "dict") -> None:
    """Add missing keys to annotations in-place.

    Args:
        annotations: Existing annotations.
        league: League name.
        season: Season name.
        match: Match name.
        container_type: Container type for match annotation.
    """
    container = None
    if container_type == "dict":
        container = dict()
    elif container_type == "list":
        container = []

    if league not in annotations:
        annotations[league] = {}
    if season not in annotations[league]:
        annotations[league][season] = {}
    if match not in annotations[league][season]:
        annotations[league][season][match] = container


def augment_clip(clip: VideoFileClip, augmentations_info: Dict[str, Any]) -> VideoFileClip:
    clip = clip.subclip(t_start=augmentations_info["start_shift"], t_end=-augmentations_info["end_shift"])
    subclips = []
    for item in augmentations_info["speed_augmentations"]:
        subclips.append(vfx.speedx(clip.subclip(item["start"], item["end"]), item["speed_change"]))
    return concatenate_videoclips(subclips)
