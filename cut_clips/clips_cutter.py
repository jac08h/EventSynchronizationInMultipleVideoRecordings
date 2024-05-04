# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cut_clips.utils import cut_video, get_camera_switch_info, \
    get_camera_switch_timestamps, \
    get_events_annotations, get_video_path, \
    get_videos_with_fully_annotated_camera_switches, \
    match_identifier, \
    trim_clip_by_camera_switches


class ClipCutter:
    """Cut clips from SoccerNetV3 actions and corresponding replays.

    Args:
        v3_annotations_dir: SoccerNetv3 annotation directory.
        v2_annotations_dir: SoccerNetv2 annotation directory.
        videos_dir: Directory containing videos of the matches.
        output_dir: Directory for saving clips.
        max_length_before_event: Maximum time to cut before a real-time event in milliseconds.
        max_length_after_event: Maximum time to cut after a real-time event in milliseconds.
        minimal_length: Do not use the clip if its resulting length in milliseconds is smaller than this value.
        camera_switch_buffer: Buffer to use before/after camera switches to account for not exact annotations, in milliseconds.
        v3_labels_filename: Filename of SoccerNetv3 labels.
        camera_labels_filename: Filename of SoccerNetv2 camera labels.
        output_annotation_filename: Filename of output annotations about clips.
        input_video_ext: Extension of input video.
        output_video_ext: Extension of output video.
        video_resolution: Resolution of input video.
        skip_not_shown_events: Skip events that have visibility annotation set to "not shown".
        skip_existing_clips: Skip clip generation if the file already exists.
    """

    def __init__(self,
                 v3_annotations_dir: Path,
                 v2_annotations_dir: Path,
                 videos_dir: Path,
                 output_dir: Path,
                 max_length_before_event: int,
                 max_length_after_event: int,
                 minimal_length: int,
                 camera_switch_buffer: int,
                 v3_labels_filename: str = "Labels-v3.json",
                 camera_labels_filename: str = "Labels-cameras.json",
                 output_annotation_filename: str = "annotations.json",
                 input_video_ext: str = "mkv",
                 output_video_ext: str = "mp4",
                 video_resolution: str = "224p",
                 skip_not_shown_events: bool = True,
                 skip_existing_clips: bool = True):
        self.v3_annotations_dir = v3_annotations_dir
        self.v3_labels_filename = v3_labels_filename
        self.v2_annotations_dir = v2_annotations_dir
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.camera_labels_filename = camera_labels_filename
        self.max_length_before_event = max_length_before_event
        self.max_length_after_event = max_length_after_event
        self.minimal_length = minimal_length
        self.camera_switch_buffer = camera_switch_buffer
        self.output_annotation_filename = output_annotation_filename
        self.input_video_ext = input_video_ext
        self.output_video_ext = output_video_ext
        self.video_resolution = video_resolution
        self.skip_existing_clips = skip_existing_clips
        self.skip_not_shown_events = skip_not_shown_events
        self.fully_annotated_videos = get_videos_with_fully_annotated_camera_switches(
            ["train", "valid", "test"]
        )

    def cut_match_clips(self, competition: str, season: str, match: str) -> None:
        """Cut clips from a single match.

        Args:
            competition: Competition.
            season: Season.
            match: Match.
        """
        events_annotation_file = self.v3_annotations_dir / competition / season / match / self.v3_labels_filename
        camera_annotation_file = self.v2_annotations_dir / competition / season / match / self.camera_labels_filename
        match_output_dir = self.output_dir / competition / season / match
        match_output_dir.mkdir(parents=True, exist_ok=True)
        video_path = get_video_path(events_annotation_file)

        camera_switch_timestamps = get_camera_switch_timestamps(camera_annotation_file)
        camera_switch_info = get_camera_switch_info(camera_annotation_file)

        has_full_camera_annotations = match_identifier(competition, season, match) in self.fully_annotated_videos
        event_annotations = get_events_annotations(events_annotation_file)
        clip_annotations = {}
        for clip_id, event_annotations in event_annotations.items():
            output_video = match_output_dir / f"{clip_id}.{self.output_video_ext}"
            if (self.skip_not_shown_events and event_annotations["metadata"]["visibility"] == "not shown") or \
                    (self.skip_existing_clips and output_video.exists()):
                continue
            half = event_annotations["metadata"]["half"]
            input_video = self.videos_dir / video_path / f"{half}_{self.video_resolution}.{self.input_video_ext}"
            start, stop, camera_info = self.get_clip_info(event_annotations,
                                                          has_full_camera_annotations,
                                                          camera_switch_timestamps,
                                                          camera_switch_info)
            if stop - start >= self.minimal_length:
                clip_annotations[clip_id] = {
                    "clipStart": start,
                    "clipStop": stop,
                    "metadata": event_annotations["metadata"],
                    "camera_info": camera_info
                }
                cut_video(str(input_video), start / 1000, stop / 1000, str(output_video), verbose=False)
        new_annotation_file = self.output_dir / competition / season / match / self.output_annotation_filename
        with open(new_annotation_file, "w") as fp:
            json.dump(clip_annotations, fp, indent=4)

    def get_clip_info(self,
                      event_annotations: Dict[str, Any],
                      has_full_camera_annotations: bool,
                      camera_switch_timestamps: Dict[int, List[int]],
                      camera_switch_info: Dict[int, List[Dict[str, str]]]
                      ) -> Tuple[int, int, Optional[Dict]]:
        """Get information about the clip.

        Args:
            event_annotations: Action and replay annotations.
            has_full_camera_annotations: True if camera switches on video are fully annotated.
            camera_switch_timestamps: Timestamps of camera switches divided by two match halves.
            camera_switch_info: Information about camera switches by half.

        Returns:
            Tuple[int, int, Optional[Dict]]: Clip start, end, and camera info if available.
        """
        camera_info = None
        half = event_annotations["metadata"]["half"]
        is_action = event_annotations["metadata"]["imageType"] == "action"
        if is_action:
            start = event_annotations["metadata"]["position"] - self.max_length_before_event
            stop = event_annotations["metadata"]["position"] + self.max_length_after_event
        else:
            start = event_annotations["metadata"]["replayStart"] + self.camera_switch_buffer
            stop = event_annotations["metadata"]["replayStop"] - self.camera_switch_buffer
        if has_full_camera_annotations or not is_action:
            event_timestamp_tag = "position" if is_action else "replayPosition"
            event_timestamp = event_annotations["metadata"][event_timestamp_tag]
            camera_index = bisect.bisect_left(camera_switch_timestamps[half], event_timestamp)
            start, stop = trim_clip_by_camera_switches(start,
                                                       stop,
                                                       half,
                                                       camera_index,
                                                       camera_switch_timestamps,
                                                       self.camera_switch_buffer)
            camera_info = camera_switch_info[half][camera_index]
        return start, stop, camera_info

    def cut_all_clips_for_all_videos(self):
        """Cut clips for all videos."""
        for competition_dir in self.v3_annotations_dir.iterdir():
            for season_dir in competition_dir.iterdir():
                for match_dir in season_dir.iterdir():
                    try:
                        self.cut_match_clips(competition_dir.name, season_dir.name, match_dir.name)
                    except (FileNotFoundError, OSError):
                        pass
