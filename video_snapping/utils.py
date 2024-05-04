# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from video_snapping.types import Position


def query_identifier(video_0: Path, video_1: Path, fps: int, descriptor: str) -> str:
    """Identifier of a query.

    Args:
        video_0: Path to first video.
        video_1: Path to second video.
        fps: Fps for sampling.
        descriptor: Descriptor name.

    Returns:
        str: Identifier.
    """
    return f"{descriptor}_{fps}fps_{video_0.parts[-2]}_{video_0.name}__{video_1.parts[-2]}_{video_1.name}". \
        replace(".", "_")


def join_matching_frames(clip_0: VideoFileClip,
                         clip_1: VideoFileClip,
                         corresponding_frames: List[Position],
                         show_matching_features: bool = False,
                         feature_extractor: Optional[cv2.Feature2D] = None,
                         k_matches: Optional[int] = None,
                         ratio_test_threshold: Optional[float] = None
                         ) -> List[np.array]:
    """Create synchronized videos by joining corresponding frames.

    Args:
        clip_0: First clip.
        clip_1: Second clip.
        corresponding_frames: Corresponding frames from first and second clip.
        show_matching_features: Show matching keypoints between frames.
        feature_extractor: Feature extractor for displaying matching features.
        k_matches: 	Count of best matches found per each query descriptor.
        ratio_test_threshold: Threshold for ratio test for k neighbors.
    Returns:
        List[np.array]: Frames of synchronized video.
    """
    bf = None
    if show_matching_features:
        if feature_extractor is None:
            raise ValueError("Matching features cannot be shown if feature detector is set to None.")
        bf = cv2.BFMatcher()
    all_result_frames = []
    frames = list(clip_0.iter_frames()), list(clip_1.iter_frames())
    for i_0, i_1 in corresponding_frames:
        frame_0 = frames[0][i_0]
        frame_1 = frames[1][i_1]
        if show_matching_features:
            kp0, kp1, good = get_matches_with_keypoints(feature_extractor,
                                                        bf,
                                                        frame_0,
                                                        frame_1,
                                                        k_matches,
                                                        ratio_test_threshold
                                                        )
            result_frame = cv2.drawMatchesKnn(img1=frame_0,
                                              keypoints1=kp0,
                                              img2=frame_1,
                                              keypoints2=kp1,
                                              matches1to2=good,
                                              outImg=None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                              )
        else:
            result_frame = np.concatenate((frame_0, frame_1), axis=1)
        all_result_frames.append(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
    return all_result_frames


def frames_to_video(frames: List[np.array], output_path: Path, codec="mp4v", fps: int = 25) -> None:
    """Concatenate frames to create a video.

    Args:
        frames: Frames of video.
        output_path: Path to store synchronized video.
        codec: 4-character code of codec used to compress the frames.
        fps: Framerate of the created video stream.
    """
    if len(frames) == 0:
        return
    video_writer = cv2.VideoWriter(filename=str(output_path),
                                   fourcc=cv2.VideoWriter_fourcc(*codec),
                                   fps=fps,
                                   frameSize=(frames[0].shape[1], frames[0].shape[0]),
                                   )
    for frame in frames:
        video_writer.write(frame)
    cv2.destroyAllWindows()
    video_writer.release()


def get_matches_with_keypoints(feature_detector: cv2.Feature2D,
                               matcher: cv2.DescriptorMatcher,
                               frame_0: np.ndarray,
                               frame_1: np.ndarray,
                               k_matches: int,
                               ratio_test_threshold: float
                               ) -> Tuple[Tuple[cv2.KeyPoint], Tuple[cv2.KeyPoint], List[List[cv2.DMatch]]]:
    """Get matches and keypoints of matched descriptors between frames.

    Args:
        feature_detector: Feature detector.
        matcher: Matcher instance.
        frame_0: First frame for matching.
        frame_1: Second frame for matching.
        k_matches: 	Count of best matches found per each query descriptor.
        ratio_test_threshold: Threshold for ratio test for k neighbors.

    Returns:
        Tuple[Tuple[cv2.KeyPoint], Tuple[cv2.KeyPoint], List[List[cv2.DMatch]]]: Matched keypoints from first frame,
            second frame and list of matches between descriptors.
    """
    kp0, des0 = feature_detector.detectAndCompute(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), None)
    kp1, des1 = feature_detector.detectAndCompute(cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY), None)
    matches = matcher.knnMatch(des0, des1, k=k_matches)
    return kp0, kp1, [[m] for m, n in matches if m.distance < ratio_test_threshold * n.distance]


def resize_clip(clip: VideoFileClip, max_size: Tuple[int, int]) -> VideoFileClip:
    """Resize video clip.

    Args:
        clip: Input clip.
        max_size: Maximum size of the clip after resizing in (height, width) format.

    Returns:
        VideoFileClip: Resized file clip.
    """
    if clip.size[0] > max_size[0] or clip.size[1] > max_size[1]:
        return VideoFileClip(clip.filename, target_resolution=max_size, resize_algorithm="area")
    else:
        return clip


def get_clips_for_testing(test_clips_annotations: Path,
                          clips_dir: Path,
                          video_ext: str = "mp4"
                          ) -> Iterator[Tuple[Path, Path]]:
    """Get clips for testing using annotation file.

    Args:
        test_clips_annotations: Annotation file.
        clips_dir: Directory containing clips.
        video_ext: Clips video extension.

    Returns:
        Iterator[Tuple[Path, Path]]: Clips for synchronization pair.
    """
    with open(test_clips_annotations) as fp:
        videos = json.load(fp)

    for comp_name, seasons in videos["clips"].items():
        for season_name, matches in seasons.items():
            for match_name, clips in matches.items():
                for a, b in clips:
                    yield Path(clips_dir / comp_name / season_name / match_name / f"{a}.{video_ext}"), \
                          Path(clips_dir / comp_name / season_name / match_name / f"{b}.{video_ext}")


def trim_image(image: np.ndarray, trim_top: float, trim_bottom: float) -> np.ndarray:
    """Remove top and bottom parts of the image.

    Args:
        image: Image.
        trim_top: Fraction of image to trim from top.
        trim_bottom: Fraction of image to trim from bottom.

    Returns:
        np.ndarray: Image with top and bottom part removed.
    """
    if trim_top == 0 and trim_bottom == 0:
        return image
    if not image.flags.writeable:
        image = image.copy()
    height = image.shape[0]
    return image[int(height * trim_top): int(height - height * trim_bottom)]
