import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .models import DroneFrame, DroneState, GeoPoint, Timestamp, VideoSegment


class DroneLogMapper:
    """Maps drone telemetry logs to video frames"""

    def __init__(self, log_path: str):
        self.logger = logging.getLogger(__name__)
        self._log_data = self._load_log_file(log_path)
        self._segments = self._extract_segments()

    def _load_log_file(self, log_path: str) -> pd.DataFrame:
        """Load and validate drone telemetry log file"""
        try:
            df = pd.read_csv(log_path)
            self._validate_columns(df)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load log file: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns in log file"""
        required_columns = [
            "time(millisecond)",
            "latitude",
            "longitude",
            "altitude(feet)",
            "compass_heading(degrees)",
            "isVideo",
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _extract_segments(self) -> List[VideoSegment]:
        """Extract video segments from log data"""
        segments = []
        is_recording = False
        segment_start_idx = None

        try:
            for idx, row in self._log_data.iterrows():
                if self._should_start_segment(row["isVideo"], is_recording):
                    is_recording = True
                    segment_start_idx = idx
                elif self._should_end_segment(
                    row["isVideo"], is_recording, idx == len(self._log_data) - 1
                ):
                    if segment_start_idx is not None:
                        segment = self._create_segment(segment_start_idx, idx)
                        segments.append(segment)
                    is_recording = False

            self.logger.info(f"Extracted {len(segments)} segments from log data")
            return segments
        except Exception as e:
            self.logger.error(f"Failed to extract segments: {str(e)}")
            raise

    def _should_start_segment(self, is_video: int, currently_recording: bool) -> bool:
        """Check if a new segment should start"""
        return is_video == 1 and not currently_recording

    def _should_end_segment(
        self, is_video: int, currently_recording: bool, is_last_row: bool
    ) -> bool:
        """Check if current segment should end"""
        return (is_video == 0 or is_last_row) and currently_recording

    def _create_segment(self, start_idx: int, end_idx: int) -> VideoSegment:
        """Create a video segment from log data range"""
        segment_data = self._log_data.loc[start_idx:end_idx].copy()
        return VideoSegment(
            start_time=segment_data["time(millisecond)"].iloc[0],
            end_time=segment_data["time(millisecond)"].iloc[-1],
            log_data=segment_data,
        )

    def map_videos_to_segments(self, video_paths: List[str]) -> Dict[str, VideoSegment]:
        """Map video files to log segments with interpolated telemetry data"""
        mapped_segments = {}

        if len(video_paths) != len(self._segments):
            self.logger.warning(
                f"Video count ({len(video_paths)}) does not match "
                f"segment count ({len(self._segments)})"
            )

        for video_path, segment in zip(video_paths, self._segments):
            try:
                mapped_segments[video_path] = self._process_video_segment(
                    video_path, segment
                )
            except Exception as e:
                self.logger.error(f"Failed to map video {video_path}: {str(e)}")
                continue

        return mapped_segments

    def _process_video_segment(
        self, video_path: str, segment: VideoSegment
    ) -> VideoSegment:
        """Process single video segment with interpolated telemetry"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Generate frame timestamps
        video_start_time = Path(video_path).stat().st_mtime * 1000  # to milliseconds
        frame_times = np.linspace(0, segment.duration_ms, frame_count)

        # Interpolate telemetry data
        interpolated_data = self._interpolate_telemetry(segment.log_data, frame_times)

        # Create frame data
        frames = []
        for i, rel_time in enumerate(frame_times):
            abs_time = int(video_start_time + rel_time)

            frame = DroneFrame(
                number=i + 1,
                timestamp=Timestamp(
                    epoch_ms=abs_time,
                    relative_ms=int(rel_time),
                    datetime_utc=pd.Timestamp(abs_time, unit="ms").strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )[:-3],
                ),
                drone_state=DroneState(
                    position=GeoPoint(
                        latitude=float(interpolated_data["latitude"][i]),
                        longitude=float(interpolated_data["longitude"][i]),
                    ),
                    altitude_feet=float(interpolated_data["altitude(feet)"][i]),
                    heading_degrees=float(
                        interpolated_data["compass_heading(degrees)"][i]
                    ),
                ),
            )
            frames.append(frame)

        segment.video_path = video_path
        segment.fps = fps
        segment.frames = frames

        return segment

    def _interpolate_telemetry(
        self, log_data: pd.DataFrame, frame_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Interpolate telemetry data for video frames"""
        interpolated = {}

        # Regular data interpolation
        for col in ["latitude", "longitude", "altitude(feet)"]:
            interpolated[col] = self._interpolate_linear(
                log_data["time(millisecond)"].values, log_data[col].values, frame_times
            )

        # Special handling for compass heading
        interpolated["compass_heading(degrees)"] = self._interpolate_compass(
            log_data["time(millisecond)"].values,
            log_data["compass_heading(degrees)"].values,
            frame_times,
        )

        return interpolated

    def _interpolate_compass(
        self, times: np.ndarray, values: np.ndarray, query_times: np.ndarray
    ) -> np.ndarray:
        """Interpolate compass heading with special handling for 0/360 wraparound"""
        diff = np.diff(values)
        wrap_points = np.where(abs(diff) > 180)[0]

        if len(wrap_points) > 0:
            unwrapped = values.copy()
            for i in wrap_points:
                if diff[i] > 0:
                    unwrapped[i + 1 :] -= 360
                else:
                    unwrapped[i + 1 :] += 360

            return self._interpolate_linear(times, unwrapped, query_times) % 360
        else:
            return self._interpolate_linear(times, values, query_times)

    @staticmethod
    def _interpolate_linear(
        x: np.ndarray, y: np.ndarray, x_new: np.ndarray
    ) -> np.ndarray:
        """Perform linear interpolation with boundary handling"""
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
        return f(x_new)
