# src/nadir/log_mapper.py
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .models import DroneFrame, DroneState, GeoPoint, Timestamp, VideoSegment


class DroneLogMapper:
    """드론 텔레메트리 로그를 비디오 프레임에 매핑합니다"""

    def __init__(self, log_path: str):
        self.logger = logging.getLogger(__name__)
        self._log_data = self._load_log_file(log_path)
        self._segments = self._extract_segments()

    def _load_log_file(self, log_path: str) -> pd.DataFrame:
        """드론 텔레메트리 로그 파일을 로드하고 검증합니다"""
        try:
            df = pd.read_csv(log_path)
            self._validate_columns(df)
            return df
        except Exception as e:
            self.logger.error(f"로그 파일 로드 실패: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """로그 파일에 필요한 열을 검증합니다"""
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
            raise ValueError(f"필요한 열이 누락되었습니다: {missing}")

    def _extract_segments(self) -> List[VideoSegment]:
        """로그 데이터에서 비디오 세그먼트를 추출합니다"""
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

            self.logger.info(
                f"로그 데이터에서 {len(segments)}개의 세그먼트를 추출했습니다"
            )
            return segments
        except Exception as e:
            self.logger.error(f"세그먼트 추출 실패: {str(e)}")
            raise

    def _should_start_segment(self, is_video: int, currently_recording: bool) -> bool:
        """새 세그먼트가 시작해야 하는지 확인합니다"""
        return is_video == 1 and not currently_recording

    def _should_end_segment(
        self, is_video: int, currently_recording: bool, is_last_row: bool
    ) -> bool:
        """현재 세그먼트가 끝나야 하는지 확인합니다"""
        return (is_video == 0 or is_last_row) and currently_recording

    def _create_segment(self, start_idx: int, end_idx: int) -> VideoSegment:
        """로그 데이터 범위에서 비디오 세그먼트를 생성합니다"""
        segment_data = self._log_data.loc[start_idx:end_idx].copy()
        return VideoSegment(
            start_time=segment_data["time(millisecond)"].iloc[0],
            end_time=segment_data["time(millisecond)"].iloc[-1],
            log_data=segment_data,
        )

    def map_videos_to_segments(self, video_paths: List[str]) -> Dict[str, VideoSegment]:
        """비디오 파일을 로그 세그먼트에 맞춰 매핑합니다"""
        mapped_segments = {}

        if len(video_paths) != len(self._segments):
            self.logger.warning(
                f"비디오 파일의 개수 ({len(video_paths)})와 세그먼트의 개수 ({len(self._segments)})가 일치하지 않습니다"
            )

        for video_path, segment in zip(video_paths, self._segments):
            try:
                mapped_segments[video_path] = self._process_video_segment(
                    video_path, segment
                )
            except Exception as e:
                self.logger.error(f"비디오 {video_path} 매핑 실패: {str(e)}")
                continue

        return mapped_segments

    def _process_video_segment(
        self, video_path: str, segment: VideoSegment
    ) -> VideoSegment:
        """단일 비디오 세그먼트를 처리합니다"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 열기 실패: {video_path}")

        # 비디오 메타데이터 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 프레임 타임스탬프 생성
        video_start_time = Path(video_path).stat().st_mtime * 1000  # 밀리초로 변환
        frame_times = np.linspace(0, segment.duration_ms, frame_count)

        # 텔레메트리 데이터 보간
        interpolated_data = self._interpolate_telemetry(segment.log_data, frame_times)

        # 프레임 데이터 생성
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
        """비디오 프레임에 대한 텔레메트리 데이터를 보간합니다"""
        interpolated = {}

        # 일반 데이터 보간
        for col in ["latitude", "longitude", "altitude(feet)"]:
            interpolated[col] = self._interpolate_linear(
                log_data["time(millisecond)"].values, log_data[col].values, frame_times
            )

        # 방위각에 대한 특별한 처리
        interpolated["compass_heading(degrees)"] = self._interpolate_compass(
            log_data["time(millisecond)"].values,
            log_data["compass_heading(degrees)"].values,
            frame_times,
        )

        return interpolated

    def _interpolate_compass(
        self, times: np.ndarray, values: np.ndarray, query_times: np.ndarray
    ) -> np.ndarray:
        """방위각을 보간합니다. 0/360도 경계 처리 포함"""
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
        """선형 보간을 수행합니다. 경계 처리 포함"""
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
        return f(x_new)
