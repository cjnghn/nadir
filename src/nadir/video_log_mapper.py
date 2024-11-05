import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class VideoSegment:
    """비디오 세그먼트를 나타내는 데이터 클래스"""

    start_time: float
    end_time: float
    log_data: pd.DataFrame
    frames: List[Dict] = field(default_factory=list)
    video_path: Optional[str] = None
    fps: float = 30.0

    @property
    def duration_ms(self) -> float:
        return self.end_time - self.start_time


class VideoLogMapper:
    """로그 데이터와 비디오 파일을 매핑하는 클래스"""

    def __init__(self, log_path: str):
        self.logger = logging.getLogger(f"{__name__}.VideoLogMapper")
        self.log_data = self._load_log_file(log_path)
        self.segments = self._extract_segments()

    def _load_log_file(self, log_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(log_path)
            required_columns = [
                "time(millisecond)",
                "latitude",
                "longitude",
                "altitude(feet)",
                "compass_heading(degrees)",
                "isVideo",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼이 누락됨: {missing_columns}")

            return df
        except Exception as e:
            self.logger.error(f"로그 파일 로드 중 오류: {str(e)}")
            raise

    def _extract_segments(self) -> List[VideoSegment]:
        """isVideo 플래그를 기반으로 비디오 세그먼트 추출"""
        segments = []
        is_recording = False
        segment_start_idx = None

        try:
            for idx, row in self.log_data.iterrows():
                if row["isVideo"] == 1 and not is_recording:
                    is_recording = True
                    segment_start_idx = idx
                elif (
                    row["isVideo"] == 0 or idx == len(self.log_data) - 1
                ) and is_recording:
                    is_recording = False
                    if segment_start_idx is not None:
                        segment_data = self.log_data.loc[segment_start_idx:idx].copy()
                        segment = VideoSegment(
                            start_time=segment_data["time(millisecond)"].iloc[0],
                            end_time=segment_data["time(millisecond)"].iloc[-1],
                            log_data=segment_data,
                        )
                        segments.append(segment)

            self.logger.info(f"로그 데이터에서 {len(segments)}개의 세그먼트 추출")
            return segments

        except Exception as e:
            self.logger.error(f"세그먼트 추출 중 오류: {str(e)}")
            raise

    def map_videos_to_segments(
        self, video_paths: List[str], method: str = "interpolate"
    ) -> Dict[str, VideoSegment]:
        """비디오 파일들을 로그 세그먼트와 매핑"""
        mapped_segments = {}

        if len(video_paths) != len(self.segments):
            self.logger.warning(
                f"비디오 파일 수({len(video_paths)})와 "
                f"세그먼트 수({len(self.segments)})가 일치하지 않습니다."
            )

        for video_path, segment in zip(video_paths, self.segments):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                segment.video_path = video_path
                segment.fps = cap.get(cv2.CAP_PROP_FPS)

                # 프레임 시간 생성
                duration_sec = segment.duration_ms / 1000
                frame_count = int(duration_sec * segment.fps)
                frame_times = np.linspace(
                    segment.start_time, segment.end_time, frame_count
                )

                # 로그 데이터 보간
                log_data = segment.log_data
                interpolated_data = {}

                # 일반 데이터 보간
                for col in ["latitude", "longitude", "altitude(feet)"]:
                    f = interp1d(
                        log_data["time(millisecond)"].values,
                        log_data[col].values,
                        kind="cubic",
                        bounds_error=False,
                        fill_value=(log_data[col].iloc[0], log_data[col].iloc[-1]),
                    )
                    interpolated_data[col] = f(frame_times)

                # 나침반 각도 특수 처리
                heading_values = log_data["compass_heading(degrees)"].values
                interpolated_data["compass_heading(degrees)"] = (
                    self._interpolate_compass(
                        log_data["time(millisecond)"].values,
                        heading_values,
                        frame_times,
                    )
                )

                # 프레임별 데이터 생성
                frames = []
                for i, frame_time in enumerate(frame_times):
                    frame_data = {
                        "frame_number": i + 1,
                        "timestamp": {
                            "milliseconds": int(frame_time),
                            "datetime_utc": datetime.fromtimestamp(
                                frame_time / 1000, tz=timezone.utc
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        "drone_data": {
                            "latitude": float(interpolated_data["latitude"][i]),
                            "longitude": float(interpolated_data["longitude"][i]),
                            "altitude_feet": float(
                                interpolated_data["altitude(feet)"][i]
                            ),
                            "compass_heading": float(
                                interpolated_data["compass_heading(degrees)"][i]
                            ),
                        },
                    }
                    frames.append(frame_data)

                segment.frames = frames
                mapped_segments[video_path] = segment
                cap.release()

            except Exception as e:
                self.logger.error(f"비디오 {video_path} 매핑 중 오류: {str(e)}")
                continue

        return mapped_segments

    def _interpolate_compass(
        self, times: np.ndarray, values: np.ndarray, query_times: np.ndarray
    ) -> np.ndarray:
        """나침반 각도에 대한 특수 보간"""
        diff = np.diff(values)
        wrap_points = np.where(abs(diff) > 180)[0]

        if len(wrap_points) > 0:
            unwrapped = np.copy(values).astype(float)
            for i in wrap_points:
                if diff[i] > 0:
                    unwrapped[i + 1 :] -= 360
                else:
                    unwrapped[i + 1 :] += 360

            f = interp1d(
                times,
                unwrapped,
                kind="cubic",
                bounds_error=False,
                fill_value=(unwrapped[0], unwrapped[-1]),
            )
            result = f(query_times)

            return result % 360
        else:
            f = interp1d(
                times,
                values,
                kind="cubic",
                bounds_error=False,
                fill_value=(values[0], values[-1]),
            )
            return f(query_times)
