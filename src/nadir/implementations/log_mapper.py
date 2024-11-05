import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..interfaces import ILogMapper
from ..models import DroneFrame, DroneState, GeoPoint, Timestamp, VideoSegment


class CSVLogMapper(ILogMapper):
    """CSV 형식의 드론 텔레메트리 로그를 처리하는 매퍼"""

    def __init__(self, log_path: str):
        """드론 텔레메트리 로그 파일을 로드하고 초기화

        Args:
            log_path: 텔레메트리 로그 파일 경로

        Raises:
            ValueError: 로그 파일 경로가 비어있거나 필수 컬럼이 누락된 경우
            FileNotFoundError: 로그 파일이 존재하지 않는 경우
        """
        self.logger = logging.getLogger(__name__)

        if not log_path:
            raise ValueError("로그 파일 경로가 필요합니다")

        self._log_data = self._load_log_file(log_path)
        self._segments = self._extract_segments()
        self.logger.info(
            f"로그 데이터에서 {len(self._segments)}개의 세그먼트를 추출했습니다"
        )

    def _load_log_file(self, log_path: str) -> pd.DataFrame:
        """드론 텔레메트리 로그 파일을 로드하고 검증"""
        try:
            df = pd.read_csv(log_path)
            self._validate_columns(df)
            return df
        except FileNotFoundError:
            self.logger.error(f"로그 파일을 찾을 수 없습니다: {log_path}")
            raise
        except Exception as e:
            self.logger.error(f"로그 파일 로드 실패: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """로그 파일의 필수 열 존재 여부 검증"""
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

    def map_videos_to_segments(self, video_paths: List[str]) -> Dict[str, VideoSegment]:
        """비디오 파일들을 해당하는 로그 세그먼트에 매핑

        Args:
            video_paths: 처리할 비디오 파일 경로 목록

        Returns:
            비디오 경로를 키로 하고 VideoSegment를 값으로 하는 딕셔너리

        Raises:
            ValueError: 비디오 경로가 비어있거나 세그먼트 수와 맞지 않는 경우
        """
        if not video_paths:
            raise ValueError("처리할 비디오 경로가 필요합니다")

        mapped_segments = {}
        if len(video_paths) != len(self._segments):
            self.logger.warning(
                f"비디오 파일의 개수 ({len(video_paths)})와 "
                f"세그먼트의 개수 ({len(self._segments)})가 일치하지 않습니다"
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

    def _extract_segments(self) -> List[VideoSegment]:
        """로그 데이터에서 비디오 세그먼트를 추출"""
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

            return segments
        except Exception as e:
            self.logger.error(f"세그먼트 추출 실패: {str(e)}")
            raise

    def _should_start_segment(self, is_video: int, currently_recording: bool) -> bool:
        """새로운 세그먼트 시작 여부 확인"""
        return is_video == 1 and not currently_recording

    def _should_end_segment(
        self, is_video: int, currently_recording: bool, is_last_row: bool
    ) -> bool:
        """현재 세그먼트 종료 여부 확인"""
        return (is_video == 0 or is_last_row) and currently_recording

    def _create_segment(self, start_idx: int, end_idx: int) -> VideoSegment:
        """로그 데이터 범위에서 비디오 세그먼트 생성"""
        try:
            segment_data = self._log_data.loc[start_idx:end_idx].copy()
            start_time = float(segment_data["time(millisecond)"].iloc[0])
            end_time = float(segment_data["time(millisecond)"].iloc[-1])

            return VideoSegment(
                start_time=start_time,
                end_time=end_time,
                log_data=segment_data,  # 이 부분은 Pydantic 모델에서 제거됨
            )
        except Exception as e:
            self.logger.error(f"세그먼트 생성 실패: {str(e)}")
            raise

    def _process_video_segment(
        self, video_path: str, segment: VideoSegment
    ) -> VideoSegment:
        """비디오 세그먼트 처리 및 프레임 데이터 생성"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 열기 실패: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            video_start_time = Path(video_path).stat().st_mtime * 1000
            frame_times = np.linspace(0, segment.duration_ms, frame_count)

            interpolated_data = self._interpolate_telemetry(self._log_data, frame_times)
            frames = self._create_frames(
                interpolated_data, frame_times, video_start_time
            )

            # 새로운 VideoSegment 객체 생성
            updated_segment = VideoSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                video_path=video_path,
                fps=fps,
                frames=frames,
            )

            return updated_segment

        except Exception as e:
            self.logger.error(f"비디오 세그먼트 처리 실패: {str(e)}")
            raise

    def _interpolate_telemetry(
        self, log_data: pd.DataFrame, frame_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """프레임 시간에 맞춰 텔레메트리 데이터 보간"""
        interpolated = {}

        # 일반 데이터 보간
        for col in ["latitude", "longitude", "altitude(feet)"]:
            interpolated[col] = self._interpolate_linear(
                log_data["time(millisecond)"].values, log_data[col].values, frame_times
            )

        # 방위각 특별 처리
        interpolated["compass_heading(degrees)"] = self._interpolate_compass(
            log_data["time(millisecond)"].values,
            log_data["compass_heading(degrees)"].values,
            frame_times,
        )

        return interpolated

    def _interpolate_compass(
        self, times: np.ndarray, values: np.ndarray, query_times: np.ndarray
    ) -> np.ndarray:
        """방위각 보간 (0/360도 경계 처리)"""
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
        """선형 보간 수행"""
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
        return f(x_new)

    def _create_frames(
        self,
        interpolated_data: Dict[str, np.ndarray],
        frame_times: np.ndarray,
        video_start_time: float,
    ) -> List[DroneFrame]:
        """프레임 데이터 생성"""
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

        return frames
