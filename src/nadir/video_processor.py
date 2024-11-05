import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from .geo_calculator import GeoCalculator
from .video_log_mapper import VideoLogMapper


class IntegratedVideoProcessor:
    """통합된 비디오 처리 시스템"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model = YOLO(model_path)
        self.logger = logging.getLogger(f"{__name__}.IntegratedVideoProcessor")
        self.config = config
        self.geo_calculator = GeoCalculator(config)

    def process_videos(self, video_paths: List[str], log_path: str) -> Dict[str, Dict]:
        """
        로그 파일과 비디오들을 처리하여 종합적인 결과 생성

        Args:
            video_paths (List[str]): 처리할 비디오 파일 경로 리스트
            log_path (str): 드론 로그 파일 경로

        Returns:
            Dict[str, Dict]: 각 비디오의 처리 결과
        """
        try:
            # 비디오-로그 매핑
            mapper = VideoLogMapper(log_path)
            mapped_segments = mapper.map_videos_to_segments(video_paths)

            final_results = {}
            for video_path, segment in mapped_segments.items():
                try:
                    # 비디오 메타데이터 추출
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        continue

                    metadata = {
                        "total_frames": len(segment.frames),
                        "duration_seconds": segment.duration_ms / 1000,
                        "fps": segment.fps,
                        "resolution": {
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        },
                    }

                    tracks = {}
                    class_counts = {}
                    total_confidence = 0
                    detection_count = 0

                    processed_frames = []

                    # 프레임별 처리
                    for frame_idx, frame_data in enumerate(segment.frames):
                        success, frame = cap.read()
                        if not success:
                            break

                        # YOLO로 객체 감지
                        results_yolo = self.model.track(frame, persist=True)
                        frame_detections = []

                        if results_yolo[0].boxes.id is not None:
                            for box, track_id, cls, conf in zip(
                                results_yolo[0].boxes.xywh.cpu(),
                                results_yolo[0].boxes.id.int().cpu().tolist(),
                                results_yolo[0].boxes.cls.int().cpu().tolist(),
                                results_yolo[0].boxes.conf.cpu().tolist(),
                            ):
                                x, y, w, h = box.tolist()
                                class_name = self.model.names[cls]

                                # 지리적 위치 계산
                                geo_location = (
                                    self.geo_calculator.calculate_geo_location(
                                        pixel_x=x,
                                        pixel_y=y,
                                        frame_width=metadata["resolution"]["width"],
                                        frame_height=metadata["resolution"]["height"],
                                        drone_lat=frame_data["drone_data"]["latitude"],
                                        drone_lon=frame_data["drone_data"]["longitude"],
                                        drone_alt_feet=frame_data["drone_data"][
                                            "altitude_feet"
                                        ],
                                        heading_degrees=frame_data["drone_data"][
                                            "compass_heading"
                                        ],
                                    )
                                )

                                detection = {
                                    "track_id": track_id,
                                    "class": class_name,
                                    "confidence": float(conf),
                                    "bbox": {
                                        "x_center": float(x),
                                        "y_center": float(y),
                                        "width": float(w),
                                        "height": float(h),
                                    },
                                    "geo_location": geo_location,
                                }
                                frame_detections.append(detection)

                                # 추적 정보 업데이트
                                if track_id not in tracks:
                                    tracks[track_id] = {
                                        "track_id": track_id,
                                        "class": class_name,
                                        "frame_history": [],
                                        "trajectory": {
                                            "pixel_coordinates": [],
                                            "geo_coordinates": [],
                                        },
                                    }
                                    class_counts[class_name] = (
                                        class_counts.get(class_name, 0) + 1
                                    )

                                track_info = {
                                    "frame_number": frame_data["frame_number"],
                                    "bbox": detection["bbox"],
                                    "geo_location": geo_location,
                                }
                                tracks[track_id]["frame_history"].append(track_info)
                                tracks[track_id]["trajectory"][
                                    "pixel_coordinates"
                                ].append([x, y])
                                tracks[track_id]["trajectory"][
                                    "geo_coordinates"
                                ].append(
                                    [
                                        geo_location["latitude"],
                                        geo_location["longitude"],
                                    ]
                                )

                                total_confidence += conf
                                detection_count += 1

                        # 프레임 정보 추가
                        processed_frame = {
                            "frame_number": frame_data["frame_number"],
                            "timestamp": frame_data["timestamp"],
                            "drone_data": frame_data["drone_data"],
                            "detections": frame_detections,
                        }
                        processed_frames.append(processed_frame)

                    # 요약 정보 생성
                    summary = {
                        "total_unique_objects": len(tracks),
                        "object_classes": class_counts,
                        "average_detection_confidence": (
                            total_confidence / detection_count
                            if detection_count > 0
                            else 0
                        ),
                    }

                    # 비디오별 최종 결과 구성
                    final_results[Path(video_path).name] = {
                        "metadata": metadata,
                        "frames": processed_frames,
                        "tracks": list(tracks.values()),
                        "summary": summary,
                    }

                    cap.release()

                except Exception as e:
                    self.logger.error(f"비디오 {video_path} 처리 중 오류: {str(e)}")
                    continue

            return final_results

        except Exception as e:
            self.logger.error(f"비디오 처리 중 오류: {str(e)}")
            raise

    def save_results(self, results: Dict[str, Dict], output_path: str) -> None:
        """
        처리 결과를 JSON 파일로 저장

        Args:
            results (Dict[str, Dict]): 처리 결과
            output_path (str): 저장할 파일 경로
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"처리 결과가 {output_path}에 저장되었습니다.")

        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {str(e)}")
            raise
