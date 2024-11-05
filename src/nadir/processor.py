import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO

from .geo_calculator import GeographicCalculator
from .log_mapper import DroneLogMapper
from .models import (
    BoundingBox,
    Detection,
    GeoPoint,
    Timestamp,
    VideoMetadata,
    VideoSegment,
    DroneFrame,
)


class DroneVideoProcessor:
    """Process drone videos with object detection and geographic mapping"""

    def __init__(
        self,
        model_path: str,
        camera_fov_degrees: float,
        confidence_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.calculator = GeographicCalculator(camera_fov_degrees)
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

    def process_videos(self, video_paths: List[str], log_path: str) -> Dict[str, Dict]:
        """Process multiple videos with synchronized drone data"""
        try:
            mapper = DroneLogMapper(log_path)
            mapped_segments = mapper.map_videos_to_segments(video_paths)

            results = {}
            for video_path, segment in mapped_segments.items():
                results[Path(video_path).name] = self._process_single_video(
                    video_path, segment
                )

            return results
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            raise

    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract metadata from video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate duration
            duration_seconds = frame_count / fps if fps > 0 else 0

            # Get creation time
            creation_time = int(
                Path(video_path).stat().st_mtime * 1000
            )  # to milliseconds

            cap.release()

            return VideoMetadata(
                total_frames=frame_count,
                duration_seconds=duration_seconds,
                timestamp=Timestamp(
                    epoch_ms=creation_time,
                    relative_ms=0,
                    datetime_utc=Path(video_path).stat().st_mtime,
                ),
                fps=fps,
                width=width,
                height=height,
            )
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {video_path}: {str(e)}")
            raise

    def _process_single_video(self, video_path: str, segment: "VideoSegment") -> Dict:
        """Process single video with object detection and tracking"""
        try:
            metadata = self._extract_video_metadata(video_path)
            tracks = {}
            processed_frames = []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            for frame_data in segment.frames:
                processed_frame = self._process_frame(cap, frame_data, metadata, tracks)
                if processed_frame:
                    processed_frames.append(processed_frame)

            cap.release()

            return {
                "metadata": self._format_metadata_dict(metadata),
                "frames": processed_frames,
                "tracks": list(tracks.values()),
                "summary": self._generate_summary(tracks),
            }
        except Exception as e:
            self.logger.error(f"Failed to process {video_path}: {str(e)}")
            raise

    def _process_frame(
        self,
        cap: cv2.VideoCapture,
        frame_data: "DroneFrame",
        metadata: VideoMetadata,
        tracks: Dict,
    ) -> Optional[Dict]:
        """Process single frame with object detection and tracking"""
        success, frame = cap.read()
        if not success:
            return None

        # Run YOLO detection and tracking
        results = self.model.track(frame, persist=True)

        detections = []
        if results[0].boxes.id is not None:
            for box, track_id, cls, conf in zip(
                results[0].boxes.xywh.cpu(),
                results[0].boxes.id.int().cpu().tolist(),
                results[0].boxes.cls.int().cpu().tolist(),
                results[0].boxes.conf.cpu().tolist(),
            ):
                if conf < self.confidence_threshold:
                    continue

                x, y, w, h = box.tolist()
                class_name = self.model.names[cls]

                # Calculate geographic location
                geo_location = self.calculator.calculate_geo_location(
                    pixel_x=x,
                    pixel_y=y,
                    frame_width=metadata.width,
                    frame_height=metadata.height,
                    drone_lat=frame_data.drone_state.position.latitude,
                    drone_lon=frame_data.drone_state.position.longitude,
                    drone_alt_feet=frame_data.drone_state.altitude_feet,
                    heading_degrees=frame_data.drone_state.heading_degrees,
                )

                detection = Detection(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=BoundingBox(
                        x_center=float(x),
                        y_center=float(y),
                        width=float(w),
                        height=float(h),
                    ),
                    geo_location=GeoPoint(
                        latitude=geo_location["latitude"],
                        longitude=geo_location["longitude"],
                    ),
                )
                detections.append(self._format_detection_dict(detection))

                # Update tracking information
                if track_id not in tracks:
                    tracks[track_id] = {
                        "track_id": track_id,
                        "class": class_name,
                        "frame_history": [],
                        "trajectory": {"pixel_coordinates": [], "geo_coordinates": []},
                    }

                # Update track history
                tracks[track_id]["frame_history"].append(
                    {
                        "frame_number": frame_data.number,
                        "bbox": self._format_bbox_dict(detection.bbox),
                        "geo_location": {
                            "latitude": geo_location["latitude"],
                            "longitude": geo_location["longitude"],
                        },
                    }
                )

                # Update trajectories
                tracks[track_id]["trajectory"]["pixel_coordinates"].append([x, y])
                tracks[track_id]["trajectory"]["geo_coordinates"].append(
                    [geo_location["latitude"], geo_location["longitude"]]
                )

        return {
            "frame_number": frame_data.number,
            "timestamp": self._format_timestamp_dict(frame_data.timestamp),
            "drone_data": {
                "latitude": frame_data.drone_state.position.latitude,
                "longitude": frame_data.drone_state.position.longitude,
                "altitude_feet": frame_data.drone_state.altitude_feet,
                "compass_heading": frame_data.drone_state.heading_degrees,
            },
            "detections": detections,
        }

    def _generate_summary(self, tracks: Dict) -> Dict:
        """Generate summary statistics from tracking results"""
        class_counts = {}
        total_confidence = 0.0
        detection_count = 0

        for track in tracks.values():
            class_name = track["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for frame in track["frame_history"]:
                if "confidence" in frame:
                    total_confidence += frame["confidence"]
                    detection_count += 1

        return {
            "total_unique_objects": len(tracks),
            "object_classes": class_counts,
            "average_detection_confidence": (
                total_confidence / detection_count if detection_count > 0 else 0
            ),
        }

    @staticmethod
    def _format_metadata_dict(metadata: VideoMetadata) -> Dict:
        """Format VideoMetadata as dictionary"""
        return {
            "total_frames": metadata.total_frames,
            "duration_seconds": metadata.duration_seconds,
            "timestamp": {
                "milliseconds": metadata.timestamp.epoch_ms,
                "datetime_utc": metadata.timestamp.datetime_utc,
            },
            "fps": metadata.fps,
            "resolution": {"width": metadata.width, "height": metadata.height},
        }

    @staticmethod
    def _format_detection_dict(detection: Detection) -> Dict:
        """Format Detection as dictionary"""
        return {
            "track_id": detection.track_id,
            "class": detection.class_name,
            "confidence": detection.confidence,
            "bbox": {
                "x_center": detection.bbox.x_center,
                "y_center": detection.bbox.y_center,
                "width": detection.bbox.width,
                "height": detection.bbox.height,
            },
            "geo_location": {
                "latitude": detection.geo_location.latitude,
                "longitude": detection.geo_location.longitude,
            },
        }

    @staticmethod
    def _format_bbox_dict(bbox: BoundingBox) -> Dict:
        """Format BoundingBox as dictionary"""
        return {
            "x_center": bbox.x_center,
            "y_center": bbox.y_center,
            "width": bbox.width,
            "height": bbox.height,
        }

    @staticmethod
    def _format_geo_point_dict(point: GeoPoint) -> Dict:
        """Format GeoPoint as dictionary"""
        return {"latitude": point.latitude, "longitude": point.longitude}

    @staticmethod
    def _format_timestamp_dict(timestamp: Timestamp) -> Dict:
        """Format Timestamp as dictionary"""
        return {
            "relative_ms": timestamp.relative_ms,
            "milliseconds": timestamp.epoch_ms,
            "datetime_utc": timestamp.datetime_utc,
        }

    def save_results(self, results: Dict, output_path: str) -> None:
        """Save processing results to JSON file"""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
