# src/nadir/implementations/trajectory_analyzer.py
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..interfaces import ITrajectoryAnalyzer
from ..models import Intersection, TrajectoryMetrics


@dataclass
class LineSegment:
    """2차원 선분 표현"""

    p1: np.ndarray
    p2: np.ndarray
    frame1: int
    frame2: int
    direction: np.ndarray
    length: float

    def __init__(self, p1: np.ndarray, p2: np.ndarray, frame1: int, frame2: int):
        self.p1 = p1
        self.p2 = p2
        self.frame1 = frame1
        self.frame2 = frame2
        self.direction = p2 - p1
        self.length = np.linalg.norm(self.direction)
        if self.length > 0:
            self.direction = self.direction / self.length


class SpatialGrid:
    """공간 분할을 위한 그리드 시스템"""

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

    def add_segment(self, seg_id: Tuple[int, int], segment: LineSegment):
        min_x = min(segment.p1[0], segment.p2[0])
        max_x = max(segment.p1[0], segment.p2[0])
        min_y = min(segment.p1[1], segment.p2[1])
        max_y = max(segment.p1[1], segment.p2[1])

        start_cell = (int(min_x // self.cell_size), int(min_y // self.cell_size))
        end_cell = (int(max_x // self.cell_size), int(max_y // self.cell_size))

        for i in range(start_cell[0], end_cell[0] + 1):
            for j in range(start_cell[1], end_cell[1] + 1):
                if (i, j) not in self.grid:
                    self.grid[(i, j)] = set()
                self.grid[(i, j)].add(seg_id)

    def get_nearby_segments(self, segment: LineSegment) -> Set[Tuple[int, int]]:
        min_x = min(segment.p1[0], segment.p2[0])
        max_x = max(segment.p1[0], segment.p2[0])
        min_y = min(segment.p1[1], segment.p2[1])
        max_y = max(segment.p1[1], segment.p2[1])

        start_cell = (int(min_x // self.cell_size), int(min_y // self.cell_size))
        end_cell = (int(max_x // self.cell_size), int(max_y // self.cell_size))

        nearby = set()
        for i in range(start_cell[0], end_cell[0] + 1):
            for j in range(start_cell[1], end_cell[1] + 1):
                if (i, j) in self.grid:
                    nearby.update(self.grid[(i, j)])
        return nearby


class OptimizedTrajectoryAnalyzer(ITrajectoryAnalyzer):
    def __init__(
        self,
        dangerous_psm_threshold: float = 1.5,
        min_trajectory_length: int = 5,
        min_intersection_angle: float = 15.0,
        grid_cell_size: float = 50.0,
    ):
        self.dangerous_psm_threshold = dangerous_psm_threshold
        self.min_trajectory_length = min_trajectory_length
        self.min_intersection_angle = min_intersection_angle
        self.grid_cell_size = grid_cell_size
        self.logger = logging.getLogger(__name__)

    def analyze_trajectories(
        self,
        tracks: Dict[int, Dict],
        fps: float,
    ) -> Dict[int, TrajectoryMetrics]:
        metrics = {}
        if not tracks:
            return metrics

        # Prepare segments and spatial grid
        track_segments = {}
        spatial_grid = SpatialGrid(self.grid_cell_size)

        for track_id, track_data in tracks.items():
            try:
                if not self._validate_track_data(track_data):
                    continue

                segments = self._create_line_segments(track_data)
                if segments:
                    track_segments[track_id] = segments
                    coords = np.array(track_data["trajectory"]["pixel_coordinates"])
                    metrics[track_id] = self._calculate_trajectory_metrics(
                        coords, track_id
                    )

                    # Add segments to spatial grid
                    for i, segment in enumerate(segments):
                        spatial_grid.add_segment((track_id, i), segment)

            except Exception as e:
                self.logger.error(f"Track {track_id} 처리 실패: {str(e)}")
                continue

        self._find_intersections(track_segments, spatial_grid, metrics, fps)
        return metrics

    def _validate_track_data(self, track_data: Dict) -> bool:
        return (
            track_data.get("trajectory", {}).get("pixel_coordinates")
            and track_data.get("frame_history")
            and len(track_data["trajectory"]["pixel_coordinates"])
            >= self.min_trajectory_length
        )

    def _smooth_trajectory(self, coords: np.ndarray, window: int = 3) -> np.ndarray:
        kernel = np.ones(window) / window
        return np.column_stack(
            [
                np.convolve(coords[:, 0], kernel, mode="valid"),
                np.convolve(coords[:, 1], kernel, mode="valid"),
            ]
        )

    def _create_line_segments(self, track_data: Dict) -> List[LineSegment]:
        try:
            coords = np.array(track_data["trajectory"]["pixel_coordinates"])
            frames = np.array(
                [frame["frame_number"] for frame in track_data["frame_history"]]
            )

            if len(coords) < 3:
                return []

            coords = self._smooth_trajectory(coords)
            frames = frames[1:]

            segments = []
            for i in range(len(coords) - 1):
                segments.append(
                    LineSegment(
                        p1=coords[i],
                        p2=coords[i + 1],
                        frame1=int(frames[i]),
                        frame2=int(frames[i + 1]),
                    )
                )

            return segments
        except Exception as e:
            self.logger.error(f"선분 생성 실패: {str(e)}")
            return []

    def _find_intersections(
        self,
        track_segments: Dict,
        spatial_grid: SpatialGrid,
        metrics: Dict[int, TrajectoryMetrics],
        fps: float,
    ) -> None:
        track_ids = sorted(track_segments.keys())

        for i, track_id1 in enumerate(track_ids):
            segments1 = track_segments[track_id1]

            for seg_idx1, seg1 in enumerate(segments1):
                nearby_segments = spatial_grid.get_nearby_segments(seg1)

                for track_id2, seg_idx2 in nearby_segments:
                    if track_id2 <= track_id1:
                        continue

                    seg2 = track_segments[track_id2][seg_idx2]
                    self._process_segment_intersection(
                        seg1, seg2, track_id1, track_id2, metrics, fps
                    )

    def _process_segment_intersection(
        self,
        seg1: LineSegment,
        seg2: LineSegment,
        track_id1: int,
        track_id2: int,
        metrics: Dict[int, TrajectoryMetrics],
        fps: float,
    ) -> None:
        result = self._find_intersection(seg1, seg2)

        if result is not None:
            intersection_point, t1, t2 = result
            angle = self._calculate_approach_angle(seg1, seg2)

            if angle >= self.min_intersection_angle:
                frame1 = seg1.frame1 + t1 * (seg1.frame2 - seg1.frame1)
                frame2 = seg2.frame1 + t2 * (seg2.frame2 - seg2.frame1)
                psm = abs(frame1 - frame2) / fps

                intersection = Intersection(
                    track_id1=track_id1,
                    track_id2=track_id2,
                    frame_number=min(frame1, frame2),
                    intersection_point=tuple(intersection_point),
                    psm=psm,
                    is_dangerous=psm < self.dangerous_psm_threshold,
                    approaching_angle=angle,
                )

                metrics[track_id1].intersections.append(intersection)
                metrics[track_id2].intersections.append(intersection)

    def _find_intersection(
        self, seg1: LineSegment, seg2: LineSegment
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        try:
            cross = np.cross(seg1.direction, seg2.direction)
            if abs(cross) < 1e-10:
                return None

            v3 = seg2.p1 - seg1.p1
            t1 = np.cross(v3, seg2.direction) / cross
            t2 = np.cross(v3, seg1.direction) / cross

            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                intersection_point = seg1.p1 + t1 * seg1.direction * seg1.length
                return intersection_point, t1, t2

            return None
        except Exception as e:
            self.logger.error(f"교차점 계산 실패: {str(e)}")
            return None

    def _calculate_approach_angle(self, seg1: LineSegment, seg2: LineSegment) -> float:
        try:
            if seg1.length == 0 or seg2.length == 0:
                return 0.0

            cos_angle = np.dot(seg1.direction, seg2.direction)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            return min(angle_deg, 180 - angle_deg)
        except Exception as e:
            self.logger.error(f"접근 각도 계산 실패: {str(e)}")
            return 0.0

    def _calculate_trajectory_metrics(
        self,
        coords: np.ndarray,
        track_id: int,
    ) -> TrajectoryMetrics:
        try:
            distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))

            return TrajectoryMetrics(
                track_id=track_id,
                total_distance=float(np.sum(distances)),
                average_speed=float(np.mean(distances)) if len(distances) > 0 else 0.0,
                max_speed=float(np.max(distances)) if len(distances) > 0 else 0.0,
                intersections=[],
            )
        except Exception as e:
            self.logger.error(f"궤적 메트릭 계산 실패: {str(e)}")
            return TrajectoryMetrics(track_id, 0.0, 0.0, 0.0, [])
