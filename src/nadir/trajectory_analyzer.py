# src/nadir/trajectory_analyzer.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree


@dataclass
class Intersection:
    """궤적 교차 정보"""

    track_id1: int
    track_id2: int
    frame_number: float
    intersection_point: Tuple[float, float]
    psm: float
    is_dangerous: bool
    approaching_angle: float


@dataclass
class TrajectoryMetrics:
    """궤적에 대한 계산된 지표"""

    track_id: int
    total_distance: float
    average_speed: float
    max_speed: float
    intersections: List[Intersection]


@dataclass
class LineSegment:
    """2차원 선분 표현"""

    p1: np.ndarray
    p2: np.ndarray
    frame1: int
    frame2: int
    direction: np.ndarray  # 캐시된 방향 벡터
    length: float  # 캐시된 길이

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


class TrajectoryAnalyzer:
    """객체 궤적을 분석하고 PSM 포함한 안전 지표를 계산"""

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

    def _smooth_trajectory(self, coords: np.ndarray, window: int = 3) -> np.ndarray:
        """벡터화된 궤적 스무딩"""
        kernel = np.ones(window) / window
        return np.column_stack(
            [
                np.convolve(coords[:, 0], kernel, mode="valid"),
                np.convolve(coords[:, 1], kernel, mode="valid"),
            ]
        )

    def _create_line_segments(self, track_data: Dict) -> List[LineSegment]:
        """벡터화된 선분 생성"""
        try:
            coords = np.array(track_data["trajectory"]["pixel_coordinates"])
            frames = np.array(
                [frame["frame_number"] for frame in track_data["frame_history"]]
            )

            if len(coords) < 3:  # minimum for smoothing
                return []

            # Vectorized smoothing
            coords = self._smooth_trajectory(coords)
            frames = frames[1:]  # Adjust frames to match smoothed coordinates

            # Vectorized segment creation
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
            print(f"Error in create_line_segments: {str(e)}")
            return []

    def _find_intersection(
        self, seg1: LineSegment, seg2: LineSegment
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """최적화된 선분 교차점 계산"""
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
            print(f"Error in find_intersection: {str(e)}")
            return None

    def _calculate_approach_angle(self, seg1: LineSegment, seg2: LineSegment) -> float:
        """캐시된 방향 벡터를 사용한 각도 계산"""
        try:
            if seg1.length == 0 or seg2.length == 0:
                return 0.0

            cos_angle = np.dot(seg1.direction, seg2.direction)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            return min(angle_deg, 180 - angle_deg)
        except Exception as e:
            print(f"Error in calculate_approach_angle: {str(e)}")
            return 0.0

    def analyze_trajectories(
        self, tracks: Dict[int, Dict], fps: float
    ) -> Dict[int, TrajectoryMetrics]:
        """공간 분할을 활용한 최적화된 궤적 분석"""
        metrics = {}
        if not tracks:
            return metrics

        # Prepare segments and spatial grid
        track_segments = {}
        spatial_grid = SpatialGrid(self.grid_cell_size)
        print("Creating line segments and spatial index...")

        for track_id, track_data in tqdm(tracks.items()):
            try:
                if (
                    not track_data.get("trajectory", {}).get("pixel_coordinates")
                    or not track_data.get("frame_history")
                    or len(track_data["trajectory"]["pixel_coordinates"])
                    < self.min_trajectory_length
                ):
                    continue

                segments = self._create_line_segments(track_data)
                if segments:
                    track_segments[track_id] = segments
                    coords = np.array(track_data["trajectory"]["pixel_coordinates"])
                    metrics[track_id] = self._calculate_trajectory_metrics(coords)
                    metrics[track_id].track_id = track_id

                    # Add segments to spatial grid
                    for i, segment in enumerate(segments):
                        spatial_grid.add_segment((track_id, i), segment)

            except Exception as e:
                print(f"Warning: Failed to process track {track_id}: {str(e)}")
                continue

        # Optimized intersection search using spatial grid
        print("Finding intersections...")
        track_ids = sorted(track_segments.keys())

        for i, track_id1 in tqdm(enumerate(track_ids)):
            segments1 = track_segments[track_id1]

            for seg_idx1, seg1 in enumerate(segments1):
                nearby_segments = spatial_grid.get_nearby_segments(seg1)

                for track_id2, seg_idx2 in nearby_segments:
                    if track_id2 <= track_id1:
                        continue

                    seg2 = track_segments[track_id2][seg_idx2]
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

        return metrics

    def _calculate_trajectory_metrics(self, coords: np.ndarray) -> TrajectoryMetrics:
        """벡터화된 궤적 메트릭 계산"""
        try:
            distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))

            return TrajectoryMetrics(
                track_id=0,
                total_distance=float(np.sum(distances)),
                average_speed=float(np.mean(distances)) if len(distances) > 0 else 0.0,
                max_speed=float(np.max(distances)) if len(distances) > 0 else 0.0,
                intersections=[],
            )
        except Exception as e:
            print(f"Error in calculate_trajectory_metrics: {str(e)}")
            return TrajectoryMetrics(0, 0.0, 0.0, 0.0, [])

    def format_metrics_dict(
        self, metrics: Dict[int, TrajectoryMetrics]
    ) -> Dict[int, Dict]:
        """JSON 직렬화를 위한 메트릭 포맷팅"""
        return {
            track_id: {
                "metrics": {
                    "total_distance": track_metrics.total_distance,
                    "average_speed": track_metrics.average_speed,
                    "max_speed": track_metrics.max_speed,
                },
                "intersections": [
                    {
                        "track_id1": i.track_id1,
                        "track_id2": i.track_id2,
                        "frame_number": i.frame_number,
                        "intersection_point": i.intersection_point,
                        "psm": i.psm,
                        "is_dangerous": i.is_dangerous,
                        "approaching_angle": i.approaching_angle,
                    }
                    for i in track_metrics.intersections
                ],
            }
            for track_id, track_metrics in metrics.items()
        }
