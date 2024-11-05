# src/nadir/geo_calculator.py
import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional


class GeographicCalculator:
    """
    드론 nadir 카메라 이미지에서 지리 좌표를 계산합니다.
    핀홀 카메라 모델과 구면 지구 근사를 사용합니다.
    """

    EARTH_RADIUS_METERS = 6_371_000
    FEET_TO_METERS = 0.3048

    def __init__(self, camera_fov_degrees: float):
        """
        카메라의 대각선 시야각으로 계산기를 초기화합니다.

        Args:
            camera_fov_degrees: 카메라의 대각선 시야각 (도)
        """
        self.camera_fov = camera_fov_degrees
        self.logger = logging.getLogger(__name__)

    def calculate_geo_location(
        self,
        pixel_x: float,
        pixel_y: float,
        frame_width: int,
        frame_height: int,
        drone_lat: float,
        drone_lon: float,
        drone_alt_feet: float,
        heading_degrees: float,
    ) -> Dict[str, float]:
        """
        픽셀 좌표를 지리 좌표로 변환합니다.

        Args:
            pixel_x, pixel_y: 대상 픽셀 좌표
            frame_width, frame_height: 프레임 크기
            drone_lat, drone_lon: 드론 GPS 위치
            drone_alt_feet: 드론 고도 (피트)
            heading_degrees: 드론 방향 (도)

        Returns:
            경도와 위도를 포함하는 딕셔너리
        """
        try:
            # 고도를 미터로 변환
            altitude_meters = drone_alt_feet * self.FEET_TO_METERS

            # 이미지 중심에서 상대 위치 계산
            rel_x = pixel_x - frame_width / 2
            rel_y = pixel_y - frame_height / 2

            # 핀홀 카메라 모델을 사용하여 픽셀당 미터 계산
            meters_per_pixel = (
                2 * altitude_meters * math.tan(math.radians(self.camera_fov / 2))
            ) / frame_width

            # 지상 거리로 변환 (미터)
            meters_x = rel_x * meters_per_pixel
            meters_y = rel_y * meters_per_pixel

            # 방향 회전 적용
            heading_rad = math.radians(heading_degrees)
            rotated_x = meters_x * math.cos(heading_rad) - meters_y * math.sin(
                heading_rad
            )
            rotated_y = meters_x * math.sin(heading_rad) + meters_y * math.cos(
                heading_rad
            )

            # 지리 좌표 오프셋으로 변환
            lat_diff = math.degrees(rotated_y / self.EARTH_RADIUS_METERS)
            lon_diff = math.degrees(
                rotated_x
                / (self.EARTH_RADIUS_METERS * math.cos(math.radians(drone_lat)))
            )

            return {"latitude": drone_lat + lat_diff, "longitude": drone_lon + lon_diff}

        except Exception as e:
            self.logger.error(f"지리 좌표 계산 실패: {str(e)}")
            return {"latitude": drone_lat, "longitude": drone_lon}
