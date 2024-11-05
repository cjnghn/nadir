import logging
import math
from typing import Any, Dict, List, Optional, Tuple


class GeoCalculator:
    """드론의 탑뷰 영상에서 지리적 위치를 계산하는 클래스"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feet_to_meters = 0.3048

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
        이미지 좌표를 실제 지리 좌표로 변환

        Args:
            pixel_x, pixel_y: 이미지 상의 픽셀 좌표
            frame_width, frame_height: 프레임 크기
            drone_lat, drone_lon: 드론의 GPS 좌표
            drone_alt_feet: 드론의 고도 (피트)
            heading_degrees: 드론의 진행 방향 (도)

        Returns:
            Dict[str, float]: 계산된 위도/경도
        """
        try:
            # 드론 고도를 미터로 변환
            altitude_meters = drone_alt_feet * self.feet_to_meters

            # 이미지 중심점으로부터의 상대 위치 계산
            rel_x = pixel_x - frame_width / 2
            rel_y = pixel_y - frame_height / 2

            # 픽셀당 실제 거리 계산 (간단한 핀홀 카메라 모델 사용)
            # FOV와 고도를 기반으로 계산
            meters_per_pixel = (
                2
                * altitude_meters
                * math.tan(math.radians(self.config["camera_fov"] / 2))
            ) / frame_width

            # 상대 위치를 미터 단위로 변환
            meters_x = rel_x * meters_per_pixel
            meters_y = rel_y * meters_per_pixel

            # 드론의 heading을 고려하여 좌표 회전
            heading_rad = math.radians(heading_degrees)
            rotated_x = meters_x * math.cos(heading_rad) - meters_y * math.sin(
                heading_rad
            )
            rotated_y = meters_x * math.sin(heading_rad) + meters_y * math.cos(
                heading_rad
            )

            # 미터 단위 거리를 위도/경도 차이로 변환
            # 지구 반경을 6371km로 가정
            lat_diff = math.degrees(rotated_y / (6371000))
            lon_diff = math.degrees(
                rotated_x / (6371000 * math.cos(math.radians(drone_lat)))
            )

            return {"latitude": drone_lat + lat_diff, "longitude": drone_lon + lon_diff}

        except Exception as e:
            logging.error(f"지리적 위치 계산 중 오류: {str(e)}")
            return {
                "latitude": drone_lat,
                "longitude": drone_lon,
            }  # 오류 시 드론 위치 반환
