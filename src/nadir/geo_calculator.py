from dataclasses import dataclass
from typing import Dict, Optional
import logging
import math


class GeographicCalculator:
    """
    Calculates geographic coordinates from drone nadir camera imagery
    using pinhole camera model and spherical earth approximation.
    """

    EARTH_RADIUS_METERS = 6_371_000
    FEET_TO_METERS = 0.3048

    def __init__(self, camera_fov_degrees: float):
        """
        Initialize calculator with camera field of view

        Args:
            camera_fov_degrees: Camera's diagonal field of view in degrees
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
        Convert pixel coordinates to geographic coordinates

        Args:
            pixel_x, pixel_y: Target pixel coordinate
            frame_width, frame_height: Frame dimensions
            drone_lat, drone_lon: Drone GPS position
            drone_alt_feet: Drone altitude in feet
            heading_degrees: Drone compass heading

        Returns:
            Dictionary containing latitude and longitude
        """
        try:
            # Convert altitude to meters
            altitude_meters = drone_alt_feet * self.FEET_TO_METERS

            # Calculate relative position from image center
            rel_x = pixel_x - frame_width / 2
            rel_y = pixel_y - frame_height / 2

            # Calculate meters per pixel using pinhole camera model
            meters_per_pixel = (
                2 * altitude_meters * math.tan(math.radians(self.camera_fov / 2))
            ) / frame_width

            # Convert to ground distance in meters
            meters_x = rel_x * meters_per_pixel
            meters_y = rel_y * meters_per_pixel

            # Apply heading rotation
            heading_rad = math.radians(heading_degrees)
            rotated_x = meters_x * math.cos(heading_rad) - meters_y * math.sin(
                heading_rad
            )
            rotated_y = meters_x * math.sin(heading_rad) + meters_y * math.cos(
                heading_rad
            )

            # Convert to geographic offsets
            lat_diff = math.degrees(rotated_y / self.EARTH_RADIUS_METERS)
            lon_diff = math.degrees(
                rotated_x
                / (self.EARTH_RADIUS_METERS * math.cos(math.radians(drone_lat)))
            )

            return {"latitude": drone_lat + lat_diff, "longitude": drone_lon + lon_diff}

        except Exception as e:
            self.logger.error(f"Geographic coordinate calculation failed: {str(e)}")
            return {"latitude": drone_lat, "longitude": drone_lon}
