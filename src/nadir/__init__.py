import logging
from pathlib import Path
from typing import Dict, List

from .processor import DroneVideoProcessor


def setup_logging() -> None:
    """Configure basic logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_default_config() -> Dict:
    """Create default processing configuration"""
    return {
        "camera": {
            "fov_degrees": 80,  # Camera field of view in degrees
        },
        "detection": {
            "confidence_threshold": 0.5,  # Minimum detection confidence
            "track_timeout_frames": 30,  # Frames before ending track
        },
        "paths": {"model": "models/yolov8s-visdrone.pt", "output": "output"},
    }


def main() -> None:
    """Main execution function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = create_default_config()

        # Initialize video processor
        processor = DroneVideoProcessor(
            model_path=config["paths"]["model"],
            camera_fov_degrees=config["camera"]["fov_degrees"],
            confidence_threshold=config["detection"]["confidence_threshold"],
        )

        # Process videos
        video_paths = ["videos/mission1/video.mp4"]
        log_path = "flight_logs/mission1.csv"

        logger.info(f"Processing {len(video_paths)} videos...")
        results = processor.process_videos(video_paths, log_path)

        # Save results
        output_path = Path(config["paths"]["output"]) / "processed_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_results(results, str(output_path))

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
