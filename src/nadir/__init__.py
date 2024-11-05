# src/nadir/__init__.py
import logging
from pathlib import Path
from typing import Dict

from .implementations.detectors import YOLODetector
from .implementations.frame_processor import DefaultFrameProcessor
from .implementations.geo_calculator import DefaultGeoCalculator
from .implementations.log_mapper import CSVLogMapper
from .implementations.result_writer import JSONResultWriter
from .implementations.trajectory_analyzer import OptimizedTrajectoryAnalyzer
from .processor import DroneVideoProcessor


def setup_logging() -> None:
    """기본 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_default_config() -> Dict:
    """기본 처리 설정 생성"""
    return {
        "camera": {
            "fov_degrees": 80,
        },
        "detection": {
            "confidence_threshold": 0.5,
        },
        "trajectory": {
            "dangerous_psm_threshold": 1.5,
            "min_intersection_angle": 0.0,
        },
        "paths": {
            # "model": "models/yolov8s-visdrone.pt",
            "model": "models/yolov8s_v3_2560_b8_e30.pt",
            "output": "output",
        },
    }


def create_processor(config: Dict, log_path: str) -> DroneVideoProcessor:
    """프로세서 인스턴스 생성

    Args:
        config: 처리 설정
        log_path: 텔레메트리 로그 파일 경로
    """
    detector = YOLODetector(config["paths"]["model"])
    geo_calculator = DefaultGeoCalculator(config["camera"]["fov_degrees"])
    trajectory_analyzer = OptimizedTrajectoryAnalyzer(
        dangerous_psm_threshold=config["trajectory"]["dangerous_psm_threshold"],
        min_intersection_angle=config["trajectory"]["min_intersection_angle"],
    )
    frame_processor = DefaultFrameProcessor(
        detector=detector,
        geo_calculator=geo_calculator,
        confidence_threshold=config["detection"]["confidence_threshold"],
    )
    log_mapper = CSVLogMapper(log_path)

    return DroneVideoProcessor(
        detector=detector,
        geo_calculator=geo_calculator,
        trajectory_analyzer=trajectory_analyzer,
        log_mapper=log_mapper,
        frame_processor=frame_processor,
        result_writer=JSONResultWriter(),
    )


def main() -> None:
    """메인 실행 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = create_default_config()
        video_paths = ["videos/mission1/video.mp4"]
        log_path = "flight_logs/mission1.csv"

        processor = create_processor(config, log_path)
        logger.info(f"{len(video_paths)}개의 비디오를 처리합니다...")
        results = processor.process_videos(video_paths)

        output_path = Path(config["paths"]["output"]) / "processed_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_results(results, str(output_path))

        logger.info("처리 성공")

    except Exception as e:
        logger.error(f"처리 실패: {str(e)}")
        raise


if __name__ == "__main__":
    main()
