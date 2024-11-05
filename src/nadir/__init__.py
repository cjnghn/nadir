# src/nadir/__init__.py
import logging
from pathlib import Path
from typing import Dict

from .processor import DroneVideoProcessor


def setup_logging() -> None:
    """기본 로깅 설정을 구성합니다"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_default_config() -> Dict:
    """기본 처리 설정을 생성합니다"""
    return {
        "camera": {
            "fov_degrees": 80,  # 카메라 시야각 (도)
        },
        "detection": {
            "confidence_threshold": 0.5,  # 최소 감지 신뢰도
        },
        "trajectory": {
            "dangerous_psm_threshold": 1.5,  # 위험한 PSM 임계값 (초)
            "min_intersection_angle": 0.0,  # 최소 교차 각도 (도)
        },
        "paths": {"model": "models/yolov8s_v3_2560_b8_e30.pt", "output": "output"},
    }


def main() -> None:
    """주 실행 함수"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 설정 로드
        config = create_default_config()

        # 비디오 프로세서 초기화
        processor = DroneVideoProcessor(
            model_path=config["paths"]["model"],
            camera_fov_degrees=config["camera"]["fov_degrees"],
            confidence_threshold=config["detection"]["confidence_threshold"],
            dangerous_psm_threshold=config["trajectory"]["dangerous_psm_threshold"],
            min_intersection_angle=config["trajectory"]["min_intersection_angle"],
        )

        # 비디오 처리
        video_paths = ["videos/mission1/video.mp4"]
        log_path = "flight_logs/mission1.csv"

        logger.info(f"{len(video_paths)}개의 비디오를 처리합니다...")
        results = processor.process_videos(video_paths, log_path)

        # 결과 저장
        output_path = Path(config["paths"]["output"]) / "processed_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_results(results, str(output_path))

        logger.info("처리 성공")

    except Exception as e:
        logger.error(f"처리 실패: {str(e)}")
        raise
