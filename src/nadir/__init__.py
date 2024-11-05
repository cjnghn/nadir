from .video_processor import IntegratedVideoProcessor


def main() -> None:
    # 설정
    config = {
        "camera_fov": 80,  # 카메라 시야각 (도)
        "track_timeout_frames": 30,  # 추적 종료 판단을 위한 프레임 수
        # 기타 필요한 설정들...
    }

    # 프로세서 초기화
    processor = IntegratedVideoProcessor(
        model_path="models/yolov8s-visdrone.pt", config=config
    )

    # 비디오 처리
    video_paths = ["videos/mission1/video.mp4"]
    log_path = "flight_logs/mission1.csv"

    results = processor.process_videos(video_paths, log_path)

    # 결과 저장
    processor.save_results(results, "processed_results.json")
