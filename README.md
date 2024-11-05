# Nadir - 드론 영상 분석 시스템

## 프로젝트 개요
Nadir는 드론에서 촬영된 영상을 분석하여 객체를 감지하고, 이동 경로를 추적하며, 안전 메트릭을 계산하는 시스템입니다. 이 시스템은 드론의 텔레메트리 데이터와 영상을 동기화하여 지상의 객체들의 실제 지리적 위치와 상호작용을 분석합니다.

## 주요 기능
- 객체 감지 및 추적: YOLOv8 기반의 실시간 객체 감지 및 추적
- 지리적 매핑: 픽셀 좌표를 실제 지리 좌표(위도/경도)로 변환
- 궤적 분석: 객체들의 이동 경로 및 상호작용 분석
- 안전 메트릭 계산: PSM 및 위험 상호작용 감지
- 텔레메트리 동기화: 드론 로그 데이터와 비디오 프레임 동기화

## 기술 스택
- Python 3.8+
- OpenCV
- YOLOv8
- NumPy
- Pandas
- SciPy
- Ultralytics

## 시스템 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- 저장 공간: 프로젝트 및 모델 파일용 2GB 이상


## 사용 방법
1. 설정 파일 준비
```python
config = {
    "camera": {
        "fov_degrees": 80,  # 카메라 시야각
    },
    "detection": {
        "confidence_threshold": 0.5,  # 객체 감지 신뢰도 임계값
    },
    "trajectory": {
        "dangerous_psm_threshold": 1.5,  # 위험 PSM 임계값 (초)
        "min_intersection_angle": 15.0,  # 최소 교차각 (도)
    }
}
```

2. 드론 영상 및 로그 파일 준비
```python
video_paths = ["videos/mission1/video.mp4"]
log_path = "flight_logs/mission1.csv"
```

3. 분석 실행
```python
from nadir import create_processor, create_default_config

config = create_default_config()
processor = create_processor(config)
results = processor.process_videos(video_paths, log_path)
```

## 입력 데이터 형식

### 드론 로그 파일 (CSV)
```csv
time(millisecond),latitude,longitude,altitude(feet),compass_heading(degrees),isVideo
1637000000000,37.12345,127.12345,400,180,1
```

### 비디오 파일
- 지원 형식: MP4, AVI
- 권장 해상도: 1920x1080 이상
- 프레임률: 30fps 이상

## 출력 데이터 형식

### 분석 결과 (JSON)
```json
{
    "metadata": {
        "total_frames": 1200,
        "duration_seconds": 40.0,
        "fps": 30.0
    },
    "frames": [...],
    "tracks": [...],
    "trajectory_analysis": {...},
    "summary": {
        "total_unique_objects": 10,
        "dangerous_interactions": 2
    }
}
```

## 아키텍처
다음과 같은 주요 컴포넌트로 구성됩니다:

- DroneVideoProcessor: 전체 처리 과정 조율
- ObjectDetector: 객체 감지 및 추적 담당
- GeoCalculator: 좌표 변환 처리
- TrajectoryAnalyzer: 궤적 분석 및 안전 메트릭 계산
- LogMapper: 텔레메트리 데이터 매핑
- ResultWriter: 분석 결과 저장

## 성능 최적화
- 공간 분할을 통한 궤적 분석 최적화
- 벡터화된 연산으로 성능 향상
- 메모리 효율적인 데이터 처리

## 로깅 및 오류 처리
- 상세한 로그 기록
- 단계별 예외 처리
- 복구 가능한 오류 관리

