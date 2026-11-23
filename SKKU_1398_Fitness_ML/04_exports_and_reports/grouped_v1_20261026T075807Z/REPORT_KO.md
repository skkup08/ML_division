# SKKU 1398 운동기구 인식 baseline 실행 보고서

범위: FitFuel 공개 데이터에서 재분할한 proxy 모델. 센터 현장 성능 및 Flutter 휴대폰 실행은 검증하지 않았습니다.

## 실행 결과
- mAP@0.5: 0.9343
- mAP@0.5:0.95: 0.7907
- Precision: 0.9117
- Recall: 0.8646
- 가중치: 5505114 bytes
- T4 단일 사진 전체 처리 중앙값: 14.21 ms (휴대폰 속도 아님)

| 클래스 | AP50 | AP50:95 | Precision | Recall |
|---|---:|---:|---:|---:|
| treadmill | 0.9950 | 0.8777 | 0.9785 | 1.0000 |
| seated_chest_press | 0.9647 | 0.8143 | 0.9437 | 0.7986 |
| shoulder_press | 0.8198 | 0.6455 | 0.8068 | 0.7203 |
| seated_leg_press | 0.8991 | 0.7508 | 0.8630 | 0.8065 |
| smith_machine | 0.9930 | 0.8650 | 0.9666 | 0.9977 |

## 데이터 및 한계
- 원본 다운로드의 data.yaml은 28종. 공개 페이지의 29종 표기와 차이가 있어 실제 파일 기준으로 처리했습니다.
- 공식 test에는 목표 5종이 없고 valid에는 Chest Press가 없어 공식 분할을 사용할 수 없었습니다.
- 원본 파일명 및 0/90/180/270도 pHash 유사도 그룹으로 묶어 seed 42로 새 분할을 학습 전에 고정했습니다.
- 최종 학습 753장, 검증 147장, 테스트 156장. 증강본이 포함되어 이미지 수가 독립 촬영 수를 뜻하지 않습니다.
- 잘못된 라벨 이미지 48장 제외, 정확 중복 195장 제외, 추가 유사 그룹 이미지 6장 격리.
- 5종이 없는 사진 3412장은 이번 학습 및 평가에서 제외했습니다. 실제 헬스장의 미지원 기구/배경 오탐률은 검증되지 않았습니다.
- 제품·광고 사진과 회전 이미지가 많습니다. 세션/촬영자/웹 출처 독립성 및 모든 라벨의 완전성은 보장하지 않습니다.
- 원본 5개 ID: Treadmill 27 → 0, Chest Press Machine 7 → 1, Shoulder Press Machine 23 → 2, Leg Press Machine 18 → 3, Smith Machine 24 → 4.
- Chest/Leg Press는 시티드 세부 형태를 보증하지 않습니다.
- 학습 중 모델 선택은 검증 데이터 기준. 테스트 결과에 따른 재학습/임계값 조정은 하지 않았습니다.
- Precision/Recall은 Ultralytics 평가 지표이며 앱 confidence 0.25에서의 동작점과 다릅니다. 별도 failure_cases.json이 confidence 0.25, matching IoU 0.5의 TP/FP/FN을 담습니다.

## 출처
FitFuel, All Gym Equipment v2, Roboflow Universe (2024), CC BY 4.0. 클래스 필터·라벨 매핑·그룹 분할 변경.
https://universe.roboflow.com/fitfuel/all-gym-equipment/dataset/2
https://docs.ultralytics.com/integrations/litert

## 앱 연결
- Python: skku_inference.py의 predict_photo(image_path, best_pt_path).
- HTTP: server.py의 /predict. FastAPI TestClient로 사진 전송과 JSON 응답 확인.
- flutter_http_example.dart는 기존 HTTP 연결 예제이며 온디바이스 구현 또는 실행 완료를 뜻하지 않습니다.
- 출력: class_id, class_name, confidence, bbox_xyxy_pixels = [x1,y1,x2,y2]; 원본 이미지 좌표.
- 모바일 내보내기 및 실제 텐서 계약: tensor_contract.json, conversion_verification.json 확인.
- EXIF 방향 보정 후 RGB, 종횡비 유지 letterbox 640, 패딩 114, float32/255. 실제 텐서 layout은 tensor_contract.json 기준.
- YOLO11 출력은 4개 좌표 + 5개 클래스 점수. 별도 objectness는 없습니다. 클래스별 NMS 후 letterbox 패딩을 빼고 resize 비율로 나눕니다.
- 카메라 YUV→RGB, 회전, 미러링, 화면 미리보기 좌표 변환은 앱에서 별도로 검증해야 합니다.

실제 내보내기 계약:
```json
{
  "inputs": [
    {
      "name": "serving_default_args_0",
      "shape": [
        1,
        3,
        640,
        640
      ],
      "dtype": "<class 'numpy.float32'>",
      "quantization": [
        0.0,
        0
      ]
    }
  ],
  "outputs": [
    {
      "name": "serving_default_output_0_output",
      "shape": [
        1,
        9,
        8400
      ],
      "dtype": "<class 'numpy.float32'>",
      "quantization": [
        0.0,
        0
      ]
    }
  ],
  "classes": {
    "0": "treadmill",
    "1": "seated_chest_press",
    "2": "shoulder_press",
    "3": "seated_leg_press",
    "4": "smith_machine"
  },
  "preprocessing": "EXIF upright RGB, letterbox 640x640 padding 114, float32 /255; use actual input tensor layout",
  "scope": "public_proxy_not_center",
  "nms": "external class-aware NMS; confidence 0.25, IoU 0.7",
  "flutter_device_tested": false,
  "bbox_units": "normalized to 640x640 canvas",
  "output_interpretation": "[batch,9,8400]: cx,cy,w,h,5 class probabilities; no separate objectness; transpose anchors, choose max class, threshold and class-aware NMS, then undo letterbox"
}
```

Flutter 온디바이스 예제: flutter_litert_example.dart. 입력/출력 형태 검사와 클래스별 NMS를 포함합니다. Dart 컴파일 및 휴대폰 실행은 수행하지 않았습니다. tflite_flutter 네이티브 런타임과 모델 연산자 호환성은 앱팀 검증이 필요합니다.
참고: https://github.com/tensorflow/flutter-tflite

실행 환경 참고: API 패키지 설치 시 xdsl 0.28.0과 typing-extensions 4.16.0의 의존성 경고가 발생했습니다. 기존 LiteRT 변환과 이후 interpreter/API 실제 호출은 성공했습니다. requirements_executed.txt에 설치 버전을 기록했습니다.
