# SKKU 1398 Fitness · 기구 인식 학습 프로젝트

## 진행 상황 (2026-10-26)

- **실제로 완료:** 26개 기구 목록 CSV, 임시 MVP 5종 클래스 규격, 공개 데이터 라벨 매핑·검수·중복 검사 Python 코드, Colab `.ipynb`, CPU 환경에서 데이터 처리 단위 테스트, API 서버 예제.
- **아직 실행하지 않음:** 사용자 계정 Colab 런타임 연결/GPU 사용, 실제 Roboflow 원본 다운로드, 대상 기구 5종 YOLO 학습, 센터 실물 사진 평가, `.tflite` 변환, Flutter 앱 실기기 실행.
- **왜 아직인지:** 이 대화에서 Colab 브라우저 세션을 조작할 수 없으며 Roboflow 데이터 인증 키와 1398 센터 현장 사진도 제공되지 않았기 때문입니다. 이 파일은 학습 절차가 준비된 **실행 가능한 노트북**이지만 훈련이 끝난 모델 파일이 아닙니다.

## 1. 프로젝트 목표

센터 자료에 적힌 기구 26개를 **최종 클래스 수로 확정하지 않습니다**. 2주 MVP는 대략 5개 클래스 분류 및 객체탐지 완주에 집중합니다.

초기 5종:

| 모델 클래스 ID | 코드 클래스 | 센터 원문 명칭 | Roboflow 일반 데이터 명칭 |
|---|---|---|---|
| 0 | treadmill | TREDMILL | Treadmill |
| 1 | seated_chest_press | SEATED CHEST PRESS | Chest Press Machine |
| 2 | shoulder_press | SHOULDER PRESS | Shoulder Press Machine |
| 3 | seated_leg_press | SEATED LEG PRESS | Leg Press Machine |
| 4 | smith_machine | SMITH MACHINE | Smith Machine |

**주의:** 일반 체스트프레스/레그프레스 클래스와 센터의 시티드 전용 기구는 외형상 반드시 일치하는 것은 아닙니다. 이 공개 학습 결과를 *센터 최종 모델*이라고 부르지 마세요. `TREDMILL`은 센터 목록 원문 그대로 기록했습니다.

## 2. 직접 실행할 Colab 노트북

`SKKU_1398_5class_Colab_Training.ipynb`를 [Google Colab](https://colab.research.google.com/)에서 업로드해 실행합니다.

1. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다. Colab 유료 사용자는 실제 할당 GPU를 코드 셀에서 확인합니다.
2. 셀 1의 `RUN_MODE="smoke"`는 **소량 데이터 학습 2 epochs**로 연결 시험만 합니다. 결과의 mAP를 발표 자료의 실성능으로 쓰지 마세요.
3. Colab **열쇠 아이콘(비밀 키)**에 `ROBOFLOW_API_KEY`를 저장하고 노트북 접근을 허용합니다. 비밀 키를 코드, 공유 노트북, 채팅에 복사하지 마세요.
4. Google Drive 마운트를 승인합니다. 기존 폴더 `MyDrive/SKKU_1398_Fitness_ML`가 있어야 합니다.
5. 데이터 다운로드·라벨 검사·정답 박스 시각화 셀까지 실행합니다. 오라벨을 눈으로 검수하고, 오류 CSV를 읽습니다.
6. `smoke` 학습 완료 후 `best.pt`가 Drive `03_training_runs/<run_name>`에 생성됐는지 확인합니다.
7. `RUN_MODE="baseline"`으로 새로 실행하고 전체 공개 데이터 기준 모델 학습 및 공개 데이터 테스트를 수행합니다.
8. 공개 데이터에서 동작하는지 확인한 뒤 **실제 센터의 사진을 추가 촬영·라벨링**하고 현장 학습/독립 테스트를 별도로 진행합니다.
9. 실제 모델을 확보한 뒤 `RUN_MOBILE_EXPORT=True`로 바꿔 LiteRT/`.tflite` 내보내기를 시도합니다. 실기기 실행·박스 좌표 검증은 Flutter 측에서 추가로 해야 합니다.

Roboflow 키를 제공하기 어렵다면 직접 다운로드한 **YOLOv8 라벨 ZIP**을 Drive `02_dataset`에 넣고 `DATASET_ZIP_NAME="파일명.zip"`을 사용합니다. `data.yaml`과 원본의 `train/valid/test` 이미지·라벨 경로를 유지해야 합니다.

## 3. Colab 코드가 무엇을 저장하나요?

- 원본 클래스명 → 센터 MVP 5종 클래스 매핑
- `dataset_audit.json`, `manifest.csv`, `label_issues.csv`, `potential_near_duplicates.csv`
- 정답 검수 이미지 `annotation_review.jpg`
- 학습된 `best.pt`, checkpoint `last_checkpoint.pt`, `results.csv`
- 원본 테스트셋 결과 `public_test_metrics.json`, 예측 화면 이미지와 JSON
- 조건부 모바일 `.tflite` 내보내기

## 4. 데이터 주의사항

- FitFuel v2에는 공개 설명상 기구 29개 범주가 있습니다. v2 페이지는 총 4,717장(학습 4,341장, 검증 334장, 테스트 42장)을 기록하며 학습 증강을 포함합니다. 다만 **이 프로젝트의 5종 필터링 이후 수량은 Colab에서 직접 검사해야 합니다**.
- 동일 사진 SHA256 중복은 제거하지만 비슷한 영상 프레임의 완전한 독립성은 보장되지 않습니다. 유사 프레임 경고 CSV를 보고 필요 시 분할을 다시 해야 합니다.
- 테스트 42장에서도 5종 목표 클래스가 적을 수 있습니다. 실제 센터의 **서로 다른 날짜·세션 촬영 이미지**를 반드시 최종 테스트로 따로 둡니다.
- 공개 데이터 학습 성공 ≠ 센터에서 잘 작동함. 1398 실제 기구의 제조사·색상·촬영 조건을 검증해야 합니다.
- 기관 내 촬영 규정과 타인 초상권을 확인해 사람이 촬영되지 않도록 하거나 필요한 동의를 확보하세요.

## 5. 앱 예제

개발용 서버를 시작하려면 `src` 폴더에서:

```bash
pip install 'ultralytics>=8.3,<9' fastapi uvicorn python-multipart pillow numpy
SKKU_MODEL_PATH=/path/to/best.pt uvicorn server:app --host 127.0.0.1 --port 8000
```

`POST /predict` 에 `multipart/form-data` 파일 필드 `file`을 보내면 `status`, `image_width`, `image_height`, `detections[]`의 `class_id`, `class_name`, `confidence`, `bbox_xyxy_pixels`를 반환합니다. 최대 8 MiB JPEG/PNG/WebP를 받습니다.

`docs/flutter_http_example.dart`는 **서버 추론 방식 테스트**입니다. 프로젝트 계획의 핵심인 **휴대폰 자체 추론(on-device)** 은 실제 모델 내보내기 및 Flutter 실행 패키지 확인 이후에 따로 작업해야 합니다. 인터넷에 API를 공개하려면 인증·HTTPS·율 제한과 자원 제한을 먼저 설계하세요.

## 6. 파일 구성

```text
SKKU_1398_5class_Colab_Training.ipynb
src/skku1398_ml.py
src/server.py
src/__init__.py
docs/1398_equipment_catalog.csv
docs/class_contract.json
docs/flutter_http_example.dart
tests/test_pipeline.py
README_KO.md
```

## 7. 출처 및 라이선스

- 성균관대 1398피트니스센터 기구 명칭: 사용자가 제공한 센터 리플렛 목록 (센터 설치 현황은 미검증)
- 학습 후보: [FitFuel All Gym Equipment, Version 2](https://universe.roboflow.com/fitfuel/all-gym-equipment/dataset/2). **CC BY 4.0**, 원저작자 **FitFuel**. 학습 데이터/결과물을 공개할 때 저작자·라이선스·원본 출처를 명시하고 변경 여부를 표시해야 합니다.
- YOLO11n: [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/). 사용·재배포·상업화 전에 Ultralytics 라이선스 조건을 확인해야 합니다.
- 모바일 내보내기: [Ultralytics model export](https://docs.ultralytics.com/modes/export/).
- Colab 실행 환경: [Google Colab FAQ](https://research.google.com/colaboratory/faq.html?hl=ko).

### 프로젝트 Google Drive

SKKU_1398_Fitness_ML: 현재 폴더 **4개 생성·검증** 완료. **이 패키지 파일은 아직 Google Drive에 업로드되지 않았습니다**(이 채팅의 Drive 연결로는 로컬 바이너리 업로드 경로를 확보할 수 없었습니다).
