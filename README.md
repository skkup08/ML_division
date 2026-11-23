# 이가온 · 운동기구 인식과 운동 자세 분석

성균관대 1398피트니스센터의 운동기구를 사진에서 찾는 모델과, 운동 영상을 포즈 좌표로 변환해 동작을 분석한 Colab 작업을 담았다. 기구 인식은 사진 속 기구의 위치와 종류를 출력한다. 자세 분석은 운동별로 학습 자료와 평가 범위가 다르다.

## 운동기구 사진 인식

26종 YOLO11n 재학습에 이미지 3,537장을 사용했다. 학습 3,128장, 검증 204장, 테스트 205장이다. 저장된 PyTorch 체크포인트의 테스트 결과는 다음과 같다.

| Precision | Recall | mAP50 | mAP50–95 |
| ---: | ---: | ---: | ---: |
| 0.9350 | 0.8694 | 0.9185 | 0.7967 |

일부 기구는 테스트 사진이 한 장뿐이다. 기구별 AP와 표본 수는 [테스트 결과](SKKU_1398_Fitness_ML/03_training_runs/equipment26_min100_retrain_20261101T020110Z/test_metrics.json)에 함께 기록했다.

| 용도 | 파일 |
| --- | --- |
| 학습 과정 확인 | [26종 재학습 Colab 노트북](SKKU_1398_Fitness_ML/01_notebooks/equipment26_min100_round1_20261101.ipynb) |
| Python 모델 사용 | [재학습 best.pt](SKKU_1398_Fitness_ML/03_training_runs/equipment26_min100_retrain_20261101T020110Z/fit/weights/best.pt) |
| 앱에 모델 탑재 | [FP32 TFLite 모델](SKKU_1398_Fitness_ML/04_exports_and_reports/equipment26_tflite_20261101T034127Z/equipment26_fp32.tflite) · [라벨 순서](SKKU_1398_Fitness_ML/04_exports_and_reports/equipment26_tflite_20261101T034127Z/labels.txt) |
| 앱 입출력 확인 | [텐서 규격](SKKU_1398_Fitness_ML/04_exports_and_reports/equipment26_tflite_20261101T034127Z/input_output_spec.json) · [변환 후 검증 결과](SKKU_1398_Fitness_ML/04_exports_and_reports/equipment26_tflite_20261101T034127Z/validation_report.json) |
| 데이터 준비 | [데이터셋 구축·선별 기록](SKKU_1398_Fitness_ML/02_dataset) · [5종 초기 프로젝트와 테스트](SKKU_1398_Fitness_ML/01_notebooks/SKKU_1398_Colab_Project) |

TFLite 모델은 `float32 [1, 3, 640, 640]`을 받아 `float32 [1, 30, 8400]`을 반환한다. 8,400개 후보마다 박스 좌표 4개와 기구별 점수 26개가 있다. 앱에서 신뢰도 필터와 중복 박스 제거(NMS)를 적용한다.

## 운동 자세 분석

11월 23일 기준으로 [부위별 폴더](Exercise_AI_Models/Equipment26)에 운동별 자세 모델 17개와 설정 파일 17개가 있다. 기존 11개에 22개 운동을 재시도해 얻은 6개를 더한 수치다. 재시도한 나머지 16개는 자료 또는 유효 관절 샘플이 부족했다. 17개 모두 정상으로 간주한 자세만 학습한 `GOOD_ONLY` 모델이다. 잘못된 자세를 구별하는 성능은 별도 평가가 필요하다.

| 용도 | 자료 |
| --- | --- |
| 어떤 운동이 학습됐는지 확인 | [운동별 상태와 자료 수](Allinone/Exercise_AI_26_Training_Status_Audit_20261123.md) |
| 최근 실행 확인 | [22종 재시도 노트북과 최종 결과표](Colab_Notebooks/Exercise_AI_22_Retry_FIXED_Colab_copy.ipynb) |
| 이전 실험 확인 | [데드리프트·스쿼트 등 학습 코드와 평가 자료](Allinone) · [동일 영상 그룹을 분리한 자세 분류 실험](Exercise_AI_New_20261117_RF_GroupHoldout_v2) |

## 파일 사용

저장소를 내려받은 뒤, 나누어 저장된 초기 Allinone 모델 묶음 두 개는 저장소 최상위에서 다음 명령으로 복원할 수 있다. 스크립트가 각 파일의 SHA-256을 검증하고 기존 Colab 코드에서 사용하는 위치에 저장한다.

```bash
python restore_models.py
```

Colab 노트북은 원래 Google Drive 폴더 구성을 사용한다. 학습을 다시 실행하려면 해당 노트북에서 지정한 데이터 경로에 원본을 준비해야 한다. 이 저장소에는 코드, 모델, 데이터 출처·분할 기록과 평가 결과를 수록했다. 학습용 사진·영상 원본은 별도로 관리한다. 노트북의 임시 실행 출력은 정리했고, 11월 23일 재시도 노트북의 최종 상태표는 보존했다.
