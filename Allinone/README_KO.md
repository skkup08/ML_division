
# 운동 자세 AI — Google Colab 한 셀 통합본

## 처음 실행
`fitness_all_in_one_colab.ipynb`를 코랩에 업로드해 코드 셀 하나를 실행합니다. Google Drive 연결을 승인하세요. 기존 셀에 덧붙이는 패치가 아니라 완전한 교체본입니다.
`fitness_all_in_one_colab.txt` 또는 `.py`의 전체 내용을 새 코드 셀에 붙여넣어도 같습니다. 다른 파일을 미리 올릴 필요가 없습니다. MediaPipe/ExtraTrees 학습은 CPU를 사용합니다.

기본 흐름: Drive 쓰기 검사 → 독립 패키지 환경 준비 → 실제 MediaPipe 사전 검사 → 임시 데이터 다운로드 → 영상별 관절 추출 → 운동/Good-Bad/제한된 오류 모델 학습 → 평가 → 모델·보고서·ZIP Drive 저장 → 임시 원본 삭제.

## 무엇을 학습하나
- 기본: Mendeley `kgbb3yn47p` 버전 3의 7종 운동, Good/Bad 라벨. 동일 영상을 묶은 ZIP 세 가지 중 Subject-wise 하나만 선택합니다.
- 기본: Exercise-Correction 공개 관절 CSV를 이용한 플랭크 골반 높음/처짐, 바이셉 컬 상체 뒤로 젖힘. 배포된 pickle 모델은 실행하지 않고 분류기를 새로 학습합니다.
- 선택: `USE_UCF101=True`일 때만 공식 UCF101 RAR를 받아 8종 운동을 운동 인식용으로 사용합니다. 전체 101종을 학습하거나 임의 Good/Bad를 붙이지 않습니다. 이전 TLS 인증서 오류 때문에 기본값은 False이며 인증서 검증을 끄지 않습니다. SSL 오류를 해결했다고 주장하지 않습니다.
- 승인받은 Fitness-AQA, 이용 가능한 Kinetics/InfiniteRep/Kaggle 등의 **영상과 실제 라벨**은 `EXTRA_MANIFEST`로 추가할 수 있습니다. 기본값에서는 자동 포함되지 않습니다. Fitness-AQA는 저자 접근 승인과 비상업 조건이 있습니다.
- 이전에 언급된 Zenodo 17558630은 정지 이미지 데이터입니다. 이 영상 전용 입력기에 이미지를 영상처럼 넣지 않으며 기본 실행에 포함하지 않습니다.
- PosePilot 같은 프로그램 저장소는 그 자체로 학습 데이터가 아닙니다. 프로그램을 복사하는 것을 데이터 학습으로 표시하지 않습니다.

**위 목록 전체의 모든 자료를 자동으로 내려받는 코드가 아닙니다. 실제 사용 소스는 source_status.json을, 실제 학습 기능은 capabilities.json을 확인하세요.**

## 기본 설정
```python
MODE = "auto"                  # 처음엔 학습, 이후 같은 설정의 완성 모델을 재사용
USE_MENDELEY = True
USE_CORRECTION_CSV = True
USE_UCF101 = False
DELETE_TEMP_DATA = True
DRIVE_RESULTS_ROOT = "/content/drive/MyDrive/fitness_ai_results"
```
`MODE="train"`은 항상 새로 학습합니다. `MODE="predict"`는 이 통합본으로 만든 완성 모델만 재사용하고, 시험 영상이 없으면 업로드 창을 엽니다.
`MODEL_DIR`을 비우면 호환 모델을 자동 탐색합니다. 완성 상태·설정·소스 버전·모델 파일 해시를 확인합니다. 이전 패치 버전의 모델은 자동 재사용하지 않습니다. 외부 데이터 경로를 지정한 auto 실행은 원본 변경 여부를 추측하지 않고 새로 학습합니다.

`EXISTING_VIDEO_DIR`에 이미 받은 Mendeley 영상 폴더를 입력하면 다운로드 없이 임시 복사해 사용합니다. 원래 subject/운동명/good-bad 파일명·구조를 유지해야 합니다. 사용자 원본은 삭제하지 않습니다.

## 모델 구조와 한계
사전학습된 MediaPipe Pose Landmarker에서 33개 관절을 얻은 뒤 얼굴 관절은 분류 특징에서 제외합니다. 위치를 몸통 중심·크기로 정규화하고, 신체 위치·각도·시간 변화 특징을 계산합니다. 외모 RGB 전체를 분류기에 입력하지 않지만 사람의 체형/환경 영향이 완전히 사라진다고 보장하지 않습니다.

새로 학습하는 부분은 ExtraTrees 기반 운동 인식, 운동별 Good/Bad, 실제 오류 라벨이 있는 경우의 오류 분류기입니다. 대형 영상 신경망이나 MediaPipe 자체를 처음부터 학습하지 않습니다.

3차원 관절 추정이 부족해 시간 특징이 없는 영상도, 충분한 영상 좌표가 있으면 영상 단위 좌표 요약을 이용한 Good/Bad 보조 모델 학습에 활용합니다. 이것은 3차원 물리적 각도를 대체하여 정확하게 측정하는 것이 아닙니다. 두 모델이 강하게 상충하면 판단을 보류합니다.

영상당 최대 96프레임, 목표 6fps가 기본값입니다. 시간 전체의 모든 순간을 검사하지 않으므로 짧은 오류는 놓칠 수 있습니다. 한 영상에 한 사람·한 운동이 전제입니다. 여러 사람이 보이면 대상자를 임의로 선택하지 않습니다.

## Good/Bad와 교정 안내
Mendeley의 Good/Bad 라벨만으로 구체적 오류 원인을 만들어내지 않습니다. Bad이지만 세부 오류 모델이 없으면 '원인 진단이 아닌 일반 점검 목록'을 제공합니다.
세부 오류 모델이 실제로 검출한 경우에만 그 오류에 연결된 고정 코칭 문구를 반환합니다. 교정 문구를 별도의 언어 모델로 학습하는 것은 아닙니다. 오류를 못 찾았다는 이유만으로 Good를 부여하지 않습니다. 플랭크의 Good는 지원되는 골반 높음/처짐 오류 범위에 한정됩니다.
분류 점수의 임계값은 실험용 고정값이며 보정된 확률/안전 점수가 아닙니다. 지원하지 않는 운동의 확실한 검출기를 별도 학습한 것은 아니므로, 미지원 운동에도 잘못된 고확신 예측이 나올 수 있습니다. 통증이 있으면 동작을 멈추고 적절한 전문가에게 확인하세요. 의료·안전 인증 시스템이 아닙니다.

## 평가
Mendeley의 동일 사람과 그 사람의 서로 다른 시점/Good-Bad 영상은 학습·검증·테스트 사이에 섞지 않습니다. UCF101의 원본 그룹 분할은 사람 독립을 보장하지 않으며 공식 벤치마크 split과도 다릅니다.
공개 오류 CSV는 사람/원본 영상 ID가 없어 사람 독립 성능을 검증할 수 없습니다. 배포 train/test를 사용하고 좌표 완전 중복을 제거하되, 인접 프레임 누출 가능성까지 배제했다는 뜻은 아닙니다.
`test_predictions.csv`와 `metrics.json`에 결과를 남깁니다. `cascade_test_reference`는 원시 argmax 연쇄 분류 참고 평가이며, 실제 예측 함수의 점수/모델 충돌 보류 정책 전체를 적용한 평가는 아닙니다. 검증 점수를 서비스 정확도 보장으로 해석하지 마세요.

## 저장과 삭제
Drive 위치: `내 드라이브/fitness_ai_results/run_한국시간_고유번호/`.
- fitness_models.joblib: 새로 학습한 분류기들
- pose_landmarker_full.task: 추론에 필요한 사전학습 관절 모델
- metrics.json, test_predictions.csv: 평가
- capabilities.json, source_status.json, training_status.json, progress.json: 범위/소스/상태
- failed_or_skipped_videos.csv, run.log: 실패 원인과 실행 기록
- video_manifest_and_splits.csv: 파일명·라벨·분할 등 메타데이터 (원본 영상/좌표 아님)
- fitness_runtime.py, config.json, requirements.txt, requirements_lock.txt: 재사용 코드/설정
- artifact_integrity.json: 재사용 전 파일 해시 검사
- model_and_reports.zip: 완료 결과 묶음
- partial_or_failure_reports.zip: 중단/실패 결과 묶음
- drive_save_complete.json: ZIP 크기·해시 읽기 검증 결과

원본 ZIP/RAR, 원본 영상, 다운로드 CSV는 Drive에 저장하지 않습니다. 영상은 압축에서 하나씩 꺼내 처리 후 삭제하며, 원본 압축은 해당 소스 처리가 끝나면 삭제합니다. 훈련용 관절/특징 배열은 메모리에서만 사용하고 결과에 저장하지 않습니다. 일반 종료/오류/사용자 중단 시 코드가 만든 고유 임시 폴더만 삭제합니다. 강제 VM 종료나 프로세스 강제 종료에는 종료 정리 코드가 실행되지 않을 수 있습니다. 사용자가 올린 시험 영상·기존 원본 폴더는 삭제하지 않습니다.

CSV 오류 분류·운동 인식·핵심 모델처럼 학습이 완료된 단계의 모델은 중간에도 저장합니다. 그러나 **관절 추출이나 진행 중인 학습을 중단 지점부터 재개하는 기능은 아닙니다**. 단계별 모델 저장은 원본/특징 캐시 저장과 다릅니다. 중단된 실행은 자동 재사용하지 않습니다.
Drive 저장은 코랩 무중단 실행 보장이 아닙니다. PC를 끄면 반드시 완료되는 구조가 아니고 런타임 종료 시 작업이 멈춥니다. 원본/특징을 버리므로 모델 완성 전에 끊기면 새 학습에서 다시 내려받아야 합니다. 완성 모델은 MODE=auto/predict로 재사용하므로 재다운로드하지 않습니다.
PC로 결과 ZIP을 자동 다운로드하지 않습니다. Drive 마운트 경로 검사는 독립적인 클라우드 동기화 보증이 아니므로 Drive 웹에서도 결과를 확인하세요.

## 새 영상 예측
한 셀에서 끝내려면 맨 위의 `UPLOAD_TEST_VIDEO=True`로 실행합니다. 처음에는 학습 후, 다음에는 완성 모델 재사용 후 업로드 영상을 검사합니다. 파일 경로를 아는 경우 `TEST_VIDEO_PATH`로 지정할 수도 있습니다.

같은 코랩 세션의 이후 예측:
```python
fitness_upload_and_predict()
fitness_predict('/content/new_video.mp4')
fitness_predict('/content/my_plank.mp4', exercise='plank')
```
예측 JSON은 결과 폴더의 `predictions/`에 저장합니다. 원본 시험 영상은 Drive에 복사하지 않습니다. 새 예측 JSON은 학습 당시 만든 ZIP을 자동 갱신하지 않으므로 predictions 폴더에서 확인하세요.
`fitness_predict`는 결과 딕셔너리도 반환합니다. 본인이 생성하고 신뢰하는 joblib만 사용하세요. 해시 검사는 우발적 파일 손상을 확인하는 기능이지 악성 모델에 대한 신뢰 인증이 아닙니다.

## 추가 영상 CSV
필수 열: source, exercise, subject_id, split, quality, errors_json
위치 열: video_path 또는 video_url (둘 중 하나)
- source: 기본 소스와 겹치지 않는 이름
- exercise: squat, bicep_curl, push_up 등의 snake_case 이름
- quality: 실제 good/bad 또는 빈칸. 빈칸을 good으로 바꾸지 않습니다.
- subject_id: 동일 사람 식별 그룹. 사람 ID가 없으면 원본 영상 그룹을 쓰되 사람 독립 성능이라고 주장하지 마세요.
- split: train/validation/test 또는 빈칸. 한 소스는 모두 지정하거나 모두 빈칸이어야 합니다.
- errors_json: 명시적 0/1 라벨만. 예: {"squat_heel_lift": 1, "squat_knee_collapse": 0}. 키가 없으면 미상이지 정상(0)이 아닙니다.
video_url은 실제 영상의 HTTP(S) 직접 주소입니다. 로그인 페이지, YouTube 페이지, Drive 공유 웹페이지 URL은 지원하지 않습니다. 상대 video_path는 코랩 현재 작업 폴더 기준입니다. 사람 ID/라벨을 파일명에서 임의로 만들어 넣지 마세요.
오류 라벨의 정의를 CORRECTIONS 사전과 명시적으로 맞추어야 하며, 이름이 비슷하다는 이유로 서로 다른 의미를 합치지 않습니다. 데이터별 접근 권한·라이선스는 계속 적용됩니다.

## 이번 코드의 검사 범위
오프라인 문법, 실제 오류 로그의 파일명 해석, 특징 차원/정규화, 사람별 분할, 합성 자료의 7종 모델 학습, Good/Bad 없는 자료 제외, 모델 저장·재사용·무결성 검사, 원인 없는 Bad의 일반 안내, 구체적 오류 안내, 모델 충돌 보류, 로컬 가상 Drive 폴더 ZIP 복사 등을 검사했습니다.
실제 Colab Drive 연결, 실제 영상 MediaPipe 추론, 전체 원본 다운로드·학습은 이 대화 환경에서 수행하지 않았습니다. `code_validation_report.json`에 구분하여 기록합니다. 코드가 사용자 Colab에서 실제 환경 사전 검사를 수행하며, 이 검사를 통과한 후 대용량 영상을 받습니다.

## 원본 출처
- https://data.mendeley.com/datasets/kgbb3yn47p/3
- https://github.com/NgoQuocBao1010/Exercise-Correction
- https://github.com/ParitoshParmar/Fitness-AQA
- https://www.crcv.ucf.edu/data/UCF101.php
- https://github.com/cvdfoundation/kinetics-dataset
- https://zenodo.org/records/17558630
- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- https://matplotlib.org/stable/users/explain/figure/backends.html
- https://research.google.com/colaboratory/faq.html

