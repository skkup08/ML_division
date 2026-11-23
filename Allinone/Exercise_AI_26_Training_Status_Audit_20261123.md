# 운동 자세 AI 학습 상태 점검 | 2026-11-23 (KST)

## 핵심 결론
- 프로젝트 분모는 **26개 기구 항목**. 검토한 저장 결과에서 정확한 기구 동작과 검수된 정상·오류 라벨, 참가자 분리 평가가 모두 확인된 항목은 **0/26**. 별도 일반 운동 모델 및 후속 실행은 합산하지 않는다.
- 코랩 재시도 **22개 실행 / 6개 모델 생성 / 16개 미학습(자료 없음 5, 관절 부족 11)**. 신규 6개 전부 정상으로 추정한 자세만 학습한 `TRAINED_GOOD_ONLY`, 신규 `TRAINED_GOOD_BAD`는 **0**.
- 보존한 기존 11개 + 신규 6개 = **17개 GOOD_ONLY 모델 파일**. 33은 코랩의 운동 키 수(11+22)다. 17/33은 모델 파일 생성 현황이며 자세 판별의 17/26 완료를 뜻하지 않는다. 설치 기구 적합성과 실제 BAD 판별 성능은 미확인이다.

## 1. 코랩 사본에서 확인한 22개 운동
[실행 출력이 저장된 코랩 사본](../Colab_Notebooks/Exercise_AI_22_Retry_FIXED_Colab_copy.ipynb). Drive에 저장된 사본의 셀 47 결과표와 셀 49 정리 로그를 확인했다. 사본과 코랩 모델은 Allinone 직속이 아닌 별도 위치에 있어 해당 원본 링크에서 대조했다.

| # | 운동 | 코드 키 | 상태 | 고유 자료 | 유효 관절 샘플 | 모델 |
|---:|---|---|---|---:|---:|---|
| 1 | 트레드밀 걷기 | `treadmill_walk` | `NOT_TRAINED_NO_DATA` | 0 | 0 | 없음 |
| 2 | 리컴번트 바이크 | `recumbent_cycling` | `NOT_TRAINED_INSUFFICIENT_POSES` | 2 | 2 | 없음 |
| 3 | 업라이트 바이크 | `upright_cycling` | `NOT_TRAINED_INSUFFICIENT_POSES` | 2 | 2 | 없음 |
| 4 | 일립티컬 | `elliptical_training` | `NOT_TRAINED_INSUFFICIENT_POSES` | 7 | 5 | 없음 |
| 5 | 시티드 체스트 프레스 | `seated_chest_press` | `NOT_TRAINED_INSUFFICIENT_POSES` | 4 | 4 | 없음 |
| 6 | 펙덱 플라이 | `pec_deck_fly` | `TRAINED_GOOD_ONLY` | 28 | 28 | [파일](../Exercise_AI_Models/Equipment26/Chest/pec_deck_fly.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/pec_deck_fly.json) |
| 7 | 롱 풀 시티드 케이블 로우 | `seated_cable_row` | `NOT_TRAINED_INSUFFICIENT_POSES` | 3 | 3 | 없음 |
| 8 | 머신 암 컬 | `machine_biceps_curl` | `NOT_TRAINED_INSUFFICIENT_POSES` | 2 | 2 | 없음 |
| 9 | 머신 앱도미널 크런치 | `abdominal_crunch` | `NOT_TRAINED_INSUFFICIENT_POSES` | 4 | 4 | 없음 |
| 10 | 로터리 토르소 | `rotary_torso` | `NOT_TRAINED_NO_DATA` | 0 | 0 | 없음 |
| 11 | 레그 익스텐션 | `leg_extension` | `TRAINED_GOOD_ONLY` | 201 | 25 | [파일](../Exercise_AI_Models/Equipment26/Legs/leg_extension.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Legs/leg_extension.json) |
| 12 | 시티드 레그 프레스 | `seated_leg_press` | `NOT_TRAINED_NO_DATA` | 0 | 0 | 없음 |
| 13 | 스미스 스쿼트 | `smith_squat` | `NOT_TRAINED_INSUFFICIENT_POSES` | 4 | 4 | 없음 |
| 14 | 더블 트위스트 | `double_twist` | `NOT_TRAINED_NO_DATA` | 0 | 0 | 없음 |
| 15 | 플랫 벤치프레스 | `flat_bench_press` | `TRAINED_GOOD_ONLY` | 212 | 35 | [파일](../Exercise_AI_Models/Equipment26/Chest/flat_bench_press.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/flat_bench_press.json) |
| 16 | 인클라인 벤치프레스 | `incline_bench_press` | `TRAINED_GOOD_ONLY` | 216 | 34 | [파일](../Exercise_AI_Models/Equipment26/Chest/incline_bench_press.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/incline_bench_press.json) |
| 17 | 디클라인 벤치프레스 | `decline_bench_press` | `TRAINED_GOOD_ONLY` | 193 | 12 | [파일](../Exercise_AI_Models/Equipment26/Chest/decline_bench_press.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/decline_bench_press.json) |
| 18 | 머신 체스트 프레스 | `machine_chest_press` | `NOT_TRAINED_INSUFFICIENT_POSES` | 4 | 4 | 없음 |
| 19 | 머신 체스트 플라이 | `chest_fly` | `TRAINED_GOOD_ONLY` | 197 | 28 | [파일](../Exercise_AI_Models/Equipment26/Chest/chest_fly.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/chest_fly.json) |
| 20 | 하이 투 로우 플라이 | `high_to_low_fly` | `NOT_TRAINED_NO_DATA` | 0 | 0 | 없음 |
| 21 | 로우 투 하이 플라이 | `low_to_high_fly` | `NOT_TRAINED_INSUFFICIENT_POSES` | 2 | 2 | 없음 |
| 22 | 인클라인 덤벨 플라이 | `incline_dumbbell_fly` | `NOT_TRAINED_INSUFFICIENT_POSES` | 2 | 2 | 없음 |

**오류 로그:** `NOT_TRAINED_NO_DATA`는 사용한 소스에서 매칭 자료를 못 찾은 상태다. `NOT_TRAINED_INSUFFICIENT_POSES`는 유효 샘플 2~5개로 최소 8개에 미달한 상태다. Kaggle 인증·다운로드는 동작했다. 펙덱 셀에 일시적으로 `ROBOFLOW_API_KEY missing`가 나왔지만 이후 레그 익스텐션 셀에서 Roboflow 분류 자료 다운로드에 성공했고 후속 셀에서도 재사용됐다. 전체 실패를 인증 문제로 해석하지 않는다. 분류 라벨은 자세 GOOD/BAD 라벨이 아니다. 셀 49는 런타임의 임시 다운로드만 삭제하고 Drive 모델을 보존했다는 기록이다.

## 2. 이전에 생성되어 보존된 11개
코랩 모델 폴더. 기존 11개와 신규 6개의 모델·JSON 파일 존재를 대조했으며 메타데이터는 모두 `TRAINED_GOOD_ONLY`. 모델 파일을 다시 실행하지 않았다.

| 운동 | 코드 키 | 상태 | 모델 |
|---|---|---|---|
| 하이 플라이 | `high_fly` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Chest/high_fly.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/high_fly_info.json) |
| 머신 숄더 프레스 | `machine_shoulder_press` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Shoulder/machine_shoulder_press.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Shoulder/machine_shoulder_press_info.json) |
| 친업 | `chin_up` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Back/chin_up.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Back/chin_up_info.json) |
| 레그 컬 | `leg_curl` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Legs/leg_curl.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Legs/leg_curl_info.json) |
| 케이블 크로스오버 | `cable_crossover` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Chest/cable_crossover.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/cable_crossover_info.json) |
| 메가맥스 랫풀다운 | `mega_max_lat_pulldown` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Back/mega_max_lat_pulldown.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Back/mega_max_lat_pulldown_info.json) |
| 딥 스테이션 딥스 | `dip_station_dip` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Chest/dip_station_dip.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Chest/dip_station_dip_info.json) |
| 파워 레그 프레스 | `power_leg_press` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Legs/power_leg_press.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Legs/power_leg_press_info.json) |
| 파워랙 스쿼트 | `power_rack_squat` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Legs/power_rack_squat.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Legs/power_rack_squat_info.json) |
| 싯업 | `sit_up` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Core/sit_up.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Core/sit_up_info.json) |
| 하이퍼 익스텐션 | `hyperextension` | `TRAINED_GOOD_ONLY` | [파일](../Exercise_AI_Models/Equipment26/Back/hyperextension.joblib) · [메타데이터](../Exercise_AI_Models/Equipment26/Back/hyperextension_info.json) |

## 3. 26개 기구의 저장 결과 대조
26항목 매핑 근거. 단일 동작 모델을 복합기구 전체로 확대하지 않고, 설치 모델이 확인되지 않은 곳은 후보로 남긴다.

| # | 기구 | 상태 | 근거·제한 |
|---:|---|---|---|
| 1 | Treadmill | 자료 없음 | treadmill_walk (걷기만 시도) |
| 2 | Recumbent Bike | 관절 부족 | recumbent_cycling |
| 3 | Upright Bike | 관절 부족 | upright_cycling |
| 4 | Elliptical | 관절 부족 | elliptical_training |
| 5 | Seated Chest Press | 관절 부족 | seated_chest_press와 machine_chest_press 모두 부족. 별도 기구인지 미확정. |
| 6 | Pectoral Fly | GOOD_ONLY 후보 | pec_deck_fly와 chest_fly 모두 같은 Kaggle 운동 검색어와 28 관절 샘플. 독립 2종으로 세지 않음. |
| 7 | High Fly | 매핑 미확정 | high_fly 파일 존재. 랫풀다운 후보이며 가슴 플라이로 자동 연결하지 않음. |
| 8 | Shoulder Press | GOOD_ONLY 후보 | machine_shoulder_press |
| 9 | Long Pull | 관절 부족 | seated_cable_row 후보; 유산소 로잉과 구분 |
| 10 | Arm Curl | 관절 부족 | machine_biceps_curl; 일반 컬은 별개 |
| 11 | Abdominal | 관절 부족 | abdominal_crunch; 일반 side_bend 또는 sit_up과 구분 |
| 12 | Chin Up | GOOD_ONLY / 변형 미확정 | chin_up; 보조형 여부 미확정 |
| 13 | Rotary Torso | 자료 없음 | rotary_torso |
| 14 | Leg Extension | GOOD_ONLY 후보 | leg_extension |
| 15 | Leg Curl | GOOD_ONLY / 변형 미확정 | leg_curl; 좌식/누운 자세 등 기구 변형 확인 |
| 16 | Seated Leg Press | 자료 없음 | seated_leg_press; 파워 레그 프레스와 구분 |
| 17 | Cable Cross Over | GOOD_ONLY 후보 | cable_crossover 보존. high_to_low_fly 및 low_to_high_fly의 별도 시도는 미학습. |
| 18 | Smith Machine | 관절 부족 | smith_squat만 시도; 자유중량 스쿼트와 다름 |
| 19 | Mega Max | GOOD_ONLY / 변형 미확정 | mega_max_lat_pulldown; 설치 복합기구 동작 미확정 |
| 20 | Chinning Dipping Hip Flexor | GOOD_ONLY / 부분 | dip_station_dip 한 동작만 |
| 21 | Power Leg Press | GOOD_ONLY 후보 | power_leg_press |
| 22 | Power Rack | GOOD_ONLY 후보 | power_rack_squat 기존 파일. 별도 신규 실험은 이 저장본 집계에 포함하지 않음. |
| 23 | Sit Up | GOOD_ONLY 후보 | sit_up |
| 24 | Double Twist | 자료 없음 | double_twist; Rotary Torso와 구분 |
| 25 | Hyper Extension | GOOD_ONLY 후보 | hyperextension |
| 26 | Super Bench Press (Incline/Decline) | GOOD_ONLY 후보 | incline/decline은 한 기구 항목의 2개 동작 후보. flat_bench_press는 별도. |

## 4. Allinone 기존 모델·결과표·오류 로그
- Allinone 루트의 [fitness_models.joblib](fitness_models.joblib)은 별도 `fitness_consolidated_v4` 실행. [training_status.json](training_status.json)은 `completed`, 종료 코드 0. [capabilities.json](capabilities.json) / [metrics.json](metrics.json)에 자세 학습이 확인된 일반 운동 7종: **벤트오버 로우, 바이셉 컬, 푸시업, 숄더 프레스, 사이드 벤드, 스쿼트, 삼두 익스텐션**. 플랭크와 바이셉 컬의 제한된 오류 분류도 별도 기록. 기구 26종 결과로 계산하지 않는다.
- 루트 run.log, 영상 분할표, 실패/제외 영상 로그: 유효 828개 영상·26 그룹(train 497 / val 157 / test 174). 영상 제외 214건(여러 사람으로 운동 대상 불명 172, 라벨 충돌 21, 관절 부족 13, 동일 해시 중복 8). 이 214는 영상 건수로 코랩 16개 운동 미학습과 별개다.
- [기존 확장 학습 결과](Other_Exercises_Expanded_20261027/run_20261027_142919/FINAL_RESULTS_KO.txt): 운동 인식 23종과 일반 운동 GOOD/BAD 7종은 다르다. 같은 Mendeley 자료를 재사용했으며, 독립 외부 평가나 기구 26종 달성은 아니다.
- 기존 데드리프트 V6 모델 및 기존 37동작 커버리지는 각각 별도 보존. 2026-10-26 별도 v6_mydeadlift 실행 폴더는 `partial`, 종료 코드 1이며 루트 v4 성공과 구분한다. 37동작 레지스트리는 8개 연결·29개 미연결이며 신규 학습 완료 수가 아니다. 같은 파일명의 여러 버전은 합치지 않았다.

## 5. 후속 학습
1. 이번 미학습 16키를 기구 목록에 먼저 연결한다. 자료가 없는 5키는 실제 기구·동작이 맞는 자료부터 확보하고, 나머지 11키는 원본과 관절 추출 실패를 나눠 보완한다.
2. 기존 파일이 있으면 코랩이 재학습을 건너뛰며 이전 11키는 재시도 목록에서 제외된다. 검수 GOOD/BAD를 추가할 때는 이 노트북 전체 재실행 대신 별도 버전을 쓴다.
3. 원본 출처·사용 권한, 장비·동작, 검수 라벨과 참가자 ID가 있는 자료로 학습한다. 같은 사람이 학습·평가에 동시에 들어가지 않도록 분리하고 정상·오류 각각의 평가량과 혼동행렬을 남긴다.
4. 이 감사에서 기존 원본, 모델, 노트북 또는 로그를 삭제·덮어쓰지 않았다. 감사 파일만 신규 저장한다.

