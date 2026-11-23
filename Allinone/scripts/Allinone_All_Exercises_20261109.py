# Allinone: 전체 운동 모델 연결 + 보유 자료 추가 학습 + 피드백 화면
# 실행 자체는 사용자의 Colab에서 합니다. 모든 37개 운동의 학습 완료를 주장하지 않습니다.
# 인터넷 운동 데이터 수집 없음; 보유 자료와 기존 사용자 모델만 사용합니다.
# 기존 모델/원본은 읽기 전용. 새 출력: Allinone/All_Exercises_Prototype
import json as _catalog_json
CATALOG = _catalog_json.loads('[\n  {\n    "id": "treadmill_walk",\n    "equipment": "트레드밀",\n    "name": "걷기",\n    "aliases": [\n      "treadmill_walk",\n      "treadmill walk",\n      "treadmill walking",\n      "트레드밀 걷기"\n    ],\n    "legacy": null,\n    "related": "walking",\n    "requested": true\n  },\n  {\n    "id": "treadmill_run",\n    "equipment": "트레드밀",\n    "name": "달리기",\n    "aliases": [\n      "treadmill_run",\n      "treadmill run",\n      "treadmill running",\n      "트레드밀 달리기"\n    ],\n    "legacy": null,\n    "related": "running",\n    "requested": true\n  },\n  {\n    "id": "machine_chest_press",\n    "equipment": "시티드 체스트 프레스",\n    "name": "머신 체스트 프레스",\n    "aliases": [\n      "machine_chest_press",\n      "machine chest press",\n      "seated chest press",\n      "머신 체스트 프레스"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "one_arm_chest_press",\n    "equipment": "시티드 체스트 프레스",\n    "name": "원암 체스트 프레스",\n    "aliases": [\n      "one_arm_chest_press",\n      "one arm chest press",\n      "single arm chest press",\n      "원암 체스트 프레스"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "pec_deck_fly",\n    "equipment": "펙토럴 플라이",\n    "name": "펙덱 플라이",\n    "aliases": [\n      "pec_deck_fly",\n      "pec deck fly",\n      "pec deck",\n      "pectoral fly",\n      "chest fly machine",\n      "펙덱 플라이"\n    ],\n    "legacy": "chest_fly_machine",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "reverse_pec_deck",\n    "equipment": "펙토럴 플라이",\n    "name": "리버스 펙덱 플라이",\n    "aliases": [\n      "reverse_pec_deck",\n      "reverse pec deck",\n      "reverse pec deck fly",\n      "리버스 펙덱"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "overhand_lat_pulldown",\n    "equipment": "랫풀다운",\n    "name": "오버핸드 그립",\n    "aliases": [\n      "overhand_lat_pulldown",\n      "overhand lat pulldown",\n      "overhand grip lat pulldown",\n      "랫풀다운 오버핸드"\n    ],\n    "legacy": null,\n    "related": "lat_pulldown",\n    "requested": true\n  },\n  {\n    "id": "neutral_lat_pulldown",\n    "equipment": "랫풀다운",\n    "name": "뉴트럴 그립",\n    "aliases": [\n      "neutral_lat_pulldown",\n      "neutral lat pulldown",\n      "neutral grip lat pulldown",\n      "랫풀다운 뉴트럴"\n    ],\n    "legacy": null,\n    "related": "lat_pulldown",\n    "requested": true\n  },\n  {\n    "id": "machine_shoulder_press",\n    "equipment": "숄더 프레스",\n    "name": "머신 숄더 프레스",\n    "aliases": [\n      "machine_shoulder_press",\n      "machine shoulder press",\n      "머신 숄더 프레스"\n    ],\n    "legacy": null,\n    "related": "shoulder_press",\n    "requested": true\n  },\n  {\n    "id": "one_arm_shoulder_press",\n    "equipment": "숄더 프레스",\n    "name": "원암 숄더 프레스",\n    "aliases": [\n      "one_arm_shoulder_press",\n      "one arm shoulder press",\n      "single arm shoulder press",\n      "원암 숄더 프레스"\n    ],\n    "legacy": null,\n    "related": "shoulder_press",\n    "requested": true\n  },\n  {\n    "id": "seated_cable_row",\n    "equipment": "롱 풀",\n    "name": "시티드 케이블 로우",\n    "aliases": [\n      "seated_cable_row",\n      "seated cable row",\n      "long pull",\n      "시티드 케이블 로우"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "one_arm_cable_row",\n    "equipment": "롱 풀",\n    "name": "원암 케이블 로우",\n    "aliases": [\n      "one_arm_cable_row",\n      "one arm cable row",\n      "single arm cable row",\n      "원암 케이블 로우"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "machine_biceps_curl",\n    "equipment": "암 컬",\n    "name": "머신 바이셉 컬",\n    "aliases": [\n      "machine_biceps_curl",\n      "machine biceps curl",\n      "machine bicep curl",\n      "machine arm curl",\n      "머신 바이셉 컬"\n    ],\n    "legacy": null,\n    "related": "bicep_curl",\n    "requested": true\n  },\n  {\n    "id": "one_arm_curl",\n    "equipment": "암 컬",\n    "name": "원암 컬",\n    "aliases": [\n      "one_arm_curl",\n      "one arm curl",\n      "single arm curl",\n      "원암 컬"\n    ],\n    "legacy": null,\n    "related": "bicep_curl",\n    "requested": true\n  },\n  {\n    "id": "pull_up",\n    "equipment": "친업",\n    "name": "풀업",\n    "aliases": [\n      "pull_up",\n      "pull up",\n      "pullup",\n      "풀업"\n    ],\n    "legacy": "pull_up",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "assisted_pull_up",\n    "equipment": "친업",\n    "name": "어시스트 풀업",\n    "aliases": [\n      "assisted_pull_up",\n      "assisted pull up",\n      "assisted pullup",\n      "어시스트 풀업"\n    ],\n    "legacy": null,\n    "related": "pull_up",\n    "requested": true\n  },\n  {\n    "id": "leg_extension",\n    "equipment": "레그 익스텐션",\n    "name": "머신 레그 익스텐션",\n    "aliases": [\n      "leg_extension",\n      "machine leg extension",\n      "leg extension",\n      "레그 익스텐션"\n    ],\n    "legacy": "leg_extension",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "single_leg_extension",\n    "equipment": "레그 익스텐션",\n    "name": "싱글 레그 익스텐션",\n    "aliases": [\n      "single_leg_extension",\n      "single leg extension",\n      "one leg extension",\n      "싱글 레그 익스텐션"\n    ],\n    "legacy": null,\n    "related": "leg_extension",\n    "requested": true\n  },\n  {\n    "id": "seated_leg_curl",\n    "equipment": "레그 컬",\n    "name": "시티드 레그 컬",\n    "aliases": [\n      "seated_leg_curl",\n      "seated leg curl",\n      "시티드 레그 컬"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "single_leg_curl",\n    "equipment": "레그 컬",\n    "name": "싱글 레그 컬",\n    "aliases": [\n      "single_leg_curl",\n      "single leg curl",\n      "one leg curl",\n      "싱글 레그 컬"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "seated_leg_press",\n    "equipment": "시티드 레그 프레스",\n    "name": "시티드 레그 프레스",\n    "aliases": [\n      "seated_leg_press",\n      "seated leg press",\n      "시티드 레그 프레스"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "single_seated_leg_press",\n    "equipment": "시티드 레그 프레스",\n    "name": "싱글 레그 프레스",\n    "aliases": [\n      "single_seated_leg_press",\n      "single seated leg press",\n      "single leg seated leg press",\n      "시티드 싱글 레그 프레스"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "cable_chest_fly",\n    "equipment": "케이블 크로스오버",\n    "name": "케이블 체스트 플라이",\n    "aliases": [\n      "cable_chest_fly",\n      "cable chest fly",\n      "cable fly",\n      "케이블 체스트 플라이"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "tricep_pushdown",\n    "equipment": "케이블 크로스오버",\n    "name": "트라이셉 푸시다운",\n    "aliases": [\n      "tricep_pushdown",\n      "tricep pushdown",\n      "triceps pushdown",\n      "트라이셉 푸시다운"\n    ],\n    "legacy": "tricep_pushdown",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "cable_lateral_raise",\n    "equipment": "케이블 크로스오버",\n    "name": "케이블 레터럴 레이즈",\n    "aliases": [\n      "cable_lateral_raise",\n      "cable lateral raise",\n      "케이블 레터럴 레이즈"\n    ],\n    "legacy": null,\n    "related": "lateral_raise",\n    "requested": true\n  },\n  {\n    "id": "smith_squat",\n    "equipment": "스미스 머신",\n    "name": "스미스 스쿼트",\n    "aliases": [\n      "smith_squat",\n      "smith squat",\n      "smith machine squat",\n      "스미스 스쿼트"\n    ],\n    "legacy": null,\n    "related": "squat",\n    "requested": true\n  },\n  {\n    "id": "smith_shoulder_press",\n    "equipment": "스미스 머신",\n    "name": "스미스 숄더 프레스",\n    "aliases": [\n      "smith_shoulder_press",\n      "smith shoulder press",\n      "smith machine shoulder press",\n      "스미스 숄더 프레스"\n    ],\n    "legacy": null,\n    "related": "shoulder_press",\n    "requested": true\n  },\n  {\n    "id": "smith_bench_press",\n    "equipment": "스미스 머신",\n    "name": "스미스 벤치프레스",\n    "aliases": [\n      "smith_bench_press",\n      "smith bench press",\n      "smith machine bench press",\n      "스미스 벤치프레스"\n    ],\n    "legacy": null,\n    "related": "bench_press",\n    "requested": true\n  },\n  {\n    "id": "dips",\n    "equipment": "치닝 디핑 힙 플렉서",\n    "name": "딥스",\n    "aliases": [\n      "dips",\n      "tricep dips",\n      "triceps dips",\n      "딥스"\n    ],\n    "legacy": "tricep_dips",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "hanging_leg_raise",\n    "equipment": "치닝 디핑 힙 플렉서",\n    "name": "행잉 레그 레이즈",\n    "aliases": [\n      "hanging_leg_raise",\n      "hanging leg raise",\n      "hanging leg raises",\n      "행잉 레그 레이즈"\n    ],\n    "legacy": null,\n    "related": "leg_raises",\n    "requested": true\n  },\n  {\n    "id": "power_leg_press",\n    "equipment": "파워 레그 프레스",\n    "name": "파워 레그 프레스",\n    "aliases": [\n      "power_leg_press",\n      "power leg press",\n      "45 degree leg press",\n      "파워 레그 프레스"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "single_power_leg_press",\n    "equipment": "파워 레그 프레스",\n    "name": "싱글 레그 프레스",\n    "aliases": [\n      "single_power_leg_press",\n      "single power leg press",\n      "single leg 45 degree leg press",\n      "파워 싱글 레그 프레스"\n    ],\n    "legacy": null,\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "barbell_back_squat",\n    "equipment": "파워 랙",\n    "name": "바벨 백 스쿼트",\n    "aliases": [\n      "barbell_back_squat",\n      "barbell back squat",\n      "barbell squat",\n      "바벨 백 스쿼트"\n    ],\n    "legacy": null,\n    "related": "squat",\n    "requested": true\n  },\n  {\n    "id": "deadlift",\n    "equipment": "파워 랙",\n    "name": "바벨 데드리프트",\n    "aliases": [\n      "deadlift",\n      "barbell deadlift",\n      "바벨 데드리프트"\n    ],\n    "legacy": "deadlift",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "barbell_lunge",\n    "equipment": "파워 랙",\n    "name": "바벨 런지",\n    "aliases": [\n      "barbell_lunge",\n      "barbell lunge",\n      "barbell lunges",\n      "바벨 런지"\n    ],\n    "legacy": null,\n    "related": "lunge",\n    "requested": true\n  },\n  {\n    "id": "bench_press",\n    "equipment": "벤치프레스",\n    "name": "벤치프레스",\n    "aliases": [\n      "bench_press",\n      "flat bench press",\n      "bench press",\n      "벤치프레스"\n    ],\n    "legacy": "bench_press",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "incline_bench_press",\n    "equipment": "벤치프레스",\n    "name": "인클라인 벤치프레스",\n    "aliases": [\n      "incline_bench_press",\n      "incline bench press",\n      "인클라인 벤치프레스"\n    ],\n    "legacy": "incline_bench_press",\n    "related": null,\n    "requested": true\n  },\n  {\n    "id": "barbell_biceps_curl",\n    "equipment": "기존·추가 운동",\n    "name": "바벨 바이셉 컬",\n    "aliases": [\n      "barbell_biceps_curl"\n    ],\n    "legacy": "barbell_biceps_curl",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "decline_bench_press",\n    "equipment": "기존·추가 운동",\n    "name": "디클라인 벤치프레스",\n    "aliases": [\n      "decline_bench_press"\n    ],\n    "legacy": "decline_bench_press",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "hammer_curl",\n    "equipment": "기존·추가 운동",\n    "name": "해머 컬",\n    "aliases": [\n      "hammer_curl"\n    ],\n    "legacy": "hammer_curl",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "hip_thrust",\n    "equipment": "기존·추가 운동",\n    "name": "힙 쓰러스트",\n    "aliases": [\n      "hip_thrust"\n    ],\n    "legacy": "hip_thrust",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "lat_pulldown",\n    "equipment": "기존·추가 운동",\n    "name": "랫풀다운 — 그립 미구분",\n    "aliases": [\n      "lat_pulldown"\n    ],\n    "legacy": "lat_pulldown",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "lateral_raise",\n    "equipment": "기존·추가 운동",\n    "name": "레터럴 레이즈 — 기구 미구분",\n    "aliases": [\n      "lateral_raise"\n    ],\n    "legacy": "lateral_raise",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "leg_raises",\n    "equipment": "기존·추가 운동",\n    "name": "레그 레이즈 — 변형 미구분",\n    "aliases": [\n      "leg_raises",\n      "leg raises"\n    ],\n    "legacy": "leg_raises",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "plank",\n    "equipment": "기존·추가 운동",\n    "name": "플랭크",\n    "aliases": [\n      "plank"\n    ],\n    "legacy": "plank",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "push_up",\n    "equipment": "기존·추가 운동",\n    "name": "푸시업",\n    "aliases": [\n      "push_up",\n      "push-up",\n      "pushup"\n    ],\n    "legacy": "push_up",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "romanian_deadlift",\n    "equipment": "기존·추가 운동",\n    "name": "루마니안 데드리프트",\n    "aliases": [\n      "romanian_deadlift"\n    ],\n    "legacy": "romanian_deadlift",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "russian_twist",\n    "equipment": "기존·추가 운동",\n    "name": "러시안 트위스트",\n    "aliases": [\n      "russian_twist"\n    ],\n    "legacy": "russian_twist",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "shoulder_press",\n    "equipment": "기존·추가 운동",\n    "name": "숄더 프레스 — 기구 미구분",\n    "aliases": [\n      "shoulder_press"\n    ],\n    "legacy": "shoulder_press",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "squat",\n    "equipment": "기존·추가 운동",\n    "name": "스쿼트 — 기구 미구분",\n    "aliases": [\n      "squat"\n    ],\n    "legacy": "squat",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "t_bar_row",\n    "equipment": "기존·추가 운동",\n    "name": "T바 로우",\n    "aliases": [\n      "t_bar_row"\n    ],\n    "legacy": "t_bar_row",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "bent_over_row",\n    "equipment": "기존·추가 운동",\n    "name": "벤트오버 로우",\n    "aliases": [\n      "bent_over_row"\n    ],\n    "legacy": "bent_over_row",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "bicep_curl",\n    "equipment": "기존·추가 운동",\n    "name": "바이셉 컬 — 기구 미구분",\n    "aliases": [\n      "bicep_curl",\n      "biceps curl",\n      "bicep curl"\n    ],\n    "legacy": "bicep_curl",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "side_bend",\n    "equipment": "기존·추가 운동",\n    "name": "사이드 벤드",\n    "aliases": [\n      "side_bend"\n    ],\n    "legacy": "side_bend",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "triceps_extension",\n    "equipment": "기존·추가 운동",\n    "name": "삼두 익스텐션",\n    "aliases": [\n      "triceps_extension"\n    ],\n    "legacy": "triceps_extension",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "walking",\n    "equipment": "기존·추가 운동",\n    "name": "걷기 — 기구 미구분",\n    "aliases": [\n      "walking"\n    ],\n    "legacy": "walking",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "running",\n    "equipment": "기존·추가 운동",\n    "name": "달리기 — 기구 미구분",\n    "aliases": [\n      "running"\n    ],\n    "legacy": "running",\n    "related": null,\n    "requested": false\n  },\n  {\n    "id": "lunge",\n    "equipment": "기존·추가 운동",\n    "name": "런지 — 기구 미구분",\n    "aliases": [\n      "lunge"\n    ],\n    "legacy": "lunge",\n    "related": null,\n    "requested": false\n  }\n]')
WORKER_CODE = '"""Batch training and prediction worker. Only load the user\'s existing trusted models."""\nimport os\nos.environ[\'MPLBACKEND\']=\'Agg\'\nimport sys, json, re, hashlib, warnings, importlib.util, math, uuid, traceback\nfrom pathlib import Path\nfrom collections import Counter\nimport numpy as np\nimport pandas as pd\nimport joblib\nimport sklearn\nfrom sklearn.exceptions import InconsistentVersionWarning\nfrom sklearn.ensemble import RandomForestClassifier, IsolationForest\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import GroupShuffleSplit\nfrom sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix\n\nIMG={\'.jpg\',\'.jpeg\',\'.png\',\'.webp\',\'.bmp\',\'.tif\',\'.tiff\'}\nVIDEO={\'.mp4\',\'.mov\',\'.avi\',\'.mkv\',\'.webm\',\'.m4v\'}\nMEDIA=IMG|VIDEO|{\'.gif\'}\nVERSION=\'all_exercises_pose58_stats6_v1\'\nTRIPLES=[(11,13,15),(12,14,16),(13,11,23),(14,12,24),(11,23,25),(12,24,26),(23,25,27),(24,26,28),(25,27,31),(26,28,32)]\nREQUIRED=set(j for t in TRIPLES for j in t)\nSELECTED=[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]\nNEG={\'bad\',\'wrong\',\'incorrect\',\'improper\',\'faulty\',\'poor\',\'unsafe\',\'invalid\',\'abnormal\',\'error\',\'mistake\',\'나쁜\',\'잘못된\',\'틀린\',\'오류\',\'불량\'}\nPOS={\'good\',\'correct\',\'proper\',\'normal\',\'정상\',\'올바른\'}\n\n\ndef write_json(path, obj):\n    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)\n    tmp=path.with_suffix(path.suffix+\'.tmp\')\n    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding=\'utf-8\')\n    tmp.replace(path)\n\n\ndef norm(x):\n    return re.sub(r\'[^a-z0-9가-힣]+\',\' \',str(x).lower()).strip()\n\n\ndef kind(path):\n    suffix=Path(path).suffix.lower()\n    return \'IMAGE\' if suffix in IMG else \'GIF\' if suffix==\'.gif\' else \'VIDEO\'\n\n\ndef sha(path):\n    h=hashlib.sha256()\n    with open(path,\'rb\') as f:\n        for b in iter(lambda:f.read(1024*1024),b\'\'):h.update(b)\n    return h.hexdigest()\n\n\ndef detect_exercise(relative, catalog):\n    text=\' \'+norm(str(relative))+\' \'\n    # A one-leg press needs machine context, not a silent seated/45-degree guess.\n    if \' single leg press \' in text or \' 싱글 레그 프레스 \' in text:\n        seated=(\'seated\' in text or \'시티드\' in text)\n        power=(\'power\' in text or \'45 degree\' in text or \'파워\' in text)\n        if seated != power:return \'single_seated_leg_press\' if seated else \'single_power_leg_press\'\n        return None\n    matches=[]\n    for c in catalog:\n        for alias in c[\'aliases\']:\n            a=norm(alias)\n            if \' \'+a+\' \' in text:matches.append((len(a),c[\'id\']))\n    if not matches:return None\n    maximum=max(m[0] for m in matches)\n    ids={i for n,i in matches if n==maximum}\n    return next(iter(ids)) if len(ids)==1 else None\n\n\ndef label_info(relative):\n    parts=Path(relative).parts\n    # Ignore neutral dataset/container names such as GOOD_BAD; inspect nearest labels.\n    for part in reversed(parts):\n        words=set(norm(Path(part).stem).split())\n        neg=words&NEG;pos=words&POS\n        if neg and pos:\n            if norm(part).replace(\' \',\'\') in {\'goodbad\',\'badgood\',\'goodbaddataset\'}:continue\n            return None,\'CONFLICTING_LABELS\'\n        if neg:return \'BAD\',\'explicit_path\'\n        if pos:return \'GOOD\',\'explicit_path\'\n    return \'GOOD\',\'assumed_good_unlabeled\'\n\n\ndef trusted_load(path):\n    # These paths are restricted by the launcher to the user\'s own artifacts.\n    with warnings.catch_warnings(record=True) as caught:\n        warnings.simplefilter(\'always\',InconsistentVersionWarning)\n        obj=joblib.load(path)\n    versions=sorted({str(w.message.original_sklearn_version) for w in caught\n                     if isinstance(w.message,InconsistentVersionWarning)})\n    if versions:\n        raise RuntimeError(\'SKLEARN_VERSION_REQUIRED:\'+\',\'.join(versions))\n    return obj\n\n\ndef load_module(path, name):\n    spec=importlib.util.spec_from_file_location(name,str(path))\n    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n    return mod\n\n\ndef frame_feature(lm):\n    p=np.array([[v.x,v.y,v.z] for v in lm],dtype=np.float32)\n    vals=[]\n    for a,b,c in TRIPLES:\n        u,v=p[a]-p[b],p[c]-p[b];d=np.linalg.norm(u)*np.linalg.norm(v)\n        vals.append(0. if d<1e-8 else float(np.arccos(np.clip(np.dot(u,v)/d,-1,1))/np.pi))\n    angles=np.array(vals,dtype=np.float32)\n    hip=(p[23]+p[24])/2;shoulder=(p[11]+p[12])/2\n    scale=max(np.linalg.norm(shoulder-hip),0.05)\n    return np.concatenate([angles,((p[SELECTED]-hip)/scale).flatten()])\n\n\ndef extract_media(path):\n    import cv2, mediapipe as mp\n    from PIL import Image, ImageOps\n    path=Path(path);k=kind(path);rows=[];sampled=0\n    if not hasattr(mp,\'solutions\'):raise RuntimeError(\'MediaPipe solutions API missing\')\n    def frames():\n        if k==\'IMAGE\':\n            with Image.open(path) as im:yield np.asarray(ImageOps.exif_transpose(im).convert(\'RGB\'))\n        elif k==\'GIF\':\n            with Image.open(path) as im:\n                n=im.n_frames\n                for i in np.unique(np.linspace(0,n-1,min(32,n)).astype(int)):\n                    im.seek(int(i));yield np.asarray(im.convert(\'RGB\'))\n        else:\n            cap=cv2.VideoCapture(str(path))\n            try:\n                if not cap.isOpened():raise ValueError(\'VIDEO_OPEN_FAILED\')\n                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n                if n<=0:raise ValueError(\'FRAME_COUNT_UNAVAILABLE\')\n                for i in np.unique(np.linspace(0,n-1,min(32,n)).astype(int)):\n                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,bgr=cap.read()\n                    if ok:yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)\n            finally:cap.release()\n    with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=1,min_detection_confidence=.5) as pose:\n        for rgb in frames():\n            sampled+=1;h,w=rgb.shape[:2]\n            if max(h,w)>960:\n                r=960/max(h,w);rgb=cv2.resize(rgb,(max(1,int(w*r)),max(1,int(h*r))))\n            res=pose.process(np.ascontiguousarray(rgb))\n            if not res.pose_landmarks:continue\n            lm=res.pose_landmarks.landmark\n            if np.mean([lm[j].visibility for j in REQUIRED])<.35:continue\n            row=frame_feature(lm)\n            if np.isfinite(row).all():rows.append(row)\n    minimum=1 if k==\'IMAGE\' else 5\n    if len(rows)<minimum:raise ValueError(f\'POSE_TOO_FEW: {len(rows)}/{sampled}, required={minimum}\')\n    a=np.stack(rows)\n    vector=a[0] if k==\'IMAGE\' else np.concatenate([\n        a.mean(0),a.std(0),np.percentile(a,10,axis=0),np.percentile(a,50,axis=0),\n        np.percentile(a,90,axis=0),np.abs(np.diff(a,axis=0)).mean(0)]).astype(np.float32)\n    return vector,dict(sampled_frames=sampled,valid_frames=len(rows),feature_count=len(vector),media_type=k)\n\n\ndef extract_original_squat_image(path):\n    """Preserve the saved squat image model\'s original PIL thumbnail preprocessing."""\n    from PIL import Image, ImageOps\n    import mediapipe as mp\n    with Image.open(path) as im:\n        im=ImageOps.exif_transpose(im).convert(\'RGB\');im.thumbnail((960,960))\n        rgb=np.asarray(im)\n    with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=1,min_detection_confidence=.5) as pose:\n        r=pose.process(np.ascontiguousarray(rgb))\n    if not r.pose_landmarks:raise ValueError(\'NO_POSE\')\n    x=frame_feature(r.pose_landmarks.landmark)\n    if not np.isfinite(x).all():raise ValueError(\'INVALID_FEATURE\')\n    return x,{\'sampled_frames\':1,\'valid_frames\':1,\'feature_count\':58,\'media_type\':\'IMAGE\'}\n\n\ndef inspect(cfg):\n    out=[]\n    for spec in cfg[\'candidates\']:\n        r=dict(spec)\n        try:\n            b=trusted_load(spec[\'model_path\'])\n            if not isinstance(b,dict):raise ValueError(\'Unknown model schema\')\n            if spec[\'backend\']==\'expanded\':\n                r[\'exercises\']=sorted(set(b.get(\'quality\',{}))|set(b.get(\'quality_image\',{}))|set(b.get(\'error_frame\',{})))\n                r[\'feature_version\']=b.get(\'feature_version\')\n            else:\n                if not hasattr(b.get(\'model\'),\'predict\'):raise ValueError(\'No predictor\')\n                r[\'exercise_in_bundle\']=b.get(\'exercise\')\n                r[\'mode\']=b.get(\'mode\')\n                r[\'feature_count\']=int(getattr(b[\'model\'],\'n_features_in_\',0))\n                if spec[\'backend\']==\'legacy\' and r[\'feature_count\']!=348:\n                    raise ValueError(\'Legacy feature contract mismatch, expected 348\')\n                if spec[\'backend\']==\'squat_image\' and r[\'feature_count\']!=58:\n                    raise ValueError(\'Image contract mismatch, expected 58\')\n                if spec[\'backend\']==\'new\' and b.get(\'feature_version\')!=VERSION:\n                    raise ValueError(\'New feature version mismatch\')\n            r.update(status=\'LOAD_CHECKED\',sklearn=sklearn.__version__,model_sha256=sha(spec[\'model_path\']))\n        except Exception as e:r.update(status=\'NOT_LOADED\',error=str(e))\n        out.append(r)\n    return {\'models\':out}\n\n\ndef group_holdout(x,y,groups):\n    if len(set(groups))<4:return None\n    for seed in range(42,62):\n        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(x,y,groups))\n        if len(set(y[a]))==2 and len(set(y[b]))==2:return a,b\n    return None\n\n\ndef train(cfg):\n    root=Path(cfg[\'root\']);run=Path(cfg[\'run\']);cache_dir=Path(cfg[\'cache\']);cache_dir.mkdir(parents=True,exist_ok=True)\n    grouped={};failures=[];rows=[];seen={};conflict=set();reused=0\n    # Bind labels and hashes before feature extraction, so conflicting duplicates cannot train.\n    for source in cfg[\'files\']:\n        r=dict(source)\n        try:\n            h=sha(r[\'path\']);r[\'digest\']=h\n            if h in seen:\n                p=seen[h]\n                if (p[\'exercise\'],p[\'label\'])!=(r[\'exercise\'],r[\'label\']):conflict.add(h)\n                else:failures.append({\'file\':r[\'path\'],\'reason\':\'BYTE_DUPLICATE\'})\n                continue\n            seen[h]=r\n        except Exception as e:failures.append({\'file\':r[\'path\'],\'reason\':str(e)})\n    unique=[r for h,r in seen.items() if h not in conflict]\n    for h in conflict:failures.append({\'digest\':h,\'reason\':\'CONFLICTING_DUPLICATE_LABELS_OR_EXERCISES\'})\n    # Check shared environment once, not hundreds of silent per-file import failures.\n    if unique:\n        import mediapipe as mp\n        if not hasattr(mp,\'solutions\'):raise RuntimeError(\'MediaPipe 0.10.21 is required\')\n        with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=1):pass\n    for i,r in enumerate(unique,1):\n        try:\n            cf=cache_dir/(r[\'digest\']+\'_\'+r[\'media_type\']+\'.npz\')\n            if cf.is_file():\n                with np.load(cf,allow_pickle=False) as z:\n                    if str(z[\'version\'].item())!=VERSION:raise ValueError(\'Cache version mismatch\')\n                    x=z[\'x\'];info=json.loads(str(z[\'info\'].item()))\n                reused+=1\n            else:\n                x,info=extract_media(r[\'path\'])\n                tmp=cf.with_suffix(\'.npz.tmp\')\n                with open(tmp,\'wb\') as f:np.savez_compressed(f,version=VERSION,x=x,info=json.dumps(info))\n                tmp.replace(cf)\n            row={**r,**info};rows.append(row)\n            grouped.setdefault((r[\'exercise\'],r[\'media_type\']),[]).append((row,x))\n        except Exception as e:failures.append({\'file\':r[\'path\'],\'reason\':str(e)})\n        if i==1 or i%10==0 or i==len(unique):\n            print(f\'신규 자료 관절 추출 {i}/{len(unique)} | 유효 {len(rows)} | 캐시 {reused}\',flush=True)\n    results=[];new_specs=[]\n    for (ex,k),items in grouped.items():\n        x=np.stack([p[1] for p in items]);meta=[p[0] for p in items]\n        y=np.array([r[\'label\'] for r in meta]);g=np.array([r[\'group\'] if r.get(\'group\') else \'file:\'+r[\'digest\'] for r in meta])\n        counts=Counter(y);ng=counts[\'GOOD\'];nb=counts[\'BAD\']\n        r={\'exercise\':ex,\'media_type\':k,\'GOOD\':ng,\'BAD\':nb,\'assumed_good\':sum(m[\'label_origin\']==\'assumed_good_unlabeled\' for m in meta)}\n        sig=hashlib.sha256(json.dumps(sorted((m[\'digest\'],m[\'label\'],str(m.get(\'group\',\'\'))) for m in meta)).encode()).hexdigest()\n        dest=run/f\'{ex}__{k}.joblib\'\n        try:\n            if ng>=5 and nb>=5:\n                split=group_holdout(x,y,g)\n                estimator=RandomForestClassifier(n_estimators=200,class_weight=\'balanced\',min_samples_leaf=2,random_state=42,n_jobs=2)\n                if split:\n                    tr,te=split;estimator.fit(x[tr],y[tr]);p=estimator.predict(x[te])\n                    evaluation={\'status\':\'SMALL_GROUP_HOLDOUT\',\'test_files\':len(te),\'balanced_accuracy\':float(balanced_accuracy_score(y[te],p)),\n                        \'confusion_matrix\':confusion_matrix(y[te],p,labels=[\'GOOD\',\'BAD\']).tolist(),\n                        \'split_level\':\'subject_when_available_else_file\',\'ground_truth_may_include_assumed_good\':r[\'assumed_good\']>0}\n                else:\n                    tr=np.arange(len(y));te=np.array([],int);estimator.fit(x,y)\n                    evaluation={\'status\':\'NOT_INDEPENDENTLY_EVALUATED\'}\n                b={\'mode\':\'BINARY\',\'model\':estimator,\'evaluation\':evaluation};r[\'mode\']=\'BINARY\'\n            elif ng>=8:\n                gx=x[y==\'GOOD\'];gg=g[y==\'GOOD\'];train_ids=np.arange(ng);cal_ids=train_ids\n                method=\'TRAINING_GOOD_PROVISIONAL\'\n                if ng>=16 and len(set(gg))>=4:\n                    train_ids,cal_ids=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=42).split(gx,groups=gg));method=\'HELDOUT_GOOD_CALIBRATION_ONLY\'\n                scale=StandardScaler().fit(gx[train_ids]);estimator=IsolationForest(n_estimators=200,random_state=42,contamination=\'auto\',n_jobs=2)\n                estimator.fit(scale.transform(gx[train_ids]));scores=-estimator.score_samples(scale.transform(gx[cal_ids]))\n                threshold=float(np.quantile(scores,.9))\n                b={\'mode\':\'ONE_CLASS\',\'model\':estimator,\'scaler\':scale,\'threshold\':threshold,\n                   \'evaluation\':{\'status\':method,\'calibration_files\':len(cal_ids),\'false_alarm_rate_not_guaranteed\':True,\'bad_detection_not_validated\':True}}\n                r[\'mode\']=\'ONE_CLASS\';te=np.array([],int)\n            else:\n                r.update(status=\'NEEDS_MORE_DATA\',mode=\'NONE\');results.append(r);continue\n            ref=x[y==\'GOOD\',:58]\n            b.update(exercise=ex,media_type=k,feature_version=VERSION,dataset_signature=sig,\n                label_policy=\'unlabeled_assumed_good; explicit negatives bad\',counts=dict(counts),\n                reference_median=np.median(ref,axis=0),reference_iqr=np.percentile(ref,75,axis=0)-np.percentile(ref,25,axis=0),\n                sklearn=sklearn.__version__,numpy=np.__version__,python=sys.version,feedback_scope=\'generic_classification_and_feature_differences_not_error_diagnosis\')\n            joblib.dump(b,dest,compress=3)\n            # Serialization test uses real feature rows; this is NOT accuracy validation.\n            loaded=trusted_load(dest)\n            a=loaded[\'scaler\'].transform(x[:1]) if b[\'mode\']==\'ONE_CLASS\' else x[:1]\n            p=loaded[\'model\'].score_samples(a) if b[\'mode\']==\'ONE_CLASS\' else loaded[\'model\'].predict(a)\n            r.update(status=\'TRAINED\',model_path=str(dest),evaluation=b[\'evaluation\'],serialization_test_passed=True)\n            new_specs.append({\'exercise\':ex,\'media_type\':k,\'backend\':\'new\',\'model_path\':str(dest),\n                \'sklearn\':sklearn.__version__,\'status\':\'LOAD_CHECKED\',\'model_sha256\':sha(dest),\'mode\':b[\'mode\']})\n            print(f\'추가 학습: {ex} [{k}] {b["mode"]} GOOD={ng}, BAD={nb}\',flush=True)\n        except Exception as e:r.update(status=\'FAILED\',error=str(e))\n        results.append(r)\n    pd.DataFrame(rows).to_csv(run/\'new_data_manifest.csv\',index=False,encoding=\'utf-8-sig\')\n    write_json(run/\'new_data_failures.json\',failures)\n    return {\'results\':results,\'models\':new_specs,\'candidate_files\':len(cfg[\'files\']),\'usable_files\':len(rows),\'failures\':len(failures)}\n\n\ndef check_dim(model,x):\n    if getattr(model,\'n_features_in_\',x.shape[1])!=x.shape[1]:raise ValueError(\'FEATURE_DIMENSION_MISMATCH\')\n\n\ndef predict(cfg):\n    spec=cfg[\'spec\'];path=Path(cfg[\'media_path\']);k=kind(path)\n    if k!=spec[\'media_type\']:raise ValueError(\'지원하지 않는 사진/영상/GIF 조합입니다.\')\n    if sha(spec[\'model_path\'])!=spec[\'model_sha256\']:raise ValueError(\'모델이 연결 이후 변경됐습니다. 재연결이 필요합니다.\')\n    if spec[\'backend\']==\'expanded\':\n        # Validate library version BEFORE the existing runtime loads and predicts.\n        b=trusted_load(spec[\'model_path\']);mod=load_module(cfg[\'expanded_runtime\'],\'expanded_runtime_trusted\')\n        folder=Path(spec[\'model_path\']).parent\n        asset=folder/\'pose_landmarker_full.task\'\n        if not asset.is_file():raise FileNotFoundError(\'기존 관절 모델 파일 누락: \'+str(asset))\n        result=mod.predict_video(folder,path,spec[\'exercise\'])\n        result.update(status=\'PREDICTED\' if result.get(\'quality\') not in {\'판단 보류\',None} else \'RECHECK\',backend=\'expanded\',model_path=spec[\'model_path\'])\n        return result\n    b=trusted_load(spec[\'model_path\'])\n    x,info=extract_original_squat_image(path) if spec[\'backend\']==\'squat_image\' else extract_media(path)\n    x=x.reshape(1,-1)\n    check_dim(b[\'model\'],x)\n    if b.get(\'mode\')==\'ONE_CLASS\':\n        score=float(-b[\'model\'].score_samples(b[\'scaler\'].transform(x))[0]);threshold=float(b[\'threshold\'])\n        pred=\'BAD_SUSPECTED\' if score>threshold else \'GOOD_RANGE\'\n        result={\'prediction\':pred,\'anomaly_score\':score,\'threshold\':threshold,\'score_note\':\'이상 점수는 오류 확률이 아닙니다.\'}\n    else:\n        p=str(b[\'model\'].predict(x)[0]);result={\'prediction\':p,\'score_note\':\'원시 모델 점수는 정확도나 안전 확률이 아닙니다.\'}\n        if hasattr(b[\'model\'],\'predict_proba\'):\n            result[\'raw_scores\']={str(c):float(v) for c,v in zip(b[\'model\'].classes_,b[\'model\'].predict_proba(x)[0])}\n    result.update(status=\'PREDICTED\',exercise=spec[\'exercise\'],media_type=k,pose=info,model_path=spec[\'model_path\'],backend=spec[\'backend\'],ground_truth=None)\n    if spec[\'backend\']==\'new\' and \'reference_median\' in b:\n        names=[\'왼쪽 팔꿈치\',\'오른쪽 팔꿈치\',\'왼쪽 어깨\',\'오른쪽 어깨\',\'왼쪽 고관절\',\'오른쪽 고관절\',\'왼쪽 무릎\',\'오른쪽 무릎\',\'왼쪽 발목\',\'오른쪽 발목\']\n        z=(x[0,:10]-b[\'reference_median\'][:10])/np.maximum(b[\'reference_iqr\'][:10],.05)\n        result[\'reference_differences\']=[{\'feature\':names[i], \'direction\':\'크게 관측\' if z[i]>0 else \'작게 관측\',\'relative_difference\':float(z[i]),\n            \'interpretation\':\'촬영 조건·체형·동작 단계 차이일 수 있으며 교정 지시나 오류 원인 판정이 아님\'} for i in np.argsort(np.abs(z))[-2:][::-1] if abs(z[i])>=2]\n    return result\n\n\ndef main():\n    cfg=json.loads(Path(sys.argv[1]).read_text(encoding=\'utf-8\'));out=Path(cfg[\'output\'])\n    try:\n        result={\'inspect\':inspect,\'train\':train,\'predict\':predict}[cfg[\'action\']](cfg)\n        write_json(out,result)\n    except BaseException as e:\n        write_json(out,{\'status\':\'FAILED\',\'error\':str(e),\'traceback\':traceback.format_exc()})\n        raise\nif __name__==\'__main__\':main()\n'
EXPANDED_SOURCE_SHA = '6b5b1308506dad4b4014093e783c7fbe9d711f479b07cbd6ebb07a364868e82b'
V6_TESTER_SHA = '1f4b8821028f4b54fe68dc4ee0720e52b1bcabb493f23475b9a30b9d4eb01650'
V6_MODEL_SHA = '4ce507fac6492424677de134a34c1d0e9815ece5ed92f7960a9f712de6e7935a'

# Existing models are read-only. New artifacts use a separate namespace.
import os, sys, re, json, subprocess, shutil, hashlib, zipfile, stat, uuid, traceback, runpy, html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter

ROOT=Path('/content/drive/MyDrive/Allinone')
CODE_VERSION='all_exercises_train_feedback_20261109_r1'
IMAGE_EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
VIDEO_EXT={'.mp4','.mov','.avi','.mkv','.webm','.m4v'}
MEDIA_EXT=IMAGE_EXT|VIDEO_EXT|{'.gif'}
# Only these raw-data directories are scanned, never test_input/model/output folders.
TRAIN_DIRS=['new_dataset','training_input']
MAX_LOCAL_EXTRACT_BYTES=3*1024**3
EXPANDED_REL='Other_Exercises_Expanded_20261027/run_20261027_142919/fitness_models.joblib'
RUNTIME_REL='Other_Exercises_Expanded_20261027/fitness_expanded_runtime.py'
LEGACY_REL='Hybrid_Training_Results_Py311/20261027_080653'
V6_REL='Deadlift_V6_Models/run_20261027_135306/deadlift_v6.joblib'
V6_TEST_REL='scripts/run_deadlift_v6_test_20261109.py'
SQUAT_REL='Squat_Image_Models/squat_image_good_bad.joblib'


def norm(x):return re.sub(r'[^a-z0-9가-힣]+',' ',str(x).lower()).strip()
def media_kind(p):return 'IMAGE' if Path(p).suffix.lower() in IMAGE_EXT else 'GIF' if Path(p).suffix.lower()=='.gif' else 'VIDEO'
def token():return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)

def write_csv(p,rows,fields):
    import csv
    with open(p,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


class AllExerciseProject:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.base=self.root/'All_Exercises_Prototype'
        self.work=Path('/content/allinone_all_exercises');self.work.mkdir(parents=True,exist_ok=True)
        self.run=self.base/'runs'/token();self.run.mkdir(parents=True,exist_ok=False)
        self.log=self.run/'execution.log';self.worker=self.work/'worker.py'
        self.worker.write_text(WORKER_CODE,encoding='utf-8')
        self.registry={};self.audit=[];self.scan_skips=[];self.env_ready={};self.expanded_runtime=self.root/RUNTIME_REL
        self.env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):self.env.pop(key,None)
        self.env.update(MPLBACKEND='Agg',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',UV_CACHE_DIR=str(self.work/'uv_cache'),UV_LINK_MODE='copy',UV_NO_PROGRESS='1')
        self.tools=self.work/'tools';self.tool_env=dict(self.env,PYTHONPATH=str(self.tools))
        self.tools.mkdir(exist_ok=True)
        if not (self.tools/'uv').is_dir():
            self.command([sys.executable,'-m','pip','install','-q','--target',str(self.tools),'uv'],'실행 환경 관리자 준비',self.tool_env)
        self.uv=[sys.executable,'-m','uv']
        # Compile-only safe helpers: no imports, model loads, or training top-level code.
        import ast
        tree=ast.parse(WORKER_CODE)
        names={'norm','detect_exercise','label_info'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        ns={'re':re,'Path':Path,'NEG':{'bad','wrong','incorrect','improper','faulty','poor','unsafe','invalid','abnormal','error','mistake','나쁜','잘못된','틀린','오류','불량'},'POS':{'good','correct','proper','normal','정상','올바른'}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<label_helpers>','exec'),ns)
        self.identify=ns['detect_exercise'];self.label=ns['label_info']

    def command(self,args,title,env=None,allow_fail=False):
        print('\n'+title,flush=True)
        with open(self.log,'a',encoding='utf-8') as f:
            f.write('\n'+title+'\n')
            p=subprocess.Popen([str(a) for a in args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1,env=env or self.env)
            try:
                for line in p.stdout:
                    print(line,end='',flush=True);f.write(line);f.flush()
                code=p.wait()
            except BaseException:
                p.terminate()
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                raise
        if code and not allow_fail:raise RuntimeError(title+' 실패. 로그: '+str(self.log))
        return code

    def ensure_env(self,sk='1.6.1'):
        if sk in self.env_ready:return self.env_ready[sk]
        if not re.fullmatch(r'\d+\.\d+\.\d+',sk):raise ValueError('지원하지 않는 모델 버전: '+sk)
        e=self.work/('py311_sk_'+sk.replace('.','_'));py=e/'bin/python'
        if not py.exists():self.command(self.uv+['venv','--python','3.11',e],f'Python 환경 준비 (모델 sklearn {sk})',self.tool_env)
        packages=['mediapipe==0.10.21','numpy==1.26.4','opencv-contrib-python==4.10.0.84',
            'protobuf==4.25.8','matplotlib==3.9.2','jax==0.4.35','jaxlib==0.4.35',
            'scipy==1.14.1','pandas==2.2.3','joblib==1.4.2','scikit-learn=='+sk,'Pillow','requests','tqdm']
        stamp=e/'package_contract.json'
        expected={'packages':packages}
        if not stamp.exists() or json.loads(stamp.read_text())!=expected:
            self.command(self.uv+['pip','install','--python',py,*packages],'패키지 준비: sklearn '+sk,self.tool_env)
            save_json(stamp,expected)
        self.env_ready[sk]=py;return py

    def worker_call(self,cfg,sk='1.6.1'):
        py=self.ensure_env(sk);key=uuid.uuid4().hex[:8]
        out=self.work/(key+'_result.json');request=self.work/(key+'_request.json')
        full={**cfg,'output':str(out)};save_json(request,full)
        code=self.command([py,'-u',self.worker,request],cfg.get('title',cfg['action']),allow_fail=True)
        result=json.loads(out.read_text()) if out.is_file() else {'status':'FAILED','error':'worker result missing','returncode':code}
        if code and result.get('status')!='FAILED':result.update(status='FAILED',returncode=code)
        return result

    def register(self,spec):
        key=(spec['exercise'],spec['media_type'])
        # Explicit-label classifiers outrank weakly labelled anomaly models for the same exact exercise.
        score={'new':35 if spec.get('mode')=='BINARY' else 10,'legacy':5,'expanded':40,'v6':50,'squat_image':40}
        if key not in self.registry or score.get(spec['backend'],0)>score.get(self.registry[key]['backend'],0):self.registry[key]=spec

    def connect_existing(self):
        print('\n[1/3] 기존 모델 연결 — 재학습하지 않습니다.',flush=True)
        specs=[];legacy=self.root/LEGACY_REL
        for item in CATALOG:
            if item.get('legacy'):
                path=legacy/(item['legacy']+'.joblib')
                if path.is_file():specs.append(dict(exercise=item['id'],media_type='VIDEO',model_path=str(path),backend='legacy'))
        exp=self.root/EXPANDED_REL
        if exp.is_file() and self.expanded_runtime.is_file():
            if digest(self.expanded_runtime)==EXPANDED_SOURCE_SHA:
                specs.append(dict(exercise='__expanded__',media_type='VIDEO',model_path=str(exp),backend='expanded'))
            else:self.audit.append({'backend':'expanded','status':'SOURCE_CHANGED','error':'기존 추론 코드가 확인 당시와 다릅니다. 자동 실행하지 않았습니다.'})
        sq=self.root/SQUAT_REL
        if sq.is_file():specs.append(dict(exercise='squat',media_type='IMAGE',model_path=str(sq),backend='squat_image'))
        # Previously trained new models: read only this program's own registry.
        latest=self.base/'latest_registry.json'
        if latest.is_file():
            for s in json.loads(latest.read_text()).get('models',[]):
                if s.get('backend')=='new' and Path(s['model_path']).is_file() and Path(s['model_path']).resolve().is_relative_to(self.base.resolve()):specs.append(s)
        pending=specs;sk='1.6.1';groups={sk:pending}
        inspected=[]
        for loop in range(3):
            next_groups={}
            for ver,cands in groups.items():
                try:r=self.worker_call({'action':'inspect','candidates':cands,'title':f'모델 {len(cands)}개 로드 검사 (sklearn {ver})'},ver)
                except Exception as e:
                    self.audit.append({'status':'ENVIRONMENT_FAILED','sklearn':ver,'error':str(e)});continue
                for s in r.get('models',[]):
                    err=s.get('error','')
                    m=re.search(r'SKLEARN_VERSION_REQUIRED:(\d+\.\d+\.\d+)$',err)
                    if m and m.group(1)!=ver and loop<2:next_groups.setdefault(m.group(1),[]).append(s);continue
                    inspected.append(s)
            groups=next_groups
            if not groups:break
        for s in inspected:
            self.audit.append(s)
            if s.get('status')!='LOAD_CHECKED':continue
            if s['backend']=='expanded':
                for ex in s['exercises']:
                    if any(c['id']==ex for c in CATALOG):self.register({**s,'exercise':ex,'mode':'BINARY_OR_LIMITED_ERROR'})
            else:self.register(s)
        v6=self.root/V6_REL;tester=self.root/V6_TEST_REL
        if v6.is_file() and tester.is_file() and (v6.parent/'environment.json').is_file():
            if digest(v6)==V6_MODEL_SHA and digest(tester)==V6_TESTER_SHA:
                self.register(dict(exercise='deadlift',media_type='VIDEO',backend='v6',mode='BINARY',
                    model_path=str(v6),model_sha256=V6_MODEL_SHA,status='PREVIOUS_EXTERNAL_TEST_CONFIRMED'))
            else:self.audit.append({'backend':'v6','status':'HASH_CHANGED','error':'V6 또는 실행 코드가 변경됨'})
        save_json(self.run/'model_audit.json',self.audit)
        print('연결한 운동/매체 조합:',len(self.registry),flush=True)

    def record_for(self,path,relative,source):
        ex=self.identify(relative,CATALOG)
        if ex is None:return None,'UNRECOGNIZED_OR_AMBIGUOUS_EXERCISE'
        k=media_kind(path)
        if (ex,k) in self.registry:return None,'EXISTING_MODEL_REUSED'
        parts=Path(relative).parts
        if any(norm(p) in {'test','tests','validation','val','test input'} for p in parts[:-1]):return None,'RESERVED_TEST_OR_VALIDATION'
        label,origin=self.label(relative)
        if label is None:return None,origin
        subject=re.search(r'\b(?:subject|participant|person|sub|s)\s*0*(\d+)\b',norm(str(relative)))
        group=source+':person:'+str(int(subject.group(1))) if subject else None
        return dict(path=str(path),relative=str(relative),exercise=ex,media_type=k,label=label,label_origin=origin,group=group,source=source),None

    def scan_train(self):
        print('\n[2/3] 보유 자료 일괄 학습 — 없는 모델만 처리합니다.',flush=True)
        files=[];counts=Counter();local=self.work/'raw_selected';local.mkdir(exist_ok=True)
        extracted_bytes=0
        for dirname in TRAIN_DIRS:
            base=self.root/dirname
            if not base.exists():continue
            for p in sorted(base.rglob('*')):
                if not p.is_file():continue
                relative=p.relative_to(base);source=dirname+':'+relative.parts[0]
                if p.suffix.lower() in MEDIA_EXT:
                    r,why=self.record_for(p,relative,source)
                    if r:files.append(r)
                    else:counts[why]+=1
                elif p.suffix.lower()=='.zip':
                    # Do not expand entire datasets into scarce Drive space.
                    try:
                        with zipfile.ZipFile(p) as z:
                            for m in z.infolist():
                                if m.is_dir():continue
                                ext=Path(m.filename).suffix.lower()
                                if ext=='.zip':
                                    self.scan_skips.append({'file':str(p),'member':m.filename,'reason':'NESTED_ZIP_REQUIRES_SOURCE_MAPPING'});continue
                                if ext not in MEDIA_EXT:continue
                                rel=relative.parent/Path(p.stem)/m.filename
                                r,why=self.record_for(Path(m.filename),rel,source)
                                if not r:counts[why]+=1;continue
                                if stat.S_ISLNK(m.external_attr>>16):counts['ZIP_SYMLINK_BLOCKED']+=1;continue
                                archive_dir=local/(hashlib.sha256(str(p).encode()).hexdigest()[:16])
                                dest=(archive_dir/m.filename).resolve()
                                if not dest.is_relative_to(archive_dir.resolve()):counts['UNSAFE_ZIP_PATH']+=1;continue
                                if extracted_bytes+m.file_size>MAX_LOCAL_EXTRACT_BYTES:
                                    counts['LOCAL_EXTRACTION_BUDGET_EXCEEDED']+=1;continue
                                extracted_bytes+=m.file_size
                                dest.parent.mkdir(parents=True,exist_ok=True)
                                # Always read verified ZIP member, not stale same-sized data.
                                with z.open(m) as src,open(dest,'wb') as dst:shutil.copyfileobj(src,dst)
                                r['path']=str(dest);files.append(r)
                    except Exception as e:self.scan_skips.append({'file':str(p),'reason':str(e)})
        save_json(self.run/'scan_notes.json',{'skipped_counts':dict(counts),'details':self.scan_skips,'internet_dataset_downloads':False})
        print('새 학습 후보:',len(files),'| 기존 모델로 건너뜀:',counts['EXISTING_MODEL_REUSED'],flush=True)
        # No download promises: material that is not present is not fabricated or relabelled as another exercise.
        if files:
            result=self.worker_call({'action':'train','files':files,'root':str(self.root),'run':str(self.run),
                'cache':str(self.base/'feature_cache'), 'title':'사진·영상·GIF 운동별 일괄 학습'},'1.6.1')
            if result.get('status')=='FAILED':raise RuntimeError(result.get('error','추가 학습 실패'))
        else:result={'candidate_files':0,'usable_files':0,'results':[],'models':[],'failures':0}
        for s in result.get('models',[]):self.register(s)
        save_json(self.run/'batch_training.json',result)
        return result

    def coverage(self,training):
        rows=[]
        for c in CATALOG:
            if not c['requested']:continue
            states={}
            for k in ('VIDEO','IMAGE','GIF'):
                s=self.registry.get((c['id'],k));states[k]=s['mode'] if s else 'NEEDS_DATA_OR_MODEL'
            related=c.get('related')
            rows.append({'id':c['id'],'기구':c['equipment'],'운동':c['name'],'영상':states['VIDEO'],
                '사진':states['IMAGE'],'GIF':states['GIF'],
                '관련_일반모델_자동대체안함':related if related and any(k[0]==related for k in self.registry) else '',
                '상태':'MODEL_CONNECTED_NOT_ALL_EXTERNALLY_VALIDATED' if any(k[0]==c['id'] for k in self.registry) else 'NEEDS_EXACT_EXERCISE_DATA'})
        write_csv(self.run/'coverage_37.csv',rows,list(rows[0]))
        save_json(self.run/'data_needed.json',[{'exercise_id':r['id'],'equipment':r['기구'],'exercise':r['운동'],
            'missing_media':[k for k,col in [('VIDEO','영상'),('IMAGE','사진'),('GIF','GIF')] if r[col]=='NEEDS_DATA_OR_MODEL'],
            'note':'그립/기구/원암/싱글레그 변형을 일반 모델로 대체하지 않음. 자료 수집 후 training_input에 넣고 재실행.'} for r in rows if 'NEEDS_DATA_OR_MODEL' in [r['영상'],r['사진'],r['GIF']]])
        registry={'version':CODE_VERSION,'models':list(self.registry.values()),'run':str(self.run),'requested_items':37}
        save_json(self.run/'registry.json',registry);save_json(self.base/'latest_registry.json',registry)
        have=sum(any(k[0]==r['id'] for k in self.registry) for r in rows)
        summary={'requested_items':37,'exact_target_items_with_any_model':have,'exact_target_items_missing':37-have,
            'connected_exercise_media_pairs':len(self.registry),'new_models_trained':len(training.get('models',[])),
            'new_usable_media':training.get('usable_files',0),'all_original_models_preserved':True,
            'all_37_trained':have==37,'all_37_externally_validated':False,'feedback':'model output + scoped non-diagnostic messages',
            'internet_dataset_downloads':False,'run':str(self.run)}
        save_json(self.run/'summary.json',summary)
        print('\n===== 전체 운동 실행 결과 =====',flush=True)
        print(f'원래 요청한 37개 항목: 모델 연결 {have} / 아직 자료·모델 필요 {37-have}',flush=True)
        print('이번에 새로 학습한 모델:',summary['new_models_trained'],flush=True)
        print('일반 운동을 포함한 연결 조합:',len(self.registry),flush=True)
        print('기존 모델 삭제·재학습 없음 / 모델 존재와 외부 정확도 검증은 별개',flush=True)
        print('전체 현황:',self.run/'coverage_37.csv',flush=True)
        print('미완료 운동:',self.run/'data_needed.json',flush=True)
        print('다음 화면에서 운동과 사진/영상/GIF를 선택해 예측할 수 있습니다.',flush=True)
        return rows,summary

    def feedback(self,result):
        lines=[]
        pred=str(result.get('prediction',result.get('quality',result.get('display_result','')))).upper()
        if result.get('status') in {'FAILED','RECHECK'} or pred=='판단 보류':
            lines+=['판정: 분석 보류',str(result.get('reason',result.get('error','유효한 판정을 만들지 못했습니다.')))]
        elif pred in {'GOOD','GOOD_RANGE'} or pred.startswith('GOOD '):
            lines+=['판정: 모델 기준 정상 범위','현재 입력은 학습한 정상 패턴 쪽으로 분류됐습니다. 모든 반복의 정확성이나 안전성을 보장하지 않습니다.']
        elif pred in {'BAD','BAD_SUSPECTED'} or pred.startswith('BAD '):
            lines+=['판정: 자세 재확인 필요','학습한 오류 패턴 쪽이거나 정상 참고 범위에서 벗어났습니다. 촬영 조건 차이도 원인일 수 있습니다.']
        else:lines+=['판정: '+(pred or '미지원')]
        corrections=result.get('correction',[])
        if corrections:
            for r in corrections:
                if r.get('suspected_issue'):lines+=['학습된 오류 패턴 의심: '+r['suspected_issue']]
                if r.get('suggestion'):lines+=['참고 피드백: '+r['suggestion']]
                if r.get('type'):lines+=['피드백 범위: '+r['type']]
        elif pred not in {'판단 보류',''}:
            lines+=['구체적인 오류 원인은 이 결과만으로 특정하지 않습니다. 같은 운동·비슷한 촬영 각도에서 관절이 잘 보이는지 먼저 확인하세요.']
        for x in result.get('reference_differences',[]):
            lines+=['참고 특징 차이: '+x['feature']+' '+x['direction']+' — 촬영·체형 차이일 수도 있음']
        lines+=['점수는 정확도나 안전 확률이 아닙니다. 일반 안내와 실제 검출된 오류를 구분해서 사용하세요.']
        return '\n'.join(lines)

    def predict(self,exercise,path):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError('입력 파일이 없습니다.')
        if path.suffix.lower() not in MEDIA_EXT:raise ValueError('지원하는 사진·영상·GIF 형식이 아닙니다.')
        spec=self.registry.get((exercise,media_kind(path)))
        if not spec:
            return {'status':'RECHECK','exercise':exercise,'reason':'이 세부 운동과 입력 형식에 맞는 모델이 아직 없습니다. 일반 운동 모델로 자동 대체하지 않았습니다.'}
        out=self.base/'feedback'/token();out.mkdir(parents=True,exist_ok=False)
        if spec['backend']=='v6':
            p=self.root/V6_TEST_REL
            if digest(p)!=V6_TESTER_SHA:raise RuntimeError('확인된 V6 테스트 코드와 다릅니다.')
            ns=runpy.run_path(str(p),run_name='allinone_v6_reuse')
            ns['main'].__globals__['VIDEO_OVERRIDE']=str(path)
            result=ns['main']()
        else:
            cfg={'action':'predict','spec':spec,'media_path':str(path),'expanded_runtime':str(self.expanded_runtime),
                 'title':f'{exercise} [{media_kind(path)}] 분석'}
            result=self.worker_call(cfg,spec.get('sklearn','1.6.1'))
        result.update(input_path=str(path),requested_exercise=exercise,feedback_text=self.feedback(result),
            output_dir=str(out),retrained=False)
        save_json(out/'result.json',result)
        (out/'feedback_KO.txt').write_text(result['feedback_text'],encoding='utf-8')
        print('\n'+result['feedback_text'],flush=True);print('저장:',out,flush=True)
        return result

    def ui(self,rows):
        from IPython.display import display,HTML,clear_output
        try:import ipywidgets as w
        except ImportError:
            print("화면 도구가 없어 함수로 실행합니다: PROJECT.predict('bench_press', '/content/drive/MyDrive/Allinone/test_input/영상.mp4')")
            return
        options=[]
        for c in CATALOG:
            if c['requested'] or any(k[0]==c['id'] for k in self.registry):
                supported='/'.join({'VIDEO':'영상','IMAGE':'사진','GIF':'GIF'}[k] for k in ('VIDEO','IMAGE','GIF') if (c['id'],k) in self.registry)
                options.append((f"{c['equipment']} · {c['name']}  [{supported or '자료 필요'}]",c['id']))
        choices=w.Dropdown(options=options,description='운동',layout=w.Layout(width='95%'))
        # Start with an available non-deadlift exercise rather than another deadlift-only workflow.
        available=[key for _,key in options if key!='deadlift' and any(k[0]==key for k in self.registry)]
        if available:choices.value='bench_press' if 'bench_press' in available else available[0]
        folder=self.root/'test_input';folder.mkdir(exist_ok=True)
        pathbox=w.Text(description='파일 경로',placeholder='파일 업로드 버튼을 누르거나 Drive 경로 입력',layout=w.Layout(width='95%'))
        upload=w.Button(description='사진/영상/GIF 업로드');predict=w.Button(description='선택 운동 분석');info=w.Button(description='37개 현황 보기')
        output=w.Output();status=w.HTML()
        def update(*args):
            c=next(c for c in CATALOG if c['id']==choices.value)
            kinds=[k for k in ('VIDEO','IMAGE','GIF') if (c['id'],k) in self.registry]
            status.value=html.escape('지원 입력: '+(', '.join(kinds) if kinds else '아직 자료·모델 필요')+' | '+('일반 모델로 자동 대체하지 않음' if not kinds else '실험용 모델, 외부 정확도 미확정'))
        choices.observe(update,names='value');update()
        def on_upload(_):
            with output:
                clear_output(wait=True)
                from google.colab import files as cf
                # Upload to a dedicated temporary directory; never overwrite the user's Drive files.
                incoming=self.work/'uploads'/token();incoming.mkdir(parents=True)
                old=os.getcwd();os.chdir(incoming)
                try:uploaded=cf.upload()
                finally:os.chdir(old)
                for name,content in uploaded.items():
                    name=Path(name).name
                    if Path(name).suffix.lower() not in MEDIA_EXT:continue
                    # User uploads for prediction are NOT put in training_input.
                    target=folder/(token()+'_'+name);target.write_bytes(content)
                    pathbox.value=str(target);print('테스트 파일 저장:',target);break
        def on_predict(_):
            with output:
                clear_output(wait=True);predict.disabled=True
                try:
                    print('선택 운동:',choices.label,flush=True)
                    res=self.predict(choices.value,pathbox.value.strip())
                    if not res.get('output_dir'):print(self.feedback(res),flush=True)
                except Exception as e:print('분석 중단:',e,flush=True)
                finally:predict.disabled=False
        def on_info(_):
            with output:
                clear_output(wait=True)
                import pandas as pd
                display(pd.DataFrame(rows)[['기구','운동','영상','사진','GIF']])
                print('자료 부족 목록:',self.run/'data_needed.json')
        upload.on_click(on_upload);predict.on_click(on_predict);info.on_click(on_info)
        display(HTML('<h3>전체 운동 자세 피드백 — 기존 모델 재사용</h3><p>실제로 수행하는 운동과 맞게 선택하세요. 변형·사진/영상 형식이 맞지 않으면 판정을 보류합니다.</p>'))
        display(choices,status,pathbox,w.HBox([upload,predict,info]),output)


def main():
    from google.colab import drive
    drive.mount('/content/drive')
    if not ROOT.is_dir():raise FileNotFoundError('기존 Allinone 폴더가 있는 Google 계정으로 연결하세요.')
    project=AllExerciseProject(ROOT)
    try:
        project.connect_existing();training=project.scan_train();rows,summary=project.coverage(training)
        save_json(project.run/'status.json',{'status':'BATCH_FINISHED_WITH_COVERAGE_REPORT','summary':summary})
        print('\n[3/3] 운동 선택·파일 업로드·피드백 화면',flush=True)
        project.ui(rows)
    except BaseException as e:
        save_json(project.run/'status.json',{'status':'FAILED','error':str(e),'traceback':traceback.format_exc()})
        print('기존 모델은 수정하지 않았습니다. 오류 로그:',project.log,flush=True)
        raise
    return project

if __name__=='__main__':
    PROJECT=main()
