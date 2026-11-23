
import os
os.environ["MPLBACKEND"] = "Agg"

import re, json, zipfile, shutil, hashlib
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score
)

ROOT = Path("/content/drive/MyDrive/Allinone")
INPUT = ROOT / "new_dataset/MyDeadlift"
OUT_ROOT = ROOT / "Deadlift_V3_Models"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
if RUN.exists():
    RUN = OUT_ROOT / (RUN.name + "_" + os.urandom(3).hex())
RUN.mkdir(parents=True, exist_ok=False)

TEMP = Path("/content/deadlift_v3")
TEMP.mkdir(parents=True, exist_ok=True)

CACHE = OUT_ROOT / "v3_feature_cache.joblib"

print("===== DEADLIFT V3 시작 =====", flush=True)
print("저장 위치:", RUN, flush=True)

# ----------------------------------------------------------
# 1. 원본 영상 준비
# ----------------------------------------------------------
VIDEO_EXT = {
    ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".m4v"
}

def safe_unzip(archive, dest):
    dest = dest.resolve()
    count = 0

    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue

            target = (dest / item.filename).resolve()

            if not target.is_relative_to(dest):
                continue

            if item.file_size > 3 * 1024**3:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() and target.stat().st_size == item.file_size:
                continue

            with z.open(item) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

            count += 1

    return count

for archive in INPUT.glob("*.zip"):
    print("ZIP 확인:", archive.name, flush=True)
    safe_unzip(archive, TEMP)

# 중첩 ZIP 지원
for archive in list(TEMP.rglob("*.zip")):
    nested = archive.parent / (archive.stem + "_files")
    nested.mkdir(exist_ok=True)
    safe_unzip(archive, nested)

videos = sorted(
    p for p in TEMP.rglob("*")
    if p.is_file() and p.suffix.lower() in VIDEO_EXT
)

print("찾은 영상:", len(videos), flush=True)

# ----------------------------------------------------------
# 2. 라벨 / 피험자 구분
# ----------------------------------------------------------
def parse_label(path):
    text = path.stem.upper()

    match = re.search(
        r"(?:^|[^A-Z0-9])(GB|PB|LLK)[_ -]*S0?([1-7])"
        r"(?=[^0-9]|$)",
        text
    )

    if match is None:
        return None

    code = match.group(1)
    subject = "S%02d" % int(match.group(2))

    return {
        "label": "GOOD" if code == "GB" else "BAD",
        "subclass": code,
        "subject": subject,
        "split": "TEST" if subject in {"S04", "S07"} else "TRAIN"
    }

# ----------------------------------------------------------
# 3. 동작별 관절 특징
# ----------------------------------------------------------
# MediaPipe 인덱스
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LF, RF = 31, 32

TRIPLES = [
    (LS,LE,LW), (RS,RE,RW),
    (LS,LH,LK), (RS,RH,RK),
    (LH,LK,LA), (RH,RK,RA),
    (LS,LH,LA), (RS,RH,RA)
]

def get_angle(a, b, c):
    u, v = a-b, c-b
    d = np.linalg.norm(u)*np.linalg.norm(v)
    if d < 1e-8:
        return 0.0
    return float(
        np.arccos(np.clip(np.dot(u,v)/d,-1,1))/np.pi
    )

def frame_feature(lm):
    p = np.array(
        [[v.x,v.y,v.z] for v in lm],
        dtype=np.float32
    )

    hip = (p[LH]+p[RH])/2
    shoulder = (p[LS]+p[RS])/2

    # 체형 차이 보정
    torso = max(
        np.linalg.norm((shoulder-hip)[:2]),
        0.05
    )

    # 주요 관절 각도
    angles = [
        get_angle(p[a],p[b],p[c])
        for a,b,c in TRIPLES
    ]

    # 상체 기울기 (영상 화면의 수직축 기준)
    trunk = shoulder[:2]-hip[:2]
    trunk_angle = np.arctan2(
        abs(trunk[0]),
        abs(trunk[1])+1e-8
    )/np.pi

    # 2D 영상에서 무릎-발목 수평 차이
    # 실제 해부학적 전후 거리와 같지는 않음
    knee_ankle = [
        (p[LK,0]-p[LA,0])/torso,
        (p[RK,0]-p[RA,0])/torso
    ]

    # 좌우 고관절/무릎/어깨 상대 위치
    chosen = [
        LS,RS,LH,RH,LK,RK,LA,RA
    ]
    coords = (
        (p[chosen,:2]-hip[:2])/torso
    ).flatten()

    # 허리 위치 대비 어깨 높이
    shoulder_height = (
        hip[1]-shoulder[1]
    )/torso

    return np.concatenate([
        np.array(angles,dtype=np.float32),
        np.array([
            trunk_angle, shoulder_height
        ],dtype=np.float32),
        np.array(knee_ankle,dtype=np.float32),
        coords
    ]).astype(np.float32)

# ----------------------------------------------------------
# 4. 영상의 시간적 특징 추출
# ----------------------------------------------------------
FEATURE_VERSION = "deadlift_temporal_v3_5bins"
N_FRAMES = 64
N_BINS = 5

def extract_temporal(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return None

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if count < 8:
        cap.release()
        return None

    indices = np.unique(
        np.linspace(
            0, count-1,
            min(N_FRAMES,count)
        ).astype(int)
    )

    valid = []
    positions = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        for idx in indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES, int(idx)
            )
            ok, frame = cap.read()

            if not ok:
                continue

            h,w = frame.shape[:2]

            if max(h,w)>960:
                s = 960/max(h,w)
                frame = cv2.resize(
                    frame,
                    (max(1,int(w*s)),max(1,int(h*s)))
                )

            rgb = cv2.cvtColor(
                frame,cv2.COLOR_BGR2RGB
            )

            res = pose.process(
                np.ascontiguousarray(rgb)
            )

            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            required = [
                LS, RS, LH, RH,
                LK, RK, LA, RA
            ]

            visibility = np.mean([
                lm[j].visibility for j in required
            ])

            if visibility < 0.35:
                continue

            valid.append(frame_feature(lm))
            positions.append(idx/max(count-1,1))

    cap.release()

    if len(valid)<8:
        return None

    X = np.stack(valid)
    t = np.array(positions)

    # 유효 프레임을 시간축에 맞게 보간
    # 이 과정은 반복 동작을 분리하지는 않음
    grid = np.linspace(0,1,N_FRAMES)

    interp = np.stack([
        np.interp(grid,t,X[:,j])
        for j in range(X.shape[1])
    ],axis=1)

    # 5개 균등 시간 구간
    bins = np.array_split(interp,N_BINS)

    # 구간별 평균 + 변동성
    means = np.stack([
        x.mean(axis=0) for x in bins
    ])

    stds = np.stack([
        x.std(axis=0) for x in bins
    ])

    # 인접 구간 간 변화량
    changes = np.diff(means,axis=0)

    # 영상 전체 분포
    global_features = np.concatenate([
        interp.mean(axis=0),
        interp.std(axis=0),
        np.percentile(interp,10,axis=0),
        np.percentile(interp,90,axis=0),
        interp.max(axis=0)-interp.min(axis=0)
    ])

    # 전체 가시 프레임의 비율
    quality = np.array([
        len(valid)/len(indices)
    ],dtype=np.float32)

    feature = np.concatenate([
        global_features,
        means.flatten(),
        stds.flatten(),
        changes.flatten(),
        quality
    ]).astype(np.float32)

    if not np.isfinite(feature).all():
        return None

    return feature

# ----------------------------------------------------------
# 5. 중복 제거 및 특징 캐시
# ----------------------------------------------------------
records = []
failures = []
seen = set()

# Drive에 캐시가 있으면 재사용
cache = {}
if CACHE.exists():
    try:
        cached = joblib.load(CACHE)
        if cached.get("version")==FEATURE_VERSION:
            cache = cached.get("items",{})
            print("기존 V3 특징 캐시 재사용",flush=True)
    except Exception:
        cache = {}

for i,path in enumerate(videos,1):
    info = parse_label(path)

    if info is None:
        failures.append({
            "file":str(path),
            "reason":"UNKNOWN_LABEL"
        })
        continue

    # 동일 파일 중복 제거
    h = hashlib.sha256()

    with open(path,"rb") as f:
        for chunk in iter(
            lambda:f.read(1024*1024),b""
        ):
            h.update(chunk)

    digest = h.hexdigest()

    if digest in seen:
        continue
    seen.add(digest)

    if digest in cache:
        feature = cache[digest]
    else:
        try:
            feature = extract_temporal(path)
        except Exception as e:
            failures.append({
                "file":str(path),
                "reason":str(e)
            })
            continue

        if feature is not None:
            cache[digest] = feature

    if feature is None:
        failures.append({
            "file":str(path),
            "reason":"POSE_EXTRACTION_FAILED"
        })
        continue

    records.append({
        **info,
        "file":str(path),
        "feature":feature
    })

    if i%10==0:
        print(
            f"특징 추출 {i}/{len(videos)} | "
            f"성공 {len(records)}",
            flush=True
        )

# 작은 특징 벡터만 Drive에 캐시
joblib.dump({
    "version":FEATURE_VERSION,
    "items":cache
},CACHE)

if not records:
    raise RuntimeError(
        "V3 자세 특징 추출에 실패했습니다."
    )

df = pd.DataFrame(records)

print("\n===== 데이터 확인 =====",flush=True)
print("성공:",len(df),flush=True)
print("실패:",len(failures),flush=True)
print(
    "라벨:",dict(Counter(df.label)),
    flush=True
)

# ----------------------------------------------------------
# 6. 피험자별 분리
# ----------------------------------------------------------
train = df[df.split=="TRAIN"]
test = df[df.split=="TEST"]

Xtrain = np.stack(train.feature.to_numpy())
Ytrain = train.label.to_numpy()
Gtrain = train.subject.to_numpy()

Xtest = np.stack(test.feature.to_numpy())
Ytest = test.label.to_numpy()

if len(set(Ytrain))<2 or len(set(Ytest))<2:
    raise RuntimeError(
        "학습/평가 데이터에 GOOD과 BAD가 모두 필요합니다."
    )

print("학습 영상:",len(Ytrain),flush=True)
print("테스트 영상:",len(Ytest),flush=True)
print("특징 차원:",Xtrain.shape[1],flush=True)

# ----------------------------------------------------------
# 7. 학습 피험자만으로 모델 선택
# ----------------------------------------------------------
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "SVM_C0.1": make_pipeline(
        StandardScaler(),
        SVC(
            C=0.1,kernel="rbf",
            class_weight="balanced"
        )
    ),
    "SVM_C1": make_pipeline(
        StandardScaler(),
        SVC(
            C=1,kernel="rbf",
            class_weight="balanced"
        )
    ),
    "SVM_C10": make_pipeline(
        StandardScaler(),
        SVC(
            C=10,kernel="rbf",
            class_weight="balanced"
        )
    )
}

cv = LeaveOneGroupOut()
comparisons = []

for name,estimator in MODELS.items():
    oof = np.empty(len(Ytrain),dtype=object)
    oof[:] = None
    folds = []

    for tr,va in cv.split(
        Xtrain,Ytrain,Gtrain
    ):
        if len(set(Ytrain[tr]))<2:
            continue

        model = clone(estimator)
        model.fit(Xtrain[tr],Ytrain[tr])
        pred = model.predict(Xtrain[va])

        oof[va] = pred
        folds.append({
            "subject":str(Gtrain[va][0]),
            "accuracy":float(
                accuracy_score(Ytrain[va],pred)
            )
        })

    valid = np.array([
        x is not None for x in oof
    ])

    if not valid.any():
        continue

    score = float(
        balanced_accuracy_score(
            Ytrain[valid],oof[valid].astype(str)
        )
    )

    comparisons.append({
        "algorithm":name,
        "cv_balanced_accuracy":score,
        "folds":folds
    })

    print(
        f"{name}: CV 균형 정확도 {score:.4f}",
        flush=True
    )

if not comparisons:
    raise RuntimeError("교차검증에 실패했습니다.")

comparisons.sort(
    key=lambda x:x["cv_balanced_accuracy"],
    reverse=True
)

winner = comparisons[0]["algorithm"]

print(
    "\n선택된 알고리즘:",winner,
    flush=True
)

# ----------------------------------------------------------
# 8. 최종 학습 및 테스트
# ----------------------------------------------------------
model = clone(MODELS[winner])
model.fit(Xtrain,Ytrain)

pred = model.predict(Xtest)

accuracy = float(
    accuracy_score(Ytest,pred)
)

balanced = float(
    balanced_accuracy_score(Ytest,pred)
)

matrix = confusion_matrix(
    Ytest,pred,labels=["GOOD","BAD"]
).tolist()

good_recall = float(
    recall_score(
        Ytest,pred,
        pos_label="GOOD"
    )
)

bad_recall = float(
    recall_score(
        Ytest,pred,
        pos_label="BAD"
    )
)

print("\n===== 데드리프트 V3 결과 =====",flush=True)
print("V1 균형 정확도: 0.5625",flush=True)
print("V2 균형 정확도: 0.5625",flush=True)
print(
    f"V3 균형 정확도: {balanced:.4f}",
    flush=True
)
print(
    f"V3 전체 정확도: {accuracy:.4f}",
    flush=True
)
print(
    f"GOOD 인식률: {good_recall:.4f}",
    flush=True
)
print(
    f"BAD 인식률: {bad_recall:.4f}",
    flush=True
)
print("혼동행렬:",matrix,flush=True)

print(classification_report(
    Ytest,pred,zero_division=0
),flush=True)

# BAD 세부 유형 평가
subclass_results = {}
for code in ["GB","PB","LLK"]:
    mask = test.subclass.to_numpy()==code

    if mask.any():
        subclass_results[code] = {
            "count":int(mask.sum()),
            "accuracy":float(
                accuracy_score(
                    Ytest[mask],pred[mask]
                )
            )
        }

# ----------------------------------------------------------
# 9. 결과 Drive 저장
# ----------------------------------------------------------
MODEL_PATH = RUN/"deadlift_v3.joblib"

joblib.dump({
    "exercise":"deadlift",
    "mode":"BINARY",
    "algorithm":winner,
    "model":model,
    "feature_version":FEATURE_VERSION,
    "frame_count":N_FRAMES,
    "time_bins":N_BINS,
    "train_subjects":sorted(set(Gtrain)),
    "test_subjects":sorted(set(test.subject)),
    "labels":["GOOD","BAD"]
},MODEL_PATH)

report = {
    "dataset":"MyDeadlift",
    "usable_videos":len(df),
    "train_videos":len(train),
    "test_videos":len(test),
    "pose_failures":len(failures),
    "algorithm":winner,
    "cv_results":comparisons,
    "accuracy":accuracy,
    "balanced_accuracy":balanced,
    "good_recall":good_recall,
    "bad_recall":bad_recall,
    "confusion_matrix":matrix,
    "subclass_results":subclass_results,
    "model_path":str(MODEL_PATH),
    "evaluation_note":(
        "S04 and S07 were used in previous "
        "experiments; not a fresh external test."
    )
}

(RUN/"report.json").write_text(
    json.dumps(
        report,ensure_ascii=False,indent=2
    ),
    encoding="utf-8"
)

pd.DataFrame(comparisons).to_csv(
    RUN/"model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "file":test.file.to_numpy(),
    "subject":test.subject.to_numpy(),
    "subclass":test.subclass.to_numpy(),
    "actual":Ytest,
    "predicted":pred
}).to_csv(
    RUN/"test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

(RUN/"failures.json").write_text(
    json.dumps(
        failures,ensure_ascii=False,indent=2
    ),
    encoding="utf-8"
)

print("\n✅ V3 모델 저장:",MODEL_PATH,flush=True)
print("✅ 기존 V1/V2 모델은 그대로 유지",flush=True)
