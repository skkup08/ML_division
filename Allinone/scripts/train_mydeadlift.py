
import os
os.environ["MPLBACKEND"] = "Agg"

import re, json, zipfile, shutil, hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path("/content/drive/MyDrive/Allinone")
DATA = ROOT / "new_dataset" / "MyDeadlift"
TEMP = Path("/content/MyDeadlift_training")
OUT = ROOT / "Deadlift_Binary_Models"

TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. ZIP 압축 해제
# -------------------------
def safe_extract(archive, destination):
    base = destination.resolve()
    count = 0

    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            target = (base / info.filename).resolve()
            if not target.is_relative_to(base):
                continue

            # 비정상적으로 큰 항목 방지
            if info.file_size > 3 * 1024**3:
                continue

            target.parent.mkdir(
                parents=True, exist_ok=True
            )

            if target.exists() and (
                target.stat().st_size == info.file_size
            ):
                continue

            with z.open(info) as src:
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            count += 1
            if count % 50 == 0:
                print("압축 해제:", count, flush=True)

    return count

for archive in DATA.glob("*.zip"):
    print("압축 해제 시작:", archive.name, flush=True)
    safe_extract(archive, TEMP)

# 내부 ZIP이 있으면 추가 압축 해제
for archive in list(TEMP.rglob("*.zip")):
    nested = archive.parent / (archive.stem + "_extracted")
    nested.mkdir(exist_ok=True)
    safe_extract(archive, nested)

print("✅ 압축 해제 완료", flush=True)

# -------------------------
# 2. 공식 라벨 파싱
# -------------------------
VIDEO_EXT = {
    ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".m4v"
}

def parse_video(path):
    # GB_S01_R01.mp4 형태를 기준으로 인식
    filename = path.stem.upper()

    match = re.search(
        r"(?<![A-Z0-9])(GB|PB|LLK)[_ -]*"
        r"(S0?[1-7])(?:[_ -]|$)",
        filename
    )

    if match:
        label_code = match.group(1)
        subject_number = int(
            match.group(2)[1:]
        )
    else:
        # 폴더 구조에 라벨이 있을 수도 있으므로
        # 파일명과 상위 폴더 검사
        parts = [
            path.stem.upper()
        ] + [
            p.upper() for p in path.parts[-4:-1]
        ]

        tokens = []
        for part in parts:
            tokens += re.findall(
                r"[A-Z]+[0-9]*", part
            )

        labels = [
            x for x in tokens
            if x in {"GB", "PB", "LLK"}
        ]

        subjects = []
        for x in tokens:
            if re.fullmatch(r"S0?[1-7]", x):
                subjects.append(int(x[1:]))

        if len(set(labels)) != 1 or not subjects:
            return None

        label_code = labels[0]
        subject_number = subjects[0]

    label = (
        "GOOD" if label_code == "GB"
        else "BAD"
    )

    # 원본 데이터셋의 공식 피험자 분할
    split = (
        "TEST" if subject_number in {4,7}
        else "TRAIN"
    )

    return {
        "label": label,
        "subclass": label_code,
        "subject": f"S{subject_number:02d}",
        "split": split
    }

videos = sorted(
    p for p in TEMP.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VIDEO_EXT
)

# 중복 파일 제거 (동일 바이트)
items = []
seen = set()
unknown = []

for path in videos:
    info = parse_video(path)
    if info is None:
        unknown.append(str(path))
        continue

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024*1024), b""
        ):
            digest.update(block)

    if digest.hexdigest() in seen:
        continue

    seen.add(digest.hexdigest())
    items.append((path, info))

print("\n===== 데이터 확인 =====", flush=True)
print("영상 파일:", len(videos), flush=True)
print("라벨 인식:", len(items), flush=True)
print("미인식:", len(unknown), flush=True)

print(
    "라벨 분포:",
    dict(Counter(
        x["subclass"] for _,x in items
    )),
    flush=True
)

if len(items) == 0:
    print("미인식 파일 예시:", unknown[:15])
    raise RuntimeError(
        "파일명에서 GB/PB/LLK를 찾지 못했습니다."
    )

# -------------------------
# 3. MediaPipe 자세 추출
# 기존 모델과 같은 6종 요약 특징
# -------------------------
JOINTS = [
    (11,13,15), (12,14,16),
    (13,11,23), (14,12,24),
    (11,23,25), (12,24,26),
    (23,25,27), (24,26,28),
    (25,27,31), (26,28,32)
]

REQUIRED = set(
    j for triple in JOINTS for j in triple
)

def angle(a,b,c):
    u, v = a-b, c-b
    denom = (
        np.linalg.norm(u) *
        np.linalg.norm(v)
    )
    if denom < 1e-8:
        return 0.0
    return float(
        np.arccos(
            np.clip(np.dot(u,v)/denom,-1,1)
        ) / np.pi
    )

def frame_feature(lm):
    p = np.array([
        [v.x,v.y,v.z] for v in lm
    ], dtype=np.float32)

    angles = [
        angle(p[a],p[b],p[c])
        for a,b,c in JOINTS
    ]

    hip = (p[23]+p[24])/2
    shoulder = (p[11]+p[12])/2
    scale = max(
        np.linalg.norm(shoulder-hip),
        0.05
    )

    selected = [
        11,12,13,14,15,16,
        23,24,25,26,27,28,
        29,30,31,32
    ]

    coords = (
        (p[selected]-hip)/scale
    ).flatten()

    return np.concatenate([
        angles, coords
    ]).astype(np.float32)

def extract(path, max_frames=32):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return None

    count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if count <= 0:
        cap.release()
        return None

    indices = np.unique(
        np.linspace(
            0,count-1,
            min(max_frames,count)
        ).astype(int)
    )

    frames = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        for idx in indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx)
            )

            ok, frame = cap.read()
            if not ok:
                continue

            h,w = frame.shape[:2]
            if max(h,w)>960:
                scale = 960/max(h,w)
                frame = cv2.resize(frame,(
                    max(1,int(w*scale)),
                    max(1,int(h*scale))
                ))

            rgb = cv2.cvtColor(
                frame,cv2.COLOR_BGR2RGB
            )

            result = pose.process(
                np.ascontiguousarray(rgb)
            )

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark

            visibility = np.mean([
                lm[i].visibility
                for i in REQUIRED
            ])

            if visibility < 0.35:
                continue

            frames.append(frame_feature(lm))

    cap.release()

    if len(frames)<5:
        return None

    X = np.stack(frames)
    movement = np.abs(
        np.diff(X,axis=0)
    )

    feature = np.concatenate([
        X.mean(axis=0),
        X.std(axis=0),
        np.percentile(X,10,axis=0),
        np.percentile(X,50,axis=0),
        np.percentile(X,90,axis=0),
        movement.mean(axis=0)
    ])

    return feature.astype(np.float32)

# -------------------------
# 4. 포즈 특징 추출
# -------------------------
records = []
failures = []

for i,(path,info) in enumerate(items,1):
    try:
        feature = extract(path)

        if feature is None:
            failures.append({
                "file":str(path),
                "reason":"NO_POSE"
            })
        else:
            records.append({
                **info,
                "file":str(path),
                "feature":feature
            })

    except Exception as e:
        failures.append({
            "file":str(path),
            "reason":str(e)
        })

    if i % 10 == 0:
        print(
            f"자세 추출 {i}/{len(items)} "
            f"| 성공 {len(records)} "
            f"| 실패 {len(failures)}",
            flush=True
        )

# -------------------------
# 5. 피험자별 분리 평가
# -------------------------
if not records:
    raise RuntimeError(
        "사용 가능한 포즈 데이터가 없습니다."
    )

df = pd.DataFrame(records)

train = df[df.split=="TRAIN"]
test = df[df.split=="TEST"]

print("\n===== 학습 자료 =====")
print("TRAIN:",len(train))
print("TEST:",len(test))
print("GOOD/BAD:",
      dict(Counter(df.label)))

if (
    train.label.nunique()!=2
    or test.label.nunique()!=2
):
    raise RuntimeError(
        "학습 또는 테스트에 GOOD/BAD "
        "양쪽 클래스가 부족합니다."
    )

Xtrain = np.stack(train.feature)
Ytrain = train.label.to_numpy()

Xtest = np.stack(test.feature)
Ytest = test.label.to_numpy()

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(Xtrain,Ytrain)
pred = model.predict(Xtest)

accuracy = float(
    accuracy_score(Ytest,pred)
)
balanced = float(
    balanced_accuracy_score(Ytest,pred)
)

matrix = confusion_matrix(
    Ytest,pred,
    labels=["GOOD","BAD"]
).tolist()

print("\n===== 피험자 분리 평가 =====")
print("정확도:",round(accuracy,4))
print(
    "균형 정확도:",
    round(balanced,4)
)
print("혼동행렬:",matrix)

print(classification_report(
    Ytest,pred,zero_division=0
))

# -------------------------
# 6. 모델과 보고서 저장
# -------------------------
# 최종 모델은 평가한 학습 모델 그대로 저장.
# 테스트 피험자는 학습에 포함하지 않음.

RUN = OUT / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
if RUN.exists():
    RUN = OUT / (RUN.name + "_" + os.urandom(3).hex())
RUN.mkdir(parents=True,exist_ok=False)

MODEL_PATH = (
    RUN / "deadlift_good_bad.joblib"
)

joblib.dump({
    "exercise":"deadlift",
    "mode":"BINARY",
    "model":model,
    "labels":["GOOD","BAD"],
    "feature_version":"pose_summary_v2",
    "train_subjects":["S01","S02","S03","S05","S06"],
    "test_subjects":["S04","S07"],
    "subclasses":{
        "GB":"GOOD",
        "PB":"BAD",
        "LLK":"BAD"
    }
}, MODEL_PATH)

report = {
    "dataset":"MyDeadlift",
    "train_videos":len(train),
    "test_videos":len(test),
    "pose_failures":len(failures),
    "accuracy":accuracy,
    "balanced_accuracy":balanced,
    "confusion_matrix":matrix,
    "model_path":str(MODEL_PATH)
}

(RUN/"report.json").write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

pd.DataFrame([
    {
        "file":r["file"],
        "label":r["label"],
        "subclass":r["subclass"],
        "subject":r["subject"],
        "split":r["split"]
    }
    for r in records
]).to_csv(
    RUN/"dataset_manifest.csv",
    index=False,
    encoding="utf-8-sig"
)

(RUN/"failures.json").write_text(
    json.dumps(
        failures,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\n===== 최종 결과 =====")
print(json.dumps(
    report,ensure_ascii=False,indent=2
))
print("✅ 모델 저장 완료:",MODEL_PATH)
