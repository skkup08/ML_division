
import os, re, json, zipfile, hashlib
import requests, joblib
import numpy as np
import mediapipe as mp

from pathlib import Path
from collections import Counter
from datetime import datetime
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path("/content/drive/MyDrive/Allinone")
DATA = ROOT / "new_dataset" / "Zenodo_Squat"
OUT = ROOT / "Squat_Image_Models"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# 1. Zenodo API에서 실제 다운로드 주소 조회
api = "https://zenodo.org/api/records/17558630"
response = requests.get(api, timeout=60)
response.raise_for_status()

items = response.json()["files"]
zips = [f for f in items if f["key"].lower().endswith(".zip")]

if not zips:
    raise RuntimeError("Zenodo ZIP 파일을 찾지 못했습니다.")

for item in zips:
    target = DATA / Path(item["key"]).name
    expected = item.get("size", 0)

    if target.exists() and (not expected or target.stat().st_size == expected):
        print("기존 ZIP 재사용:", target.name, flush=True)
        continue

    url = item["links"].get("self") or item["links"].get("content")
    if not url:
        raise RuntimeError("다운로드 주소 없음: " + item["key"])

    temp = target.with_suffix(".part")
    print("다운로드:", item["key"], flush=True)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(temp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    if expected and temp.stat().st_size != expected:
        raise RuntimeError("다운로드 크기 불일치")

    temp.replace(target)

# 2. 압축 해제 (경로 탈출 방지)
EXTRACT = DATA / "extracted"
EXTRACT.mkdir(exist_ok=True)

for archive in DATA.glob("*.zip"):
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue

            dest = (EXTRACT / member.filename).resolve()
            if not dest.is_relative_to(EXTRACT.resolve()):
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)

            if not dest.exists() or dest.stat().st_size != member.file_size:
                with z.open(member) as src, open(dest, "wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)

print("압축 해제 완료", flush=True)

# 3. 이미지 자동 라벨
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_label(path):
    relative = str(path.relative_to(EXTRACT)).lower()
    words = re.sub(r"[^a-z0-9]+", " ", relative).split()

    if any(w in words for w in ["bad", "wrong", "incorrect"]):
        return "BAD"
    if any(w in words for w in ["good", "correct"]):
        return "GOOD"
    return None

images = [
    p for p in EXTRACT.rglob("*")
    if p.suffix.lower() in EXTS
]

labels = Counter(get_label(p) for p in images)
print("이미지 총수:", len(images), flush=True)
print("라벨 분포:", dict(labels), flush=True)

# 4. 사진에서 관절 각도 및 상대 위치 추출
JOINTS = [
    (11,13,15), (12,14,16),
    (13,11,23), (14,12,24),
    (11,23,25), (12,24,26),
    (23,25,27), (24,26,28),
    (25,27,31), (26,28,32)
]

SELECTED = [
    11,12,13,14,15,16,23,24,
    25,26,27,28,29,30,31,32
]

def angle(a,b,c):
    u, v = a-b, c-b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-8:
        return 0.
    return float(np.arccos(
        np.clip(np.dot(u,v)/denom, -1, 1)
    ) / np.pi)

def extract(lm):
    p = np.array([
        [x.x,x.y,x.z] for x in lm
    ], dtype=np.float32)

    angles = [
        angle(p[a],p[b],p[c])
        for a,b,c in JOINTS
    ]

    hip = (p[23]+p[24])/2
    shoulder = (p[11]+p[12])/2
    scale = max(np.linalg.norm(shoulder-hip), 0.05)
    coords = ((p[SELECTED]-hip)/scale).flatten()

    return np.concatenate([angles,coords]).astype(np.float32)

X, Y, names, hashes = [], [], [], set()
failures = Counter()

with mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
) as pose:

    for i,path in enumerate(images,1):
        label = get_label(path)
        if label is None:
            failures["NO_LABEL"] += 1
            continue

        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((960,960))
                rgb = np.asarray(im)

            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in hashes:
                failures["DUPLICATE_IMAGE"] += 1
                continue

            result = pose.process(np.ascontiguousarray(rgb))
            if not result.pose_landmarks:
                failures["NO_POSE"] += 1
                continue

            feature = extract(result.pose_landmarks.landmark)
            if not np.isfinite(feature).all():
                failures["INVALID_FEATURE"] += 1
                continue

            hashes.add(digest)
            X.append(feature)
            Y.append(label)
            names.append(str(path))

        except Exception as e:
            failures[type(e).__name__] += 1

        if i % 200 == 0:
            print(
                f"처리 {i}/{len(images)}, 성공 {len(X)}",
                flush=True
            )

# 5. 이진 분류 학습
counts = Counter(Y)
print("실제 학습 가능한 자료:", dict(counts), flush=True)
print("제외:", dict(failures), flush=True)

summary = {
    "dataset": "Zenodo 17558630",
    "usable_labels": dict(counts),
    "excluded": dict(failures)
}

if counts["GOOD"] >= 10 and counts["BAD"] >= 10:
    X = np.stack(X)
    Y = np.array(Y)

    train_idx, test_idx = train_test_split(
        np.arange(len(Y)),
        test_size=0.2,
        stratify=Y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X[train_idx], Y[train_idx])
    pred = model.predict(X[test_idx])

    print("\n===== 스쿼트 GOOD/BAD 평가 =====")
    print(classification_report(
        Y[test_idx], pred, zero_division=0
    ))

    summary["confusion_matrix"] = confusion_matrix(
        Y[test_idx], pred, labels=["GOOD","BAD"]
    ).tolist()
    summary["test_count"] = len(test_idx)

    # 평가 완료 후 전체 자료로 최종 학습
    final_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X,Y)

    output = OUT / "squat_image_good_bad.joblib"

    joblib.dump({
        "exercise": "squat",
        "media_type": "IMAGE",
        "mode": "BINARY",
        "model": final_model,
        "feature_version": "image_pose_v1",
        "labels": ["GOOD","BAD"]
    }, output)

    summary["status"] = "TRAINED"
    summary["model_path"] = str(output)
    print("✅ 이미지 모델 저장:", output)

else:
    summary["status"] = "INSUFFICIENT_LABELED_POSES"
    print("GOOD/BAD 자료가 부족해 모델을 생성하지 않았습니다.")

(OUT/"training_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n===== 최종 결과 =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
