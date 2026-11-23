
import os
os.environ["MPLBACKEND"] = "Agg"

import ast, re, json, zipfile, shutil
import cv2, joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

ROOT = Path("/content/drive/MyDrive/Allinone")
INPUT = ROOT / "new_dataset/MyDeadlift"
SCRIPT = ROOT / "scripts/train_mydeadlift.py"

OUT_ROOT = ROOT / "Deadlift_Improved_Models"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
if RUN.exists():
    RUN = OUT_ROOT / (RUN.name + "_" + os.urandom(3).hex())
RUN.mkdir(parents=True)

TEMP = Path("/content/deadlift_v2_data")
TEMP.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1. 기존 자세 추출 함수 복구
# 학습 실행 부분은 가져오지 않음
# ----------------------------
source = SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source)

function_names = {
    "parse_video",
    "angle",
    "frame_feature",
    "extract"
}
constant_names = {"JOINTS", "REQUIRED"}

nodes = []

for node in tree.body:
    if (
        isinstance(node, ast.FunctionDef)
        and node.name in function_names
    ):
        nodes.append(node)

    elif isinstance(node, ast.Assign):
        targets = [
            t.id for t in node.targets
            if isinstance(t, ast.Name)
        ]
        if any(t in constant_names for t in targets):
            nodes.append(node)

found_functions = {
    n.name for n in nodes
    if isinstance(n, ast.FunctionDef)
}
if not function_names.issubset(found_functions):
    raise RuntimeError(
        "기존 학습 코드에서 필요한 함수를 찾지 못했습니다: "
        + str(function_names-found_functions)
    )

exec(compile(
    ast.fix_missing_locations(
        ast.Module(body=nodes, type_ignores=[])
    ),
    "<restored_deadlift_features>",
    "exec"
))

print("✅ 기존 자세 추출 함수 복구", flush=True)

# ----------------------------
# 2. ZIP 압축 해제
# Drive 대신 코랩 임시 저장소 사용
# ----------------------------
zips = list(INPUT.glob("*.zip"))

if not zips:
    raise FileNotFoundError("MyDeadlift ZIP이 없습니다.")

for archive in zips:
    print("압축 확인:", archive.name, flush=True)

    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue

            target = (TEMP / member.filename).resolve()
            if not target.is_relative_to(TEMP.resolve()):
                continue

            target.parent.mkdir(
                parents=True, exist_ok=True
            )

            if (
                target.exists()
                and target.stat().st_size == member.file_size
            ):
                continue

            with z.open(member) as src:
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

# ----------------------------
# 3. 영상 및 라벨 확인
# ----------------------------
VIDEO_EXT = {
    ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".m4v"
}

videos = sorted(
    p for p in TEMP.rglob("*")
    if p.is_file() and p.suffix.lower() in VIDEO_EXT
)

print("영상 발견:", len(videos), flush=True)

records = []
failures = []

for i, path in enumerate(videos, 1):
    info = parse_video(path)

    if info is None:
        failures.append({
            "file": str(path),
            "reason": "UNKNOWN_LABEL"
        })
        continue

    try:
        features = extract(path)

        if features is None:
            failures.append({
                "file": str(path),
                "reason": "NO_POSE"
            })
            continue

        records.append({
            **info,
            "file": str(path),
            "features": features
        })

    except Exception as e:
        failures.append({
            "file": str(path),
            "reason": str(e)
        })

    if i % 10 == 0:
        print(
            f"자세 추출 {i}/{len(videos)} | "
            f"성공 {len(records)}",
            flush=True
        )

if not records:
    raise RuntimeError("학습 가능한 영상이 없습니다.")

df = pd.DataFrame(records)

# ----------------------------
# 4. 기존과 동일한 피험자 분리
# ----------------------------
train = df[df["split"] == "TRAIN"]
test = df[df["split"] == "TEST"]

Xtrain = np.stack(train.features.to_numpy())
Ytrain = train.label.to_numpy()
Gtrain = train.subject.to_numpy()

Xtest = np.stack(test.features.to_numpy())
Ytest = test.label.to_numpy()

print("\n===== 데이터 구성 =====", flush=True)
print("학습:", len(Ytrain), dict(Counter(Ytrain)), flush=True)
print("테스트:", len(Ytest), dict(Counter(Ytest)), flush=True)
print("학습 피험자:", sorted(set(Gtrain)), flush=True)
print("테스트 피험자:", sorted(set(test.subject)), flush=True)

if len(set(Ytrain)) < 2 or len(set(Ytest)) < 2:
    raise RuntimeError("GOOD/BAD 데이터가 부족합니다.")

# ----------------------------
# 5. 여러 머신러닝 알고리즘 비교
# ----------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "SVM_RBF_C0.1": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf", C=0.1,
            class_weight="balanced"
        )
    ),
    "SVM_RBF_C1": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf", C=1,
            class_weight="balanced"
        )
    ),
    "SVM_RBF_C10": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf", C=10,
            class_weight="balanced"
        )
    )
}

# 피험자 한 명씩 검증에 사용
cv = LeaveOneGroupOut()
comparison = []

for name, model in models.items():
    fold_results = []

    for tr_idx, val_idx in cv.split(
        Xtrain, Ytrain, groups=Gtrain
    ):
        y_tr = Ytrain[tr_idx]
        y_val = Ytrain[val_idx]

        if len(set(y_tr)) < 2:
            continue

        model.fit(Xtrain[tr_idx], y_tr)
        pred = model.predict(Xtrain[val_idx])

        # 한 피험자에게 클래스가 하나만 있다면
        # 해당 클래스 정답률로 평가
        if len(set(y_val)) == 2:
            score = balanced_accuracy_score(
                y_val, pred
            )
        else:
            score = accuracy_score(
                y_val, pred
            )

        fold_results.append(float(score))

    if not fold_results:
        continue

    mean_score = float(np.mean(fold_results))

    comparison.append({
        "model": name,
        "cv_score": mean_score,
        "fold_scores": fold_results
    })

    print(
        f"{name}: CV {mean_score:.4f}",
        flush=True
    )

if not comparison:
    raise RuntimeError("교차검증 실패")

# 학습 피험자에서만 최적 모델 선정
comparison.sort(
    key=lambda x: x["cv_score"],
    reverse=True
)

best_name = comparison[0]["model"]
best_model = models[best_name]

print("\n선택된 모델:", best_name, flush=True)

# ----------------------------
# 6. 전체 학습 피험자로 재학습
# ----------------------------
best_model.fit(Xtrain, Ytrain)

# 최종 테스트 피험자로 평가
pred = best_model.predict(Xtest)

accuracy = float(
    accuracy_score(Ytest, pred)
)
balanced = float(
    balanced_accuracy_score(Ytest, pred)
)

matrix = confusion_matrix(
    Ytest, pred,
    labels=["GOOD", "BAD"]
).tolist()

print("\n===== 최종 평가 =====", flush=True)
print("기존 정확도: 0.6167", flush=True)
print("새 정확도:", round(accuracy, 4), flush=True)
print("기존 균형 정확도: 0.5625", flush=True)
print("새 균형 정확도:", round(balanced, 4), flush=True)
print("혼동행렬:", matrix, flush=True)

print(classification_report(
    Ytest, pred, zero_division=0
), flush=True)

# ----------------------------
# 7. 기존 모델은 그대로 보존
# ----------------------------
model_file = RUN / "deadlift_v2.joblib"

joblib.dump({
    "exercise": "deadlift",
    "mode": "BINARY",
    "feature_version": "pose_summary_v2",
    "algorithm": best_name,
    "model": best_model,
    "train_subjects": sorted(set(Gtrain)),
    "test_subjects": sorted(set(test.subject))
}, model_file)

report = {
    "dataset": "MyDeadlift",
    "training_videos": len(Ytrain),
    "test_videos": len(Ytest),
    "pose_failures": len(failures),
    "selected_algorithm": best_name,
    "cross_validation": comparison,
    "old_accuracy": 0.6166666666666667,
    "new_accuracy": accuracy,
    "old_balanced_accuracy": 0.5625,
    "new_balanced_accuracy": balanced,
    "confusion_matrix": matrix,
    "model_path": str(model_file),
    "note": (
        "Test subjects S04 and S07 have already been "
        "examined in an earlier experiment. "
        "Results are not fully independent validation."
    )
}

(RUN / "report.json").write_text(
    json.dumps(
        report, ensure_ascii=False, indent=2
    ),
    encoding="utf-8"
)

pd.DataFrame(comparison).to_csv(
    RUN / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "file": test.file.to_numpy(),
    "actual": Ytest,
    "predicted": pred
}).to_csv(
    RUN / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n✅ 새 모델 저장:", model_file, flush=True)
print("✅ 기존 모델 유지", flush=True)
