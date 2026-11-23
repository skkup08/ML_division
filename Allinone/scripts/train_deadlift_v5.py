
import os
os.environ["MPLBACKEND"] = "Agg"

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix
)

ROOT = Path("/content/drive/MyDrive/Allinone")

SOURCE = (
    ROOT / "Deadlift_V4_Analysis"
    / "20261027_125857"
    / "video_features.csv"
)

OUT = ROOT / "Deadlift_V5_Models"
OUT.mkdir(parents=True, exist_ok=True)

RUN = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
if RUN.exists():
    RUN = OUT / (RUN.name + "_" + os.urandom(3).hex())
RUN.mkdir(parents=True, exist_ok=False)

# ------------------------------------------------
# 1. 기존 관절 특징 불러오기
# ------------------------------------------------

df = pd.read_csv(SOURCE)

METRICS = [
    "trunk_angle",
    "hip_angle",
    "knee_angle",
    "knee_offset",
    "shoulder_hip_offset",
    "knee_asymmetry"
]

FEATURES = [
    f"{m}_{s}"
    for m in METRICS
    for s in ["mean", "std", "p10", "p90", "range"]
]

FEATURES += [
    f"{m}_bin{i}"
    for m in METRICS
    for i in range(1, 6)
]

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise RuntimeError("특징 누락: " + str(missing))

# 기존 분할 유지
train = df[
    ~df.subject.isin(["S04", "S07"])
].copy()

test = df[
    df.subject.isin(["S04", "S07"])
].copy()

if train.empty or test.empty:
    raise RuntimeError("학습 또는 테스트 데이터가 없습니다.")

X = train[FEATURES].to_numpy()
y = np.where(train.label == "GB", "GOOD", "BAD")
groups = train.subject.to_numpy()

Xt = test[FEATURES].to_numpy()
yt = np.where(test.label == "GB", "GOOD", "BAD")

print("===== 데이터 확인 =====", flush=True)
print("학습 영상:", len(train), flush=True)
print("테스트 영상:", len(test), flush=True)
print("사용 특징:", len(FEATURES), flush=True)

# ------------------------------------------------
# 2. 모델 후보
# ------------------------------------------------

MODELS = {
    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])
}

# ------------------------------------------------
# 3. 피험자별 교차검증
# ------------------------------------------------

cv = LeaveOneGroupOut()
results = []

for name, estimator in MODELS.items():
    oof_prob = np.full(len(y), np.nan)

    for tr, val in cv.split(X, y, groups):
        model = clone(estimator)
        model.fit(X[tr], y[tr])

        classes = list(model.classes_)
        bad_index = classes.index("BAD")

        oof_prob[val] = model.predict_proba(
            X[val]
        )[:, bad_index]

    valid = np.isfinite(oof_prob)
    if not valid.all():
        continue

    oof_pred = np.where(
        oof_prob >= 0.5,
        "BAD",
        "GOOD"
    )

    score = balanced_accuracy_score(y, oof_pred)

    good_recall = np.mean(
        oof_pred[y == "GOOD"] == "GOOD"
    )

    results.append({
        "name": name,
        "cv_balanced_accuracy": float(score),
        "cv_good_recall": float(good_recall),
        "probabilities": oof_prob
    })

    print(
        f"{name}: CV={score:.4f}, "
        f"GOOD 인식률={good_recall:.4f}",
        flush=True
    )

if not results:
    raise RuntimeError("교차검증 실패")

# 균형 정확도를 우선하고 GOOD 인식률로 동률 결정
results.sort(
    key=lambda r: (
        r["cv_balanced_accuracy"],
        r["cv_good_recall"]
    ),
    reverse=True
)

best = results[0]
name = best["name"]
prob = best["probabilities"]

print("\n선택 모델:", name, flush=True)

# ------------------------------------------------
# 4. GOOD/BAD 경계 보정
# ------------------------------------------------
# 교차검증 예측값으로만 임계값 결정
# 확률값은 미보정 모델 점수로 취급

good_scores = prob[y == "GOOD"]
bad_scores = prob[y == "BAD"]

# 정상 자세를 BAD로 잘못 판단하는 비율을
# CV GOOD 기준 약 10% 이하로 억제하는 방향
bad_threshold = max(
    0.55,
    float(np.quantile(good_scores, 0.90))
)

# BAD 영상을 GOOD으로 단정하지 않도록
# GOOD 판정에도 보수적 기준 설정
good_threshold = min(
    0.45,
    float(np.quantile(bad_scores, 0.10))
)

def classify(scores):
    return np.where(
        scores <= good_threshold,
        "GOOD",
        np.where(
            scores >= bad_threshold,
            "BAD",
            "RECHECK"
        )
    )

oof_labels = classify(prob)

print("\n===== CV 판정 기준 =====", flush=True)
print("GOOD 경계:", round(good_threshold, 4), flush=True)
print("BAD 경계:", round(bad_threshold, 4), flush=True)

# ------------------------------------------------
# 5. 판정 평가
# ------------------------------------------------

def evaluate(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    decided = predicted != "RECHECK"

    coverage = float(np.mean(decided))

    good_mask = actual == "GOOD"
    bad_mask = actual == "BAD"

    good_recall = float(
        np.mean(predicted[good_mask] == "GOOD")
    )
    bad_recall = float(
        np.mean(predicted[bad_mask] == "BAD")
    )

    false_bad_rate = float(
        np.mean(predicted[good_mask] == "BAD")
    )
    false_good_rate = float(
        np.mean(predicted[bad_mask] == "GOOD")
    )

    decided_accuracy = (
        float(accuracy_score(
            actual[decided],
            predicted[decided]
        ))
        if decided.any()
        else None
    )

    return {
        "coverage": coverage,
        "good_recall_all": good_recall,
        "bad_recall_all": bad_recall,
        "good_false_bad_rate": false_bad_rate,
        "bad_false_good_rate": false_good_rate,
        "recheck_count": int(np.sum(~decided)),
        "decided_accuracy": decided_accuracy,
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=["GOOD", "BAD", "RECHECK"]
        ).tolist()
    }

cv_metrics = evaluate(y, oof_labels)

# ------------------------------------------------
# 6. 전체 학습 피험자로 최종 모델 학습
# ------------------------------------------------

model = clone(MODELS[name])
model.fit(X, y)

bad_idx = list(model.classes_).index("BAD")

test_prob = model.predict_proba(
    Xt
)[:, bad_idx]

test_labels = classify(test_prob)
test_metrics = evaluate(yt, test_labels)

print("\n===== V5 평가 결과 =====", flush=True)

print("학습 피험자 교차검증:")
print(json.dumps(
    cv_metrics,
    ensure_ascii=False,
    indent=2
), flush=True)

print("\nS04/S07 참고 평가:")
print(json.dumps(
    test_metrics,
    ensure_ascii=False,
    indent=2
), flush=True)

# ------------------------------------------------
# 7. 결과 보존
# ------------------------------------------------

MODEL = RUN / "deadlift_v5.joblib"

joblib.dump({
    "exercise": "deadlift",
    "version": "V5",
    "mode": "BINARY_WITH_RECHECK",
    "algorithm": name,
    "model": model,
    "features": FEATURES,
    "good_threshold": good_threshold,
    "bad_threshold": bad_threshold,
    "classes": list(model.classes_),
    "feature_version": "deadlift_v4_metrics",
    "note": "Scores are not calibrated probabilities"
}, MODEL)

pd.DataFrame({
    "subject": test.subject,
    "actual": yt,
    "bad_score": test_prob,
    "predicted": test_labels
}).to_csv(
    RUN / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([
    {
        "algorithm": r["name"],
        "cv_balanced_accuracy": r["cv_balanced_accuracy"],
        "cv_good_recall": r["cv_good_recall"]
    }
    for r in results
]).to_csv(
    RUN / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

report = {
    "algorithm": name,
    "train_videos": len(train),
    "test_videos": len(test),
    "good_threshold": good_threshold,
    "bad_threshold": bad_threshold,
    "cv_metrics": cv_metrics,
    "test_metrics": test_metrics,
    "model_path": str(MODEL),
    "warning": (
        "S04/S07 have been used in earlier experiments. "
        "Not an independent final validation."
    )
}

(RUN / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n===== V5 최종 요약 =====", flush=True)
print("선택 모델:", name, flush=True)
print(
    "CV GOOD 오탐률:",
    f"{cv_metrics['good_false_bad_rate']:.2%}",
    flush=True
)
print(
    "CV 재확인 비율:",
    f"{1-cv_metrics['coverage']:.2%}",
    flush=True
)
print(
    "참고 테스트 GOOD 오탐률:",
    f"{test_metrics['good_false_bad_rate']:.2%}",
    flush=True
)
print(
    "참고 테스트 BAD 탐지율:",
    f"{test_metrics['bad_recall_all']:.2%}",
    flush=True
)
print("모델 저장:", MODEL, flush=True)
