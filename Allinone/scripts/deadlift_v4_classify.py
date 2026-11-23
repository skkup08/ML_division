
import os, json
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from datetime import datetime
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
    recall_score
)

ROOT = Path("/content/drive/MyDrive/Allinone")
SOURCE = ROOT / "Deadlift_V4_Analysis/20261027_125857/video_features.csv"
OUT = ROOT / "Deadlift_V4_Classification"
OUT.mkdir(parents=True,exist_ok=True)

RUN = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN.mkdir(exist_ok=True)

df = pd.read_csv(SOURCE)

# 1. 특징 선택
# 사전에 정한 6개 관절 특징을 사용.
# 전체 데이터에서 찾은 순위를 이용해
# 테스트 피험자에 맞춰 특징을 최적화하지 않음.
METRICS = [
    "trunk_angle",
    "hip_angle",
    "knee_angle",
    "knee_offset",
    "shoulder_hip_offset",
    "knee_asymmetry"
]

FEATURES = [
    f"{metric}_{stat}"
    for metric in METRICS
    for stat in ["mean","std","p10","p90","range"]
]

# 시간 구간별 특징도 포함
FEATURES += [
    f"{metric}_bin{i}"
    for metric in METRICS
    for i in range(1,6)
]

# 유효한 특징만 사용
FEATURES = [f for f in FEATURES if f in df.columns]

# 2. 피험자 분리
train = df[~df.subject.isin(["S04","S07"])].copy()
test = df[df.subject.isin(["S04","S07"])].copy()

if train.empty or test.empty:
    raise RuntimeError("피험자 분리 실패")

# 결측값 처리는 학습 fold 내부에서 수행
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def model_pipeline(estimator, scale=False):
    steps = [("imputer",SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler",StandardScaler()))
    steps.append(("classifier",estimator))
    return Pipeline(steps)

MODELS = {
    "RandomForest": model_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    ),
    "ExtraTrees": model_pipeline(
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    ),
    "SVM": model_pipeline(
        SVC(C=1,kernel="rbf",class_weight="balanced"),
        scale=True
    )
}

def evaluate(mode):
    X = train[FEATURES].to_numpy()
    Xtest = test[FEATURES].to_numpy()

    if mode=="THREE_CLASS":
        y = train.label.to_numpy()
        ytest = test.label.to_numpy()
        classes = ["GB","PB","LLK"]
    else:
        y = np.where(train.label=="GB","GOOD","BAD")
        ytest = np.where(test.label=="GB","GOOD","BAD")
        classes = ["GOOD","BAD"]

    groups = train.subject.to_numpy()
    cv = LeaveOneGroupOut()
    scores = []

    for name,est in MODELS.items():
        predictions = np.empty(len(y),dtype=object)
        predictions[:] = None

        for tr,va in cv.split(X,y,groups):
            # 3클래스 비교에는 모든 클래스가
            # 학습 fold에 있어야 함
            if not set(classes).issubset(set(y[tr])):
                continue

            m = clone(est)
            m.fit(X[tr],y[tr])
            predictions[va] = m.predict(X[va])

        valid = np.array([p is not None for p in predictions])

        if not valid.all():
            continue

        oof = predictions.astype(str)

        # GB/GOOD을 무조건 BAD로 만드는 모델 방지
        if mode=="THREE_CLASS":
            actual_good = y=="GB"
            good_recall = np.mean(oof[actual_good]=="GB")
        else:
            actual_good = y=="GOOD"
            good_recall = np.mean(oof[actual_good]=="GOOD")

        cv_bal = balanced_accuracy_score(y,oof)

        scores.append({
            "mode":mode,
            "algorithm":name,
            "cv_balanced_accuracy":float(cv_bal),
            "cv_good_recall":float(good_recall)
        })

    if not scores:
        print(mode,"교차검증 불가")
        return None

    # CV 균형 정확도 기준 모델 선택.
    # GOOD 재현율은 동률일 때만 사용.
    scores.sort(
        key=lambda x:(
            x["cv_balanced_accuracy"],
            x["cv_good_recall"]
        ),
        reverse=True
    )

    best = scores[0]
    model = clone(MODELS[best["algorithm"]])
    model.fit(X,y)

    pred = model.predict(Xtest)

    result = {
        "mode":mode,
        "selected_algorithm":best["algorithm"],
        "cv_results":scores,
        "test_accuracy":float(accuracy_score(ytest,pred)),
        "test_balanced_accuracy":float(
            balanced_accuracy_score(ytest,pred)
        ),
        "test_confusion_matrix":confusion_matrix(
            ytest,pred,labels=classes
        ).tolist(),
        "labels":classes
    }

    if mode=="THREE_CLASS":
        binary_actual = np.where(ytest=="GB","GOOD","BAD")
        binary_pred = np.where(pred=="GB","GOOD","BAD")

        result["converted_binary_accuracy"] = float(
            accuracy_score(binary_actual,binary_pred)
        )
        result["converted_binary_balanced_accuracy"] = float(
            balanced_accuracy_score(binary_actual,binary_pred)
        )
        result["converted_binary_matrix"] = confusion_matrix(
            binary_actual,binary_pred,
            labels=["GOOD","BAD"]
        ).tolist()

    joblib.dump({
        "exercise":"deadlift",
        "mode":mode,
        "model":model,
        "features":FEATURES,
        "classes":classes,
        "feature_version":"deadlift_v4_metrics"
    },RUN/f"deadlift_{mode.lower()}.joblib")

    pd.DataFrame({
        "subject":test.subject.to_numpy(),
        "actual":ytest,
        "predicted":pred
    }).to_csv(
        RUN/f"predictions_{mode.lower()}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n====",mode,"====",flush=True)
    print("선택 모델:",best["algorithm"],flush=True)
    print("교차검증 균형 정확도:",
          round(best["cv_balanced_accuracy"],4),flush=True)
    print("테스트 정확도:",
          round(result["test_accuracy"],4),flush=True)
    print("테스트 균형 정확도:",
          round(result["test_balanced_accuracy"],4),flush=True)
    print("혼동행렬:",result["test_confusion_matrix"],flush=True)

    if mode=="THREE_CLASS":
        print("GOOD/BAD로 합친 균형 정확도:",
              round(result["converted_binary_balanced_accuracy"],4),
              flush=True)

    return result

print("학습 영상:",len(train),flush=True)
print("테스트 영상:",len(test),flush=True)
print("사용 특징:",len(FEATURES),flush=True)

results = []
for mode in ["THREE_CLASS","BINARY"]:
    r = evaluate(mode)
    if r is not None:
        results.append(r)

(RUN/"comparison.json").write_text(
    json.dumps(results,ensure_ascii=False,indent=2),
    encoding="utf-8"
)

print("\n===== 최종 요약 =====",flush=True)
for r in results:
    print(
        r["mode"],
        "CV:",
        round(r["cv_results"][0]["cv_balanced_accuracy"],4),
        "테스트:",
        round(r["test_balanced_accuracy"],4),
        flush=True
    )

print("저장 위치:",RUN,flush=True)
