
import os, re, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from train_hybrid_kaggle_py311 import extract

ROOT = Path("/content/drive/MyDrive/Allinone")

MODELS = (
    ROOT / "Hybrid_Training_Results_Py311"
    / "20261027_080653"
)

# 추가 GOOD/BAD 자료를 검색할 폴더
SEARCH_DIRS = [
    ROOT / "BAD_Validation_Input",
    ROOT / "new_dataset"
]

OUT = ROOT / "BAD_Validation_Results" / (
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
OUT.mkdir(parents=True,exist_ok=True)

EXTS = {
    ".mp4",".mov",".avi",".mkv",".webm",
    ".jpg",".jpeg",".png",".webp",".bmp",".gif"
}

NEGATIVE = {
    "bad","wrong","incorrect","faulty",
    "improper","poor","unsafe","invalid",
    "abnormal","error","mistake",
    "나쁜","잘못된","오류","불량"
}

POSITIVE = {
    "good","correct","proper","normal",
    "올바른","정상"
}

def tokens(s):
    return re.sub(
        r"[^a-z0-9가-힣]+"," ",
        str(s).lower()
    ).split()

def label_of(path):
    all_words = []
    for part in path.parts:
        all_words += tokens(part)
    if any(w in NEGATIVE for w in all_words):
        return "BAD"
    if any(w in POSITIVE for w in all_words):
        return "GOOD"
    return "UNLABELED"

model_files = sorted(MODELS.glob("*.joblib"))
names = [p.stem for p in model_files]

def exercise_of(path):
    text = " ".join(tokens(str(path)))
    # 가장 긴 이름 우선
    for name in sorted(names,key=len,reverse=True):
        phrase = name.replace("_"," ")
        if re.search(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
            text
        ):
            return name
    return None

# 학습에 이미 사용한 Kaggle 영상은 제외
media = set()

for folder in SEARCH_DIRS:
    if not folder.exists():
        continue
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if "Kaggle_WorkoutFitness" in p.parts:
            continue
        media.add(p)

media = sorted(media)

print("추가 검증 자료 후보:",len(media),flush=True)

records=[]
excluded=[]

for i,path in enumerate(media,1):
    exercise=exercise_of(path)
    label=label_of(path)

    if exercise is None:
        excluded.append({
            "file":str(path),
            "reason":"UNKNOWN_EXERCISE"
        })
        continue

    if label=="UNLABELED":
        excluded.append({
            "file":str(path),
            "reason":"NO_EXPLICIT_VALIDATION_LABEL"
        })
        continue

    # 사진은 특징 벡터 길이는 같지만
    # 기존 영상 모델과 데이터 분포가 다르므로
    # 영상 모델의 성능 평가에 섞지 않음
    if path.suffix.lower() in {
        ".jpg",".jpeg",".png",".webp",".bmp"
    }:
        excluded.append({
            "file":str(path),
            "reason":"IMAGE_REQUIRES_SEPARATE_VALIDATION"
        })
        continue

    try:
        bundle=joblib.load(
            MODELS / (exercise+".joblib")
        )
        feature=extract(path)

        if feature is None:
            raise ValueError("POSE_EXTRACTION_FAILED")

        X=feature.reshape(1,-1)

        if bundle["mode"]=="ONE_CLASS":
            score=float(
                -bundle["model"].score_samples(
                    bundle["scaler"].transform(X)
                )[0]
            )
            prediction=(
                "BAD" if score>bundle["threshold"]
                else "GOOD"
            )
            threshold=float(bundle["threshold"])
        else:
            prediction=str(
                bundle["model"].predict(X)[0]
            )
            score=None
            threshold=None

        records.append({
            "exercise":exercise,
            "file":str(path),
            "actual":label,
            "predicted":prediction,
            "correct":label==prediction,
            "anomaly_score":score,
            "threshold":threshold,
            "mode":bundle["mode"]
        })

    except Exception as e:
        excluded.append({
            "file":str(path),
            "reason":str(e)
        })

    if i%20==0:
        print("검증 진행:",i,"/",len(media),flush=True)

# 결과 저장
df=pd.DataFrame(records)

if len(df):
    df.to_csv(
        OUT/"validation_results.csv",
        index=False,encoding="utf-8-sig"
    )

    print("\n===== GOOD/BAD 검증 결과 =====")
    print(df.groupby(
        ["exercise","actual","predicted"]
    ).size().to_string())

    print("\n전체 일치율:",
          round(float(df.correct.mean()),4))
    print("주의: 표본 수와 라벨 신뢰도를 함께 확인해야 합니다.")
else:
    print("\nBAD/GOOD 검증 가능한 신규 영상이 없습니다.")
    print("기존 모델은 보존했습니다.")

with open(OUT/"excluded.json","w",encoding="utf-8") as f:
    json.dump(excluded,f,ensure_ascii=False,indent=2)

summary={
    "candidate_files":len(media),
    "validated_files":len(records),
    "excluded_files":len(excluded),
    "actual_good":sum(r["actual"]=="GOOD" for r in records),
    "actual_bad":sum(r["actual"]=="BAD" for r in records)
}

(OUT/"summary.json").write_text(
    json.dumps(summary,ensure_ascii=False,indent=2),
    encoding="utf-8"
)

print("\n===== 최종 요약 =====")
print(json.dumps(summary,ensure_ascii=False,indent=2))
print("결과 저장:",OUT)
