
import os
os.environ["MPLBACKEND"] = "Agg"

import re, json, zipfile, shutil, hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp

ROOT = Path("/content/drive/MyDrive/Allinone")
DATA = ROOT / "new_dataset/MyDeadlift"
TEMP = Path("/content/deadlift_v4_analysis")
OUT = ROOT / "Deadlift_V4_Analysis"
CACHE = OUT / "feature_cache.csv"

TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

RUN = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
if RUN.exists():
    RUN = OUT / (RUN.name + "_" + os.urandom(3).hex())
RUN.mkdir(parents=True)

# 1. ZIP을 코랩 임시 저장소에 압축 해제
for archive in DATA.glob("*.zip"):
    print("압축 확인:", archive.name, flush=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            dest = (TEMP / member.filename).resolve()
            if not dest.is_relative_to(TEMP.resolve()):
                continue
            if member.file_size > 3 * 1024**3:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == member.file_size:
                continue
            with z.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

# 중첩 ZIP도 검사
for archive in list(TEMP.rglob("*.zip")):
    dest_dir = archive.parent / (archive.stem + "_extracted")
    dest_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            target = (dest_dir / member.filename).resolve()
            if not target.is_relative_to(dest_dir.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

videos = sorted(
    p for p in TEMP.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTS
)

def parse(path):
    m = re.search(
        r"(?:^|[^A-Z0-9])(GB|PB|LLK)[_ -]*S0?([1-7])(?=[^0-9]|$)",
        path.stem.upper()
    )
    if not m:
        return None
    return m.group(1), "S%02d" % int(m.group(2))

# MediaPipe 좌표
LS, RS = 11, 12
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LF, RF = 31, 32

def angle(a, b, c):
    u = a-b
    v = c-b
    den = np.linalg.norm(u)*np.linalg.norm(v)
    if den < 1e-8:
        return np.nan
    return np.degrees(np.arccos(
        np.clip(np.dot(u,v)/den, -1, 1)
    ))

def measure(landmarks):
    p = np.array(
        [[x.x,x.y,x.z] for x in landmarks],
        dtype=np.float32
    )

    hip = (p[LH]+p[RH])/2
    shoulder = (p[LS]+p[RS])/2
    torso = shoulder-hip
    scale = max(np.linalg.norm(torso[:2]), 0.05)

    # 2D 투영 관절 각도
    hip_angle = np.mean([
        angle(p[LS,:2],p[LH,:2],p[LK,:2]),
        angle(p[RS,:2],p[RH,:2],p[RK,:2])
    ])
    knee_angle = np.mean([
        angle(p[LH,:2],p[LK,:2],p[LA,:2]),
        angle(p[RH,:2],p[RK,:2],p[RA,:2])
    ])

    # 상체와 화면 수직축 사이의 2D 각도
    trunk_angle = np.degrees(
        np.arctan2(abs(torso[0]), abs(torso[1])+1e-8)
    )

    # 무릎-발목 수평 차이 (촬영 방향 영향을 받음)
    knee_offset = np.mean([
        abs(p[LK,0]-p[LA,0]),
        abs(p[RK,0]-p[RA,0])
    ]) / scale

    # 어깨와 엉덩이의 화면 내 수평 차이
    shoulder_hip_offset = abs(
        shoulder[0]-hip[0]
    ) / scale

    # 좌우 무릎 각도의 차이
    left_knee = angle(
        p[LH,:2],p[LK,:2],p[LA,:2]
    )
    right_knee = angle(
        p[RH,:2],p[RK,:2],p[RA,:2]
    )

    return {
        "trunk_angle": trunk_angle,
        "hip_angle": hip_angle,
        "knee_angle": knee_angle,
        "knee_offset": knee_offset,
        "shoulder_hip_offset": shoulder_hip_offset,
        "knee_asymmetry": abs(left_knee-right_knee)
    }

METRICS = [
    "trunk_angle", "hip_angle", "knee_angle",
    "knee_offset", "shoulder_hip_offset",
    "knee_asymmetry"
]

def extract(path, sample_frames=48):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 8:
        cap.release()
        return None

    idxs = np.unique(np.linspace(
        0, count-1, min(count,sample_frames)
    ).astype(int))

    series = []
    times = []

    try:
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue

                h,w = frame.shape[:2]
                if max(h,w)>960:
                    s = 960/max(h,w)
                    frame = cv2.resize(
                        frame,(max(1,int(w*s)),max(1,int(h*s)))
                    )

                rgb = cv2.cvtColor(
                    frame,cv2.COLOR_BGR2RGB
                )
                result = pose.process(
                    np.ascontiguousarray(rgb)
                )

                if not result.pose_landmarks:
                    continue

                lm = result.pose_landmarks.landmark
                required = [
                    LS,RS,LH,RH,LK,RK,LA,RA
                ]

                if np.mean([
                    lm[j].visibility for j in required
                ]) < 0.35:
                    continue

                series.append(measure(lm))
                times.append(idx/max(count-1,1))
    finally:
        cap.release()

    if len(series)<8:
        return None

    frame_df = pd.DataFrame(series)
    frame_df["time"] = times
    frame_df = frame_df.dropna()

    if len(frame_df)<8:
        return None

    summary = {
        "valid_frames": len(frame_df),
        "valid_ratio": len(frame_df)/len(idxs)
    }

    # 전체 운동의 분포 요약
    for metric in METRICS:
        vals = frame_df[metric].to_numpy()
        summary[metric+"_mean"] = float(np.mean(vals))
        summary[metric+"_std"] = float(np.std(vals))
        summary[metric+"_p10"] = float(np.percentile(vals,10))
        summary[metric+"_p90"] = float(np.percentile(vals,90))
        summary[metric+"_range"] = float(
            np.percentile(vals,90)-np.percentile(vals,10)
        )

        # 시간 구간별 평균 (실제 반복 구간은 아님)
        for j in range(5):
            lo, hi = j/5, (j+1)/5
            block = frame_df[
                (frame_df.time>=lo) &
                (frame_df.time<hi if j<4 else frame_df.time<=hi)
            ]
            summary[f"{metric}_bin{j+1}"] = (
                float(block[metric].mean())
                if len(block) else np.nan
            )

    return summary

# 기존 분석 결과가 있으면 특징 추출 재사용
cache = {}
if CACHE.exists():
    try:
        cached = pd.read_csv(CACHE)
        if "digest" in cached.columns:
            cache = {
                str(r["digest"]):r.to_dict()
                for _,r in cached.iterrows()
            }
            print("기존 특징 캐시 로드",flush=True)
    except Exception:
        cache = {}

records = []
failures = []
seen = set()

print("발견한 영상:",len(videos),flush=True)

for i,path in enumerate(videos,1):
    info = parse(path)
    if info is None:
        failures.append({
            "file":str(path),"reason":"UNKNOWN_LABEL"
        })
        continue

    digest = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    digest = digest.hexdigest()

    if digest in seen:
        continue
    seen.add(digest)

    if digest in cache:
        features = {
            k:v for k,v in cache[digest].items()
            if k not in {"digest","file","label","subject"}
        }
    else:
        try:
            features = extract(path)
        except Exception as e:
            failures.append({
                "file":str(path),"reason":str(e)
            })
            continue

    if features is None:
        failures.append({
            "file":str(path),
            "reason":"NO_USABLE_POSE"
        })
        continue

    label,subject = info

    records.append({
        "digest":digest,
        "file":str(path),
        "label":label,
        "subject":subject,
        **features
    })

    if i%10==0:
        print(
            f"분석 {i}/{len(videos)} | 성공 {len(records)}",
            flush=True
        )

if not records:
    raise RuntimeError("분석 가능한 영상이 없습니다.")

df = pd.DataFrame(records)
df.to_csv(CACHE,index=False)
df.to_csv(
    RUN/"video_features.csv",
    index=False,encoding="utf-8-sig"
)

# 2. GOOD / PB / LLK 특징 분포 분석
print("\n===== 데이터 분포 =====",flush=True)
print(df.groupby(["subject","label"]).size().to_string())

summary_cols = [m+"_mean" for m in METRICS]
group_means = df.groupby("label")[summary_cols].agg(
    ["mean","std","count"]
)
group_means.to_csv(
    RUN/"label_feature_summary.csv",
    encoding="utf-8-sig"
)

# 3. 피험자 내부의 GB 대비 차이
# 동일 피험자끼리 비교해 개인별 체형 차이 완화
subject_means = df.groupby(
    ["subject","label"]
)[summary_cols].mean()

comparisons = []

for subject in sorted(df.subject.unique()):
    if (subject,"GB") not in subject_means.index:
        continue

    good = subject_means.loc[(subject,"GB")]

    for bad_label in ("PB","LLK"):
        if (subject,bad_label) not in subject_means.index:
            continue

        bad = subject_means.loc[(subject,bad_label)]

        for feature in summary_cols:
            comparisons.append({
                "subject":subject,
                "bad_type":bad_label,
                "feature":feature,
                "good_mean":float(good[feature]),
                "bad_mean":float(bad[feature]),
                "difference":float(bad[feature]-good[feature])
            })

diff_df = pd.DataFrame(comparisons)
diff_df.to_csv(
    RUN/"subject_matched_differences.csv",
    index=False,encoding="utf-8-sig"
)

# 4. 특징별 일관성 순위
# 값 자체의 크기가 다른 특징들을 단순 비교하지 않도록
# 피험자 간 차이의 평균 / 표준편차 계산
ranking = []

if not diff_df.empty:
    for (bad_type, feature), g in diff_df.groupby(
        ["bad_type","feature"]
    ):
        vals = g.difference.to_numpy()
        mean = float(np.mean(vals))
        sd = float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan
        consistency = float(max(
            np.mean(vals>0),np.mean(vals<0)
        ))

        ranking.append({
            "bad_type":bad_type,
            "feature":feature,
            "n_subjects":len(vals),
            "mean_difference":mean,
            "std_difference":sd,
            "sign_consistency":consistency,
            "standardized_effect":(
                abs(mean)/sd if np.isfinite(sd) and sd>1e-8
                else np.nan
            )
        })

rank_df = pd.DataFrame(ranking)

if not rank_df.empty:
    rank_df = rank_df.sort_values(
        ["bad_type","sign_consistency","standardized_effect"],
        ascending=[True,False,False]
    )
    rank_df.to_csv(
        RUN/"feature_ranking.csv",
        index=False,encoding="utf-8-sig"
    )

# 5. GOOD/PB/LLK 그래프 생성
for metric in METRICS:
    feature = metric+"_mean"

    groups = [
        df[df.label==label][feature].dropna().to_numpy()
        for label in ["GB","PB","LLK"]
    ]

    if not all(len(g) for g in groups):
        continue

    fig,ax = plt.subplots(figsize=(7,4))
    ax.boxplot(
        groups,
        tick_labels=["GOOD (GB)","BAD (PB)","BAD (LLK)"]
    )
    ax.set_title(metric+" - per video distribution")
    ax.set_ylabel(metric)
    ax.grid(axis="y",alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RUN/f"{metric}_comparison.png",
        dpi=150
    )
    plt.close(fig)

# 6. 피험자별 상체/무릎 분포 그래프
for metric in ["trunk_angle","knee_angle","knee_offset"]:
    feature = metric+"_mean"
    fig,ax = plt.subplots(figsize=(9,4))

    for label in ["GB","PB","LLK"]:
        vals = df[df.label==label].groupby(
            "subject"
        )[feature].mean()
        ax.plot(
            vals.index,vals.values,
            marker="o",label=label
        )

    ax.set_title(metric+" by subject")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RUN/f"{metric}_by_subject.png",
        dpi=150
    )
    plt.close(fig)

# 7. 최종 보고서
report = {
    "total_videos":len(videos),
    "usable_videos":len(df),
    "excluded_videos":len(failures),
    "label_counts":dict(Counter(df.label)),
    "subjects":sorted(df.subject.unique().tolist()),
    "interpretation":(
        "Exploratory descriptive analysis only. "
        "2D angles and screen-space offsets depend on camera view. "
        "Trunk tilt is not a direct measurement of lumbar rounding. "
        "Temporal bins are equal-duration, not true repetition phases."
    ),
    "output_dir":str(RUN)
}

(RUN/"report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),
    encoding="utf-8"
)
(RUN/"failures.json").write_text(
    json.dumps(failures,ensure_ascii=False,indent=2),
    encoding="utf-8"
)

print("\n===== 최종 분석 결과 =====",flush=True)
print("전체 영상:",len(videos),flush=True)
print("분석 성공:",len(df),flush=True)
print("제외:",len(failures),flush=True)
print("라벨 분포:",dict(Counter(df.label)),flush=True)

if not rank_df.empty:
    for error in ["PB","LLK"]:
        print("\n",error,"구분 후보 특징 TOP 5",flush=True)
        print(
            rank_df[rank_df.bad_type==error][[
                "feature","n_subjects",
                "sign_consistency","standardized_effect"
            ]].head(5).to_string(index=False),
            flush=True
        )

print("\n✅ 결과 저장:",RUN,flush=True)
