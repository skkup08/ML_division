"""Extract one video's exact V4 features for the saved V6 model; no training."""
import os
os.environ["MPLBACKEND"] = "Agg"
import sys, json, hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

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
            for sample_no, idx in enumerate(idxs, 1):
                if sample_no == 1 or sample_no % 8 == 0 or sample_no == len(idxs):
                    print(f"관절 추출: {sample_no}/{len(idxs)} 프레임 검사", flush=True)
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

ORIGINAL_EXTRACTOR_SHA256 = '0fa15de17438ca772973bb118a048c47e493149ffc04c0d0b86d9d98ee355d51'


def main():
    video, feature_file, info_file = map(Path, sys.argv[1:4])
    if not video.is_file():
        raise FileNotFoundError(str(video))
    if mp.__version__ != "0.10.21" or not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe 0.10.21 / solutions 환경이 필요합니다.")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError("영상을 열지 못했습니다.")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        info = {
            "total_frames": n, "fps": fps,
            "duration_seconds": n / fps if fps > 0 else None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "sampling_requested": min(48, max(0, n)),
            "mediapipe": mp.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "pandas": pd.__version__,
            "python": sys.version,
            "original_extractor_sha256": ORIGINAL_EXTRACTOR_SHA256,
        }
    finally:
        cap.release()
    print("영상 정보:", n, "프레임 /", round(fps, 3), "fps", flush=True)
    summary = extract(video, sample_frames=48)
    if summary is None:
        info.update(status="POSE_EXTRACTION_FAILED")
        info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("유효 관절 프레임이 8개 미만입니다. 자세 GOOD/BAD를 강제로 판정하지 않습니다.")
    feature_columns = [c for c in summary if c not in ("valid_frames", "valid_ratio")]
    if len(feature_columns) != 60:
        raise RuntimeError(f"특징 개수가 다릅니다: {len(feature_columns)} (예상 60)")
    row = pd.DataFrame([summary])
    if row[feature_columns].isna().all(axis=None):
        raise RuntimeError("사용 가능한 특징이 없습니다.")
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(feature_file, index=False, encoding="utf-8-sig")
    info.update(status="FEATURES_EXTRACTED", valid_frames=int(summary["valid_frames"]),
                valid_ratio=float(summary["valid_ratio"]), feature_count=len(feature_columns),
                missing_feature_values=int(row[feature_columns].isna().sum().sum()))
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("관절 추출 완료:", info["valid_frames"], "유효 프레임 / 특징", len(feature_columns), "개", flush=True)


if __name__ == "__main__":
    main()
