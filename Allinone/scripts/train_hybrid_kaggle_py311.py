
import os
os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from PIL import Image

JOINTS = [
    (11,13,15), (12,14,16),
    (13,11,23), (14,12,24),
    (11,23,25), (12,24,26),
    (23,25,27), (24,26,28),
    (25,27,31), (26,28,32)
]

REQUIRED = set(i for t in JOINTS for i in t)

VIDEO_EXT = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
}
IMAGE_EXT = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

def joint_angle(a, b, c):
    u = a - b
    v = c - b
    d = np.linalg.norm(u) * np.linalg.norm(v)
    if d < 1e-8:
        return 0.0
    return float(
        np.arccos(np.clip(np.dot(u,v)/d,-1,1))/np.pi
    )

def get_pose_features(lm):
    pts = np.array(
        [[p.x,p.y,p.z] for p in lm],
        dtype=np.float32
    )

    angles = np.array([
        joint_angle(pts[a],pts[b],pts[c])
        for a,b,c in JOINTS
    ], dtype=np.float32)

    hip = (pts[23]+pts[24])/2
    shoulder = (pts[11]+pts[12])/2
    scale = max(np.linalg.norm(shoulder-hip),0.05)

    keypoints = [
        11,12,13,14,15,16,
        23,24,25,26,27,28,29,30,31,32
    ]

    coords = ((pts[keypoints]-hip)/scale).flatten()
    return np.concatenate([angles,coords])

def iter_frames(path, max_frames=32):
    path = Path(path)
    ext = path.suffix.lower()

    if ext in IMAGE_EXT:
        with Image.open(path) as img:
            yield np.array(img.convert("RGB"))
        return

    if ext == ".gif":
        with Image.open(path) as img:
            n = getattr(img, "n_frames", 1)
            indices = np.unique(
                np.linspace(
                    0,n-1,min(max_frames,n)
                ).astype(int)
            )
            for idx in indices:
                img.seek(int(idx))
                yield np.array(img.convert("RGB"))
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return

    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return

        indices = np.unique(
            np.linspace(
                0,n-1,min(max_frames,n)
            ).astype(int)
        )

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
            ok, frame = cap.read()

            if ok:
                yield cv2.cvtColor(
                    frame,cv2.COLOR_BGR2RGB
                )
    finally:
        cap.release()

def extract(path):
    path = Path(path)
    vectors = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        for rgb in iter_frames(path,32):
            h,w = rgb.shape[:2]

            if max(h,w)>960:
                ratio=960/max(h,w)
                rgb=cv2.resize(rgb,(
                    max(1,int(w*ratio)),
                    max(1,int(h*ratio))
                ))

            res=pose.process(
                np.ascontiguousarray(rgb)
            )

            if not res.pose_landmarks:
                continue

            lm=res.pose_landmarks.landmark
            visibility=np.mean([
                lm[i].visibility for i in REQUIRED
            ])

            if visibility<0.35:
                continue

            vectors.append(get_pose_features(lm))

    # 기존 영상 모델은 최소 5개 프레임으로 학습됨
    minimum = (
        1 if path.suffix.lower() in IMAGE_EXT
        else 5
    )

    if len(vectors)<minimum:
        return None

    X=np.stack(vectors)

    movement=(
        np.abs(np.diff(X,axis=0))
        if len(X)>=2
        else np.zeros_like(X[:1])
    )

    result=np.concatenate([
        X.mean(axis=0),
        X.std(axis=0),
        np.percentile(X,10,axis=0),
        np.percentile(X,50,axis=0),
        np.percentile(X,90,axis=0),
        movement.mean(axis=0)
    ]).astype(np.float32)

    if not np.all(np.isfinite(result)):
        return None

    return result
