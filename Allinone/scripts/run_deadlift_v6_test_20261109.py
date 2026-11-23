# 데드리프트 V6: Drive에 올린 영상 1개 예측 (재학습 없음)
# 한 셀 전체 실행. 기존 모델과 데이터는 삭제하거나 덮어쓰지 않습니다.
# Python 3.11 MediaPipe 환경 / 모델 저장 버전 환경을 분리합니다.
import os, sys, re, json, shutil, subprocess, hashlib, traceback, uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path("/content/drive/MyDrive/Allinone")
VIDEO_OVERRIDE = ""  # 비워 두면 test_input의 test_deadlift.mp4 또는 .mp4.mp4 자동 선택
MODEL_RELATIVE = "Deadlift_V6_Models/run_20261027_135306/deadlift_v6.joblib"

EXPECTED_MODEL_SHA256 = '4ce507fac6492424677de134a34c1d0e9815ece5ed92f7960a9f712de6e7935a'

EXTRACTOR_CODE = '"""Extract one video\'s exact V4 features for the saved V6 model; no training."""\nimport os\nos.environ["MPLBACKEND"] = "Agg"\nimport sys, json, hashlib\nfrom pathlib import Path\nimport cv2\nimport numpy as np\nimport pandas as pd\nimport mediapipe as mp\n\nLS, RS = 11, 12\n\nLH, RH = 23, 24\n\nLK, RK = 25, 26\n\nLA, RA = 27, 28\n\nLF, RF = 31, 32\n\ndef angle(a, b, c):\n    u = a-b\n    v = c-b\n    den = np.linalg.norm(u)*np.linalg.norm(v)\n    if den < 1e-8:\n        return np.nan\n    return np.degrees(np.arccos(\n        np.clip(np.dot(u,v)/den, -1, 1)\n    ))\n\ndef measure(landmarks):\n    p = np.array(\n        [[x.x,x.y,x.z] for x in landmarks],\n        dtype=np.float32\n    )\n\n    hip = (p[LH]+p[RH])/2\n    shoulder = (p[LS]+p[RS])/2\n    torso = shoulder-hip\n    scale = max(np.linalg.norm(torso[:2]), 0.05)\n\n    # 2D 투영 관절 각도\n    hip_angle = np.mean([\n        angle(p[LS,:2],p[LH,:2],p[LK,:2]),\n        angle(p[RS,:2],p[RH,:2],p[RK,:2])\n    ])\n    knee_angle = np.mean([\n        angle(p[LH,:2],p[LK,:2],p[LA,:2]),\n        angle(p[RH,:2],p[RK,:2],p[RA,:2])\n    ])\n\n    # 상체와 화면 수직축 사이의 2D 각도\n    trunk_angle = np.degrees(\n        np.arctan2(abs(torso[0]), abs(torso[1])+1e-8)\n    )\n\n    # 무릎-발목 수평 차이 (촬영 방향 영향을 받음)\n    knee_offset = np.mean([\n        abs(p[LK,0]-p[LA,0]),\n        abs(p[RK,0]-p[RA,0])\n    ]) / scale\n\n    # 어깨와 엉덩이의 화면 내 수평 차이\n    shoulder_hip_offset = abs(\n        shoulder[0]-hip[0]\n    ) / scale\n\n    # 좌우 무릎 각도의 차이\n    left_knee = angle(\n        p[LH,:2],p[LK,:2],p[LA,:2]\n    )\n    right_knee = angle(\n        p[RH,:2],p[RK,:2],p[RA,:2]\n    )\n\n    return {\n        "trunk_angle": trunk_angle,\n        "hip_angle": hip_angle,\n        "knee_angle": knee_angle,\n        "knee_offset": knee_offset,\n        "shoulder_hip_offset": shoulder_hip_offset,\n        "knee_asymmetry": abs(left_knee-right_knee)\n    }\n\nMETRICS = [\n    "trunk_angle", "hip_angle", "knee_angle",\n    "knee_offset", "shoulder_hip_offset",\n    "knee_asymmetry"\n]\n\ndef extract(path, sample_frames=48):\n    cap = cv2.VideoCapture(str(path))\n    if not cap.isOpened():\n        return None\n\n    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n    if count < 8:\n        cap.release()\n        return None\n\n    idxs = np.unique(np.linspace(\n        0, count-1, min(count,sample_frames)\n    ).astype(int))\n\n    series = []\n    times = []\n\n    try:\n        with mp.solutions.pose.Pose(\n            static_image_mode=True,\n            model_complexity=1,\n            min_detection_confidence=0.5\n        ) as pose:\n            for sample_no, idx in enumerate(idxs, 1):\n                if sample_no == 1 or sample_no % 8 == 0 or sample_no == len(idxs):\n                    print(f"관절 추출: {sample_no}/{len(idxs)} 프레임 검사", flush=True)\n                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))\n                ok, frame = cap.read()\n                if not ok:\n                    continue\n\n                h,w = frame.shape[:2]\n                if max(h,w)>960:\n                    s = 960/max(h,w)\n                    frame = cv2.resize(\n                        frame,(max(1,int(w*s)),max(1,int(h*s)))\n                    )\n\n                rgb = cv2.cvtColor(\n                    frame,cv2.COLOR_BGR2RGB\n                )\n                result = pose.process(\n                    np.ascontiguousarray(rgb)\n                )\n\n                if not result.pose_landmarks:\n                    continue\n\n                lm = result.pose_landmarks.landmark\n                required = [\n                    LS,RS,LH,RH,LK,RK,LA,RA\n                ]\n\n                if np.mean([\n                    lm[j].visibility for j in required\n                ]) < 0.35:\n                    continue\n\n                series.append(measure(lm))\n                times.append(idx/max(count-1,1))\n    finally:\n        cap.release()\n\n    if len(series)<8:\n        return None\n\n    frame_df = pd.DataFrame(series)\n    frame_df["time"] = times\n    frame_df = frame_df.dropna()\n\n    if len(frame_df)<8:\n        return None\n\n    summary = {\n        "valid_frames": len(frame_df),\n        "valid_ratio": len(frame_df)/len(idxs)\n    }\n\n    # 전체 운동의 분포 요약\n    for metric in METRICS:\n        vals = frame_df[metric].to_numpy()\n        summary[metric+"_mean"] = float(np.mean(vals))\n        summary[metric+"_std"] = float(np.std(vals))\n        summary[metric+"_p10"] = float(np.percentile(vals,10))\n        summary[metric+"_p90"] = float(np.percentile(vals,90))\n        summary[metric+"_range"] = float(\n            np.percentile(vals,90)-np.percentile(vals,10)\n        )\n\n        # 시간 구간별 평균 (실제 반복 구간은 아님)\n        for j in range(5):\n            lo, hi = j/5, (j+1)/5\n            block = frame_df[\n                (frame_df.time>=lo) &\n                (frame_df.time<hi if j<4 else frame_df.time<=hi)\n            ]\n            summary[f"{metric}_bin{j+1}"] = (\n                float(block[metric].mean())\n                if len(block) else np.nan\n            )\n\n    return summary\n\nORIGINAL_EXTRACTOR_SHA256 = \'0fa15de17438ca772973bb118a048c47e493149ffc04c0d0b86d9d98ee355d51\'\n\n\ndef main():\n    video, feature_file, info_file = map(Path, sys.argv[1:4])\n    if not video.is_file():\n        raise FileNotFoundError(str(video))\n    if mp.__version__ != "0.10.21" or not hasattr(mp, "solutions"):\n        raise RuntimeError("MediaPipe 0.10.21 / solutions 환경이 필요합니다.")\n    cap = cv2.VideoCapture(str(video))\n    try:\n        if not cap.isOpened():\n            raise RuntimeError("영상을 열지 못했습니다.")\n        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n        fps = float(cap.get(cv2.CAP_PROP_FPS))\n        info = {\n            "total_frames": n, "fps": fps,\n            "duration_seconds": n / fps if fps > 0 else None,\n            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),\n            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),\n            "sampling_requested": min(48, max(0, n)),\n            "mediapipe": mp.__version__, "numpy": np.__version__,\n            "opencv": cv2.__version__, "pandas": pd.__version__,\n            "python": sys.version,\n            "original_extractor_sha256": ORIGINAL_EXTRACTOR_SHA256,\n        }\n    finally:\n        cap.release()\n    print("영상 정보:", n, "프레임 /", round(fps, 3), "fps", flush=True)\n    summary = extract(video, sample_frames=48)\n    if summary is None:\n        info.update(status="POSE_EXTRACTION_FAILED")\n        info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")\n        raise RuntimeError("유효 관절 프레임이 8개 미만입니다. 자세 GOOD/BAD를 강제로 판정하지 않습니다.")\n    feature_columns = [c for c in summary if c not in ("valid_frames", "valid_ratio")]\n    if len(feature_columns) != 60:\n        raise RuntimeError(f"특징 개수가 다릅니다: {len(feature_columns)} (예상 60)")\n    row = pd.DataFrame([summary])\n    if row[feature_columns].isna().all(axis=None):\n        raise RuntimeError("사용 가능한 특징이 없습니다.")\n    feature_file.parent.mkdir(parents=True, exist_ok=True)\n    row.to_csv(feature_file, index=False, encoding="utf-8-sig")\n    info.update(status="FEATURES_EXTRACTED", valid_frames=int(summary["valid_frames"]),\n                valid_ratio=float(summary["valid_ratio"]), feature_count=len(feature_columns),\n                missing_feature_values=int(row[feature_columns].isna().sum().sum()))\n    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")\n    print("관절 추출 완료:", info["valid_frames"], "유효 프레임 / 특징", len(feature_columns), "개", flush=True)\n\n\nif __name__ == "__main__":\n    main()\n'

PREDICTOR_CODE = '"""Load the user\'s existing trusted V6 model and predict one extracted video row."""\nimport os\nos.environ["MPLBACKEND"] = "Agg"\nimport sys, json, hashlib, warnings\nfrom pathlib import Path\nimport joblib\nimport numpy as np\nimport pandas as pd\nimport sklearn\nfrom sklearn.exceptions import InconsistentVersionWarning\n\n\ndef sha256(path):\n    h = hashlib.sha256()\n    with open(path, "rb") as f:\n        for block in iter(lambda: f.read(1024 * 1024), b""):\n            h.update(block)\n    return h.hexdigest()\n\n\ndef main():\n    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))\n    model_file = Path(cfg["model_path"])\n    features_file = Path(cfg["feature_path"])\n    out = Path(cfg["result_path"])\n    expected = cfg["training_environment"]\n    for key, installed in [("sklearn", sklearn.__version__), ("numpy", np.__version__), ("pandas", pd.__version__)]:\n        if installed != expected[key]:\n            raise RuntimeError(f"{key} 버전 불일치: {installed} / 학습 기록 {expected[key]}")\n    before = sha256(model_file)\n    if before != cfg["model_sha256_before"]:\n        raise RuntimeError("준비 중 모델 파일이 변경됐습니다. 실행을 중단합니다.")\n    # Only the user\'s own previously trained model is loaded; do not use arbitrary downloaded joblib files.\n    with warnings.catch_warnings():\n        warnings.simplefilter("error", InconsistentVersionWarning)\n        bundle = joblib.load(model_file)\n    if bundle.get("version") != "V6" or bundle.get("exercise") != "deadlift":\n        raise RuntimeError("지정한 파일이 데드리프트 V6 모델이 아닙니다.")\n    names = list(bundle["features"])\n    if len(names) != len(set(names)):\n        raise RuntimeError("모델 특징 이름이 중복됐습니다.")\n    frame = pd.read_csv(features_file)\n    if len(frame) != 1:\n        raise RuntimeError("영상 한 개의 특징 1행이 필요합니다.")\n    missing = sorted(set(names) - set(frame.columns))\n    if missing:\n        raise RuntimeError("모델에 필요한 특징 누락: " + str(missing))\n    X = frame[names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)\n    if X.shape[1] != getattr(bundle["model"], "n_features_in_", X.shape[1]):\n        raise RuntimeError("모델 입력 차원과 특징 차원이 다릅니다.")\n    if np.isnan(X).all():\n        raise RuntimeError("모든 특징이 결측이라 예측을 중단합니다.")\n    model = bundle["model"]\n    prediction = str(model.predict(X)[0])\n    if prediction not in {"GOOD", "BAD"}:\n        raise RuntimeError("예상하지 못한 예측 라벨: " + prediction)\n    scores = None\n    if hasattr(model, "predict_proba"):\n        scores = {str(c): float(p) for c, p in zip(model.classes_, model.predict_proba(X)[0])}\n    pose = json.loads(Path(cfg["pose_info_path"]).read_text(encoding="utf-8"))\n    known = None\n    manifest = model_file.parent / "nested_cv_predictions.csv"\n    if manifest.is_file():\n        digest_table = pd.read_csv(manifest, usecols=["digest"], dtype=str)\n        known = bool(digest_table["digest"].eq(cfg["video_sha256"]).any())\n    result = {\n        "status": "PREDICTED", "exercise": "deadlift", "version": "V6",\n        "input_video": cfg["video_path"], "video_sha256": cfg["video_sha256"],\n        "model_path": str(model_file), "model_sha256": before,\n        "algorithm": bundle.get("selected_model"),\n        "prediction": prediction,\n        "display_result": "GOOD (모델 기준 정상 예측)" if prediction == "GOOD" else "BAD (모델 기준 오류 의심)",\n        "raw_model_scores": scores,\n        "scores_note": "Uncalibrated model outputs; not posture correctness or safety probabilities.",\n        "pose": pose, "feature_count": len(names),\n        "missing_feature_values": int(np.isnan(X).sum()),\n        "byte_identical_training_video": known,\n        "unseen_content_guaranteed": False,\n        "ground_truth": None, "accuracy": None,\n        "accuracy_note": "A single external example without an independently verified label does not measure accuracy.",\n        "retrained": False, "model_unchanged": sha256(model_file) == before,\n        "inference_environment": {\n            "python": sys.version, "sklearn": sklearn.__version__,\n            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,\n        },\n        "training_environment": expected,\n        "feature_contract": "Original V4 48-frame sampling and 60-column aggregation, ordered by the V6 bundle.",\n        "warning": "This prediction is not an expert evaluation of exercise safety; both GOOD and BAD may be wrong.",\n    }\n    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")\n    pd.DataFrame([{\n        "video": cfg["video_path"], "prediction": prediction,\n        "valid_frames": pose["valid_frames"], "feature_count": len(names),\n        "model_path": str(model_file),\n    }]).to_csv(out.parent / "prediction.csv", index=False, encoding="utf-8-sig")\n    lines = [\n        "데드리프트 V6 외부 영상 동작 확인", "",\n        "영상: " + cfg["video_path"], "판정: " + result["display_result"],\n        "유효 관절 프레임: " + str(pose["valid_frames"]) + "/" + str(pose["sampling_requested"]),\n        "입력 특징 수: " + str(len(names)), "모델: " + str(model_file),\n        "재학습: 없음", "모델 파일 변경: 없음", "",\n        "정답 라벨이 없는 영상 한 개이므로 정확도를 계산하지 않았습니다.",\n        "GOOD/BAD는 모델의 예측이며, 전문가가 확인한 자세 정답이나 안전성 판정이 아닙니다.",\n    ]\n    (out.parent / "result_KO.txt").write_text("\\n".join(lines), encoding="utf-8")\n    print("예측 완료:", result["display_result"], flush=True)\n\n\nif __name__ == "__main__":\n    main()\n'



def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def choose_video(folder):
    if VIDEO_OVERRIDE:
        path = Path(VIDEO_OVERRIDE)
        if not path.is_file():
            raise FileNotFoundError("지정한 영상이 없습니다: " + str(path))
        return path
    if not folder.is_dir():
        raise FileNotFoundError("테스트 영상 폴더가 없습니다: " + str(folder))
    media = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
    preferred = [p for p in media if re.fullmatch(r"test_deadlift(?:\.mp4)+", p.name, re.IGNORECASE)]
    if len(preferred) == 1:
        return preferred[0]
    if len(media) == 1:
        return media[0]
    if not media:
        raise FileNotFoundError("test_input 폴더에 영상이 없습니다. 업로드 완료와 계정을 확인하세요.")
    raise RuntimeError("후보 영상이 여러 개입니다. 코드 위의 VIDEO_OVERRIDE에 하나의 경로를 지정하세요:\n" + "\n".join(map(str, media)))


def main():
    from google.colab import drive
    drive.mount("/content/drive")
    if not ROOT.is_dir():
        raise FileNotFoundError("Allinone 폴더가 없습니다. 기존 모델을 저장한 Google 계정인지 확인하세요.")
    video = choose_video(ROOT / "test_input")
    model_path = ROOT / MODEL_RELATIVE
    if not model_path.is_file():
        candidates = list((ROOT / "Deadlift_V6_Models").rglob("deadlift_v6.joblib"))
        raise FileNotFoundError("확인된 V6 모델 경로가 없습니다. 임의의 다른 모델로 바꾸지 않습니다.\n찾은 파일: " + str(candidates))
    env_file = model_path.parent / "environment.json"
    if not env_file.is_file():
        raise FileNotFoundError("모델의 environment.json이 없습니다. 버전을 추측해 실행하지 않습니다.")
    training_env = json.loads(env_file.read_text(encoding="utf-8"))
    for name in ("sklearn", "numpy", "pandas"):
        if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", str(training_env.get(name, ""))):
            raise RuntimeError("학습 환경 버전을 확인할 수 없습니다: " + name)
    python_version = str(training_env.get("python", "")).split()[0]
    if not re.fullmatch(r"\d+\.\d+\.\d+", python_version):
        raise RuntimeError("학습 환경의 Python 버전을 확인할 수 없습니다.")
    if digest_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("모델 파일이 확인 당시와 달라졌습니다. 재학습하거나 임의 모델을 선택하지 않고 중단합니다.")

    token = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    result_dir = ROOT / "External_Test_Results" / ("deadlift_v6_" + token)
    result_dir.mkdir(parents=True, exist_ok=False)
    work = Path("/content/allinone_v6_external_test")
    work.mkdir(parents=True, exist_ok=True)
    local_run = work / token
    local_run.mkdir()
    log_file = result_dir / "execution.log"
    status_file = result_dir / "status.json"
    print("영상:", video, flush=True)
    print("사용 모델:", model_path, flush=True)
    print("결과 저장:", result_dir, flush=True)

    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               UV_CACHE_DIR=str(work / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(work / "python"),
               UV_LINK_MODE="copy", UV_NO_PROGRESS="1")

    def run(command, stage, custom_env=None):
        print("\n" + stage, flush=True)
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n" + stage + "\n")
            p = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                 bufsize=1, env=custom_env or env)
            try:
                for line in p.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = p.wait()
            except BaseException:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                raise
            finally:
                if p.stdout:
                    p.stdout.close()
            if code != 0:
                raise RuntimeError(f"{stage}: 종료 코드 {code}. 상세 로그: {log_file}")

    try:
        status_file.write_text(json.dumps({"status": "RUNNING", "video": str(video)}, ensure_ascii=False), encoding="utf-8")
        local_video = local_run / video.name
        shutil.copy2(video, local_video)
        local_model = local_run / "deadlift_v6.joblib"
        shutil.copy2(model_path, local_model)
        video_hash = digest_file(local_video)
        if digest_file(local_model) != EXPECTED_MODEL_SHA256:
            raise RuntimeError("모델 복사 검증 실패")

        pose_script = local_run / "extract_v4_compatible.py"
        predict_script = local_run / "predict_saved_v6.py"
        pose_script.write_text(EXTRACTOR_CODE, encoding="utf-8")
        predict_script.write_text(PREDICTOR_CODE, encoding="utf-8")
        shutil.copy2(pose_script, result_dir / pose_script.name)
        shutil.copy2(predict_script, result_dir / predict_script.name)

        # Install the tiny environment manager without modifying notebook ML packages.
        tools = work / "tools"
        tools.mkdir(exist_ok=True)
        tool_env = env.copy()
        tool_env["PYTHONPATH"] = str(tools)
        if not (tools / "uv").is_dir():
            run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "--target", tools, "uv"],
                "[1/5] 실행 환경 도구 준비")
        uv = [sys.executable, "-m", "uv"]

        pose_env = work / "pose311"
        predict_env = work / ("model_py" + python_version.replace(".", "_"))
        pose_py = pose_env / "bin/python"
        predict_py = predict_env / "bin/python"
        if not pose_py.exists():
            run(uv + ["venv", "--python", "3.11", pose_env], "[2/5] 관절 추출용 Python 준비", tool_env)
        if not predict_py.exists():
            # Prefer Colab's interpreter when it is the recorded exact version.
            requested = sys.executable if sys.version.split()[0] == python_version else python_version
            run(uv + ["venv", "--python", requested, predict_env], "[2/5] 저장 모델용 Python 준비", tool_env)

        pose_packages = [
            "mediapipe==0.10.21", "numpy==1.26.4", "opencv-contrib-python==4.10.0.84",
            "protobuf==4.25.8", "matplotlib==3.9.2", "pandas==2.2.3",
            "jax==0.4.35", "jaxlib==0.4.35", "scipy==1.14.1",
        ]
        prediction_packages = [
            "scikit-learn==" + training_env["sklearn"], "numpy==" + training_env["numpy"],
            "pandas==" + training_env["pandas"], "joblib==1.4.2", "scipy==1.14.1",
        ]
        run(uv + ["pip", "install", "--python", pose_py, *pose_packages], "[2/5] 관절 추출 패키지 확인", tool_env)
        run(uv + ["pip", "install", "--python", predict_py, *prediction_packages], "[2/5] 학습 기록과 모델 패키지 맞추기", tool_env)
        (result_dir / "requirements_pose.txt").write_text("\n".join(pose_packages) + "\n", encoding="utf-8")
        (result_dir / "requirements_prediction.txt").write_text("\n".join(prediction_packages) + "\n", encoding="utf-8")

        feature_path = result_dir / "video_features.csv"
        pose_info = result_dir / "pose_info.json"
        run([pose_py, "-u", pose_script, local_video, feature_path, pose_info], "[3/5] 새 영상 관절 특징 추출 (재학습 아님)")

        config = {
            "model_path": str(model_path), "model_sha256_before": EXPECTED_MODEL_SHA256,
            "video_path": str(video), "video_sha256": video_hash,
            "feature_path": str(feature_path), "pose_info_path": str(pose_info),
            "result_path": str(result_dir / "result.json"), "training_environment": training_env,
        }
        config_path = local_run / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        run([predict_py, "-u", predict_script, config_path], "[4/5] 기존 V6 모델로 판정")

        result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "PREDICTED":
            raise RuntimeError("예측 결과 파일이 완성되지 않았습니다.")
        if not result.get("model_unchanged") or digest_file(model_path) != EXPECTED_MODEL_SHA256:
            raise RuntimeError("모델 파일 변경 여부 확인 실패")
        status_file.write_text(json.dumps({"status": "COMPLETED", "result": str(result_dir / "result.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[5/5] ===== 데드리프트 영상 테스트 결과 =====", flush=True)
        print("영상:", video.name, flush=True)
        print("최종 결과:", result["display_result"], flush=True)
        print("유효 관절 프레임:", str(result["pose"]["valid_frames"]) + "/" + str(result["pose"]["sampling_requested"]), flush=True)
        print("입력 특징:", result["feature_count"], "개", flush=True)
        print("알고리즘:", result["algorithm"], flush=True)
        if result["raw_model_scores"] is not None:
            print("원시 모델 점수(정확도나 안전 확률 아님):", json.dumps(result["raw_model_scores"], ensure_ascii=False), flush=True)
        if result["byte_identical_training_video"] is True:
            print("주의: 학습 자료와 동일한 파일 해시입니다. 새 외부 검증이 아닙니다.", flush=True)
        else:
            print("영상 1개의 동작 확인입니다. 정답 라벨이 없어서 정확도는 계산하지 않습니다.", flush=True)
        print("재학습 없음 / 기존 모델 변경 없음", flush=True)
        print("결과 폴더:", result_dir, flush=True)
        print("이 최종 결과 부분을 복사해서 보내주세요.", flush=True)
        return result
    except BaseException as exc:
        info = {"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                "error": str(exc), "error_type": type(exc).__name__, "log_file": str(log_file)}
        status_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("\n판정을 완료하지 못했습니다. 결과를 GOOD/BAD로 만들어 내지 않았습니다.", flush=True)
        print("오류 로그:", log_file, flush=True)
        raise


if __name__ == "__main__":
    RESULT = main()
