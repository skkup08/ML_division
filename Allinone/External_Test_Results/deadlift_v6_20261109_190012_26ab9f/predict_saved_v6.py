"""Load the user's existing trusted V6 model and predict one extracted video row."""
import os
os.environ["MPLBACKEND"] = "Agg"
import sys, json, hashlib, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    model_file = Path(cfg["model_path"])
    features_file = Path(cfg["feature_path"])
    out = Path(cfg["result_path"])
    expected = cfg["training_environment"]
    for key, installed in [("sklearn", sklearn.__version__), ("numpy", np.__version__), ("pandas", pd.__version__)]:
        if installed != expected[key]:
            raise RuntimeError(f"{key} 버전 불일치: {installed} / 학습 기록 {expected[key]}")
    before = sha256(model_file)
    if before != cfg["model_sha256_before"]:
        raise RuntimeError("준비 중 모델 파일이 변경됐습니다. 실행을 중단합니다.")
    # Only the user's own previously trained model is loaded; do not use arbitrary downloaded joblib files.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bundle = joblib.load(model_file)
    if bundle.get("version") != "V6" or bundle.get("exercise") != "deadlift":
        raise RuntimeError("지정한 파일이 데드리프트 V6 모델이 아닙니다.")
    names = list(bundle["features"])
    if len(names) != len(set(names)):
        raise RuntimeError("모델 특징 이름이 중복됐습니다.")
    frame = pd.read_csv(features_file)
    if len(frame) != 1:
        raise RuntimeError("영상 한 개의 특징 1행이 필요합니다.")
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise RuntimeError("모델에 필요한 특징 누락: " + str(missing))
    X = frame[names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if X.shape[1] != getattr(bundle["model"], "n_features_in_", X.shape[1]):
        raise RuntimeError("모델 입력 차원과 특징 차원이 다릅니다.")
    if np.isnan(X).all():
        raise RuntimeError("모든 특징이 결측이라 예측을 중단합니다.")
    model = bundle["model"]
    prediction = str(model.predict(X)[0])
    if prediction not in {"GOOD", "BAD"}:
        raise RuntimeError("예상하지 못한 예측 라벨: " + prediction)
    scores = None
    if hasattr(model, "predict_proba"):
        scores = {str(c): float(p) for c, p in zip(model.classes_, model.predict_proba(X)[0])}
    pose = json.loads(Path(cfg["pose_info_path"]).read_text(encoding="utf-8"))
    known = None
    manifest = model_file.parent / "nested_cv_predictions.csv"
    if manifest.is_file():
        digest_table = pd.read_csv(manifest, usecols=["digest"], dtype=str)
        known = bool(digest_table["digest"].eq(cfg["video_sha256"]).any())
    result = {
        "status": "PREDICTED", "exercise": "deadlift", "version": "V6",
        "input_video": cfg["video_path"], "video_sha256": cfg["video_sha256"],
        "model_path": str(model_file), "model_sha256": before,
        "algorithm": bundle.get("selected_model"),
        "prediction": prediction,
        "display_result": "GOOD (모델 기준 정상 예측)" if prediction == "GOOD" else "BAD (모델 기준 오류 의심)",
        "raw_model_scores": scores,
        "scores_note": "Uncalibrated model outputs; not posture correctness or safety probabilities.",
        "pose": pose, "feature_count": len(names),
        "missing_feature_values": int(np.isnan(X).sum()),
        "byte_identical_training_video": known,
        "unseen_content_guaranteed": False,
        "ground_truth": None, "accuracy": None,
        "accuracy_note": "A single external example without an independently verified label does not measure accuracy.",
        "retrained": False, "model_unchanged": sha256(model_file) == before,
        "inference_environment": {
            "python": sys.version, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        },
        "training_environment": expected,
        "feature_contract": "Original V4 48-frame sampling and 60-column aggregation, ordered by the V6 bundle.",
        "warning": "This prediction is not an expert evaluation of exercise safety; both GOOD and BAD may be wrong.",
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{
        "video": cfg["video_path"], "prediction": prediction,
        "valid_frames": pose["valid_frames"], "feature_count": len(names),
        "model_path": str(model_file),
    }]).to_csv(out.parent / "prediction.csv", index=False, encoding="utf-8-sig")
    lines = [
        "데드리프트 V6 외부 영상 동작 확인", "",
        "영상: " + cfg["video_path"], "판정: " + result["display_result"],
        "유효 관절 프레임: " + str(pose["valid_frames"]) + "/" + str(pose["sampling_requested"]),
        "입력 특징 수: " + str(len(names)), "모델: " + str(model_file),
        "재학습: 없음", "모델 파일 변경: 없음", "",
        "정답 라벨이 없는 영상 한 개이므로 정확도를 계산하지 않았습니다.",
        "GOOD/BAD는 모델의 예측이며, 전문가가 확인한 자세 정답이나 안전성 판정이 아닙니다.",
    ]
    (out.parent / "result_KO.txt").write_text("\n".join(lines), encoding="utf-8")
    print("예측 완료:", result["display_result"], flush=True)


if __name__ == "__main__":
    main()
