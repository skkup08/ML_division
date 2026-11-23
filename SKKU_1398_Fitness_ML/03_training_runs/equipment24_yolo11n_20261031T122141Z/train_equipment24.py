"""Train the verified 24-class equipment detector and persist checkpoints to Drive."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, traceback, zipfile
import torch, yaml, ultralytics
from ultralytics import YOLO, settings

ROOT = Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML')
ARCHIVE = ROOT / '02_dataset/SKKU1398_equipment_v1/yolo/equipment24_expanded_min100_20261031.zip'
EXPECTED_SHA = '587491a3e2d6ea98310f9c7de489a12b3a139c94de74b765499cd77e92a24f41'
WORK = Path('/content/equipment24_training')
DATA = WORK / 'equipment24_expanded_20261031'
RESUME_RUN = os.environ.get('EQUIPMENT24_RESUME_RUN', '').strip()
assert torch.cuda.is_available(), 'GPU runtime required; select T4 GPU in Colab and rerun.'
settings.update({k: False for k in ('wandb', 'comet', 'mlflow', 'clearml', 'neptune', 'hub') if k in settings})
tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN = Path(RESUME_RUN) if RESUME_RUN else ROOT / '03_training_runs' / ('equipment24_yolo11n_' + tag)
if RESUME_RUN:
    assert RUN.parent == ROOT / '03_training_runs' and (RUN / 'fit/weights/last.pt').is_file()
else:
    RUN.mkdir(parents=True, exist_ok=False)
status = {'state': 'preparing', 'run': str(RUN), 'model': 'yolo11n.pt', 'classes': 24,
          'epochs_requested': 100, 'epochs_completed': 0, 'dataset_sha256': EXPECTED_SHA,
          'test_used_for_training': False, 'center_validation': False, 'started_utc': tag}
def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)
def save_status(**updates):
    status.update(updates)
    status['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(RUN / 'status.json', status)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
save_status()
print('RUN_DIRECTORY', RUN, flush=True)
try:
    WORK.mkdir(exist_ok=True)
    local_zip = WORK / ARCHIVE.name
    shutil.copy2(ARCHIVE, local_zip)
    assert sha(local_zip) == EXPECTED_SHA, 'Dataset archive SHA256 mismatch'
    with zipfile.ZipFile(local_zip) as z:
        assert z.testzip() is None
        for item in z.infolist():
            target = (WORK / item.filename).resolve()
            assert target.is_relative_to(WORK.resolve()), 'Unsafe archive member'
        z.extractall(WORK)
    subprocess.run([sys.executable, str(DATA / 'verify_dataset.py')], check=True)
    summary = json.loads((DATA / 'summary.json').read_text())
    assert summary['images'] == 3321 and summary['splits'] == {'train': 2928, 'val': 197, 'test': 196}
    assert summary['all_classes_meet_target']
    mapping = json.loads((DATA / 'class_mapping.json').read_text())
    assert len(mapping) == 24
    config = yaml.safe_load((DATA / 'data.yaml').read_text())
    config['path'] = str(DATA)
    config_path = WORK / 'data_train.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    for name in ['summary.json', 'class_mapping.json', 'README_ko.txt']:
        shutil.copy2(DATA / name, RUN / name)
    shutil.copy2(config_path, RUN / 'data_train.yaml')
    if '__file__' in globals(): shutil.copy2(Path(__file__), RUN / 'train_equipment24.py')
    (RUN / 'requirements_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    write_json(RUN / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
               'ultralytics': ultralytics.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory})
    def checkpoint_progress(trainer):
        vals = {k: float(v) for k, v in trainer.metrics.items()}
        save_status(state='training', epochs_completed=int(trainer.epoch + 1),
                    latest_validation_metrics=vals, best_fitness=float(trainer.best_fitness),
                    last_checkpoint=str(trainer.last), best_checkpoint=str(trainer.best))
        print('CHECKPOINT_SAVED', trainer.epoch + 1, json.dumps(vals), flush=True)
    model = YOLO(str(RUN / 'fit/weights/last.pt') if RESUME_RUN else 'yolo11n.pt')
    model.add_callback('on_model_save', checkpoint_progress)
    save_status(state='training', gpu=torch.cuda.get_device_name(0))
    if RESUME_RUN:
        model.train(resume=True, device=0, workers=2)
    else:
        model.train(data=str(config_path), epochs=100, patience=20, imgsz=640, batch=16,
                    device=0, workers=2, seed=1398, deterministic=True, cache=False,
                    fraction=1.0, pretrained=True, val=True, save=True, save_period=10,
                    project=str(RUN), name='fit', exist_ok=False, plots=True, verbose=False,
                    close_mosaic=10)
    best = RUN / 'fit/weights/best.pt'
    assert best.is_file() and best.stat().st_size > 0
    save_status(state='evaluating_test', best_checkpoint_sha256=sha(best))
    final = YOLO(str(best))
    metrics = final.val(data=str(config_path), split='test', imgsz=640, batch=16, device=0,
                        workers=2, plots=True, save_json=True, project=str(RUN),
                        name='test_evaluation', exist_ok=True, verbose=False)
    ids = list(map(int, metrics.box.ap_class_index))
    per_class = []
    for i, c in enumerate(ids):
        row = dict(mapping[c])
        row.update(test_images=summary['class_image_counts'][final.names[c]]['test'],
                   precision=float(metrics.box.p[i]), recall=float(metrics.box.r[i]),
                   ap50=float(metrics.box.ap50[i]), ap50_95=float(metrics.box.ap[i]))
        per_class.append(row)
    report = {'images': 196, 'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
              'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
              'per_class': per_class, 'center_validation': False,
              'limitations': ['Some classes have only 1–3 test images; their metrics are unstable.',
                             'Public/product images do not establish real gym performance.',
                             'Web image license metadata includes unknown entries.']}
    write_json(RUN / 'test_metrics.json', report)
    preds = RUN / 'predictions'; preds.mkdir(exist_ok=True)
    for p in sorted((DATA / 'images/test').glob('*.jpg'))[:12]:
        result = final.predict(str(p), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        result.save(filename=str(preds / p.name))
        write_json(preds / (p.stem + '.json'), {
            'image_width': result.orig_shape[1], 'image_height': result.orig_shape[0],
            'detections': [{'class_id': int(b.cls.item()),
                            'center_id': mapping[int(b.cls.item())]['center_id'],
                            'class_name': result.names[int(b.cls.item())],
                            'confidence': float(b.conf.item()),
                            'bbox_xyxy_pixels': b.xyxy[0].tolist()} for b in result.boxes]})
    save_status(state='completed', test_metrics={k: report[k] for k in ['precision','recall','mAP50','mAP50_95']},
                best_checkpoint=str(best), training_completed=True, evaluation_completed=True)
    export = ROOT / '04_exports_and_reports' / RUN.name
    export.mkdir(exist_ok=True)
    for p in [best, RUN / 'class_mapping.json', RUN / 'test_metrics.json', RUN / 'status.json']:
        shutil.copy2(p, export / p.name)
    write_json(export / 'artifact_integrity.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in export.iterdir() if p.is_file() and p.name != 'artifact_integrity.json'})
    print('TRAINING_AND_TEST_COMPLETE', json.dumps(status, ensure_ascii=False), flush=True)
except BaseException as e:
    save_status(state='interrupted' if isinstance(e, KeyboardInterrupt) else 'failed',
                error=type(e).__name__ + ': ' + str(e), traceback=traceback.format_exc())
    raise
