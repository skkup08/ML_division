from pathlib import Path
import json, shutil, time, subprocess, sys, random
from datetime import datetime, timezone
import yaml, numpy as np, torch
from ultralytics import YOLO
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML')
WORK=Path('/content/skku1398_workspace')
DATA=WORK/'prepared/grouped_v1'
TAG=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN=ROOT/'03_training_runs'/('grouped_v1_'+TAG)
RUN.mkdir(parents=True)
(ROOT/'04_exports_and_reports/latest_run.txt').write_text(str(RUN))
shutil.copy2(ROOT/'02_dataset/fitfuel_v2_grouped_v1/audit.json',RUN/'dataset_audit.json')
(RUN/'requirements_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
status={'scope':'public_proxy_regrouped_holdout_not_center','started_utc':TAG,'smoke':'pending','baseline':'pending','evaluation':'pending','mobile':'pending'}
def save_status(): (RUN/'status.json').write_text(json.dumps(status,indent=2))
save_status()
# Smoke uses only a subset of train/validation. Test is never used for smoke.
SMOKE=WORK/('smoke_'+TAG);SMOKE.mkdir()
for sp,n in [('train',60),('val',25)]:
    files=sorted((DATA/sp/'images').glob('*'));random.Random(42).shuffle(files)
    chosen=[]
    for c in range(5):
        opts=[p for p in files if any(int(l.split()[0])==c for l in (DATA/sp/'labels'/(p.stem+'.txt')).read_text().splitlines() if l.strip())]
        if opts and opts[0] not in chosen:chosen.append(opts[0])
    chosen+=(p for p in files if p not in chosen)
    for p in chosen[:n]:
        dst=SMOKE/sp/'images'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        dst=SMOKE/sp/'labels'/(p.stem+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(DATA/sp/'labels'/dst.name,dst)
(SMOKE/'data.yaml').write_text(yaml.safe_dump({'path':str(SMOKE),'train':'train/images','val':'val/images','names':yaml.safe_load((DATA/'data.yaml').read_text())['names']}))
def copy_artifacts(trainer):
    src=Path(trainer.save_dir);dest=RUN/src.name;dest.mkdir(exist_ok=True)
    for p in src.rglob('*'):
        if p.is_file() and p.suffix in ['.pt','.csv','.yaml','.png','.jpg','.json']:
            target=dest/p.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
try:
    status['smoke']='running';save_status()
    smoke=YOLO('yolo11n.pt');smoke.add_callback('on_train_end',copy_artifacts)
    smoke.train(data=str(SMOKE/'data.yaml'),epochs=2,imgsz=320,batch=8,device=0,workers=2,seed=42,project=str(WORK/'runs'/TAG),name='smoke',plots=True,verbose=False)
    status['smoke']='passed';status['baseline']='running';save_status()
    del smoke;torch.cuda.empty_cache()
    model=YOLO('yolo11n.pt')
    model.add_callback('on_model_save',copy_artifacts);model.add_callback('on_train_end',copy_artifacts)
    model.train(data=str(DATA/'data.yaml'),epochs=30,imgsz=640,batch=16,device=0,workers=2,seed=42,patience=12,cache=False,project=str(WORK/'runs'/TAG),name='baseline',plots=True,verbose=False)
    status['baseline']='completed';save_status()
    best=RUN/'baseline/weights/best.pt';assert best.is_file()
    final=YOLO(str(best));status['evaluation']='running';save_status()
    metrics=final.val(data=str(DATA/'data.yaml'),split='test',imgsz=640,batch=8,device=0,plots=True,save_json=True,project=str(RUN),name='test_evaluation',verbose=False)
    ids=list(map(int,metrics.box.ap_class_index))
    result={'scope':status['scope'],'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'class_metrics':{final.names[c]:{'ap50':float(metrics.box.ap50[i]),'ap50_95':float(metrics.box.ap[i]),'precision':float(metrics.box.p[i]),'recall':float(metrics.box.r[i])} for i,c in enumerate(ids)},'test_images':len(list((DATA/'test/images').glob('*'))),'weights_bytes':best.stat().st_size,'center_validation':False}
    (RUN/'public_grouped_test_metrics.json').write_text(json.dumps(result,indent=2))
    samples=sorted((DATA/'test/images').glob('*'))[:10]
    # Warm-up then synchronized end-to-end latency on one fixed held-out photo.
    final.predict(str(samples[0]),imgsz=640,device=0,verbose=False)
    timings=[]
    for _ in range(20):
        torch.cuda.synchronize();t=time.perf_counter();final.predict(str(samples[0]),imgsz=640,device=0,verbose=False);torch.cuda.synchronize();timings.append((time.perf_counter()-t)*1000)
    (RUN/'inference_benchmark.json').write_text(json.dumps({'hardware':torch.cuda.get_device_name(0),'runs':20,'batch':1,'imgsz':640,'end_to_end_ms_mean':float(np.mean(timings)),'end_to_end_ms_median':float(np.median(timings)),'includes':'local image decode, preprocessing, inference and NMS; warm cache','mobile_speed_measured':False},indent=2))
    from skku1398_ml import predictions_to_json
    pred_dir=RUN/'predictions';pred_dir.mkdir()
    for p in samples:
        pred=final.predict(str(p),imgsz=640,device=0,conf=.25,verbose=False)[0]
        shutil.copy2(p,pred_dir/('original_'+p.name));pred.save(filename=str(pred_dir/('predicted_'+p.name)))
        (pred_dir/(p.stem+'.json')).write_text(json.dumps(predictions_to_json(pred),indent=2))
    status['evaluation']='completed';save_status()
    print('TRAIN_AND_EVAL_COMPLETE',str(RUN),json.dumps(result),flush=True)
except Exception as exc:
    status['error']=type(exc).__name__+': '+str(exc);save_status();raise
