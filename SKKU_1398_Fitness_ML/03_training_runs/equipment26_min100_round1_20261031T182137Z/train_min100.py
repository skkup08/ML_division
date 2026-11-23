"""Train the verified 26-class min100 dataset; no existing runs are overwritten."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, shutil, subprocess, sys, traceback, gc

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def main():
    import torch, yaml, ultralytics
    from ultralytics import YOLO, settings
    root=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML')
    data=Path('/content/equipment26_min100_v2')
    expected='3faea06d469f471e85cd2dd976a55049cafcb9db5e4d96809e42a923cb754706'
    assert torch.cuda.is_available(),'Colab GPU 런타임이 필요합니다.'
    assert sha(data/'manifest.json')==expected,'Dataset version mismatch'
    subprocess.run([sys.executable,str(data/'verify_dataset.py')],check=True)
    summary=json.loads((data/'summary.json').read_text())
    mapping=json.loads((data/'class_mapping.json').read_text())
    assert summary['images']==3537 and summary['splits']=={'train':3128,'val':204,'test':205}
    names={m['yolo_id']:m['name'] for m in mapping}
    previous='equipment26_yolo11n_20261031T125943Z'
    candidates=[root/'04_exports_and_reports'/previous/'best.pt',root/'03_training_runs'/previous/'fit/weights/best.pt']
    source=next((p for p in candidates if p.is_file()),None)
    assert source is not None,'기존 26종 best.pt를 찾을 수 없습니다.'
    model=YOLO(str(source))
    assert model.names==names, f'Checkpoint class mapping mismatch: {model.names}'
    settings.update({k:False for k in ('wandb','comet','mlflow','clearml','neptune','hub') if k in settings})
    resume=os.environ.get('EQUIPMENT26_MIN100_RESUME_RUN','').strip()
    if resume:
        run=Path(resume)
        assert run.parent==root/'03_training_runs'
        saved=json.loads((run/'status.json').read_text())
        assert saved['dataset_manifest_sha256']==expected and saved['state']!='completed'
        checkpoint=run/'fit/weights/last.pt'
        assert checkpoint.is_file()
        model=YOLO(str(checkpoint))
    else:
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run=root/'03_training_runs'/('equipment26_min100_round1_'+tag)
        run.mkdir(parents=True,exist_ok=False)
    status={'state':'preparing','run':str(run),'dataset_manifest_sha256':expected,'classes':26,'epochs_requested':100,'epochs_completed':0,'train_images':3128,'val_images':204,'test_images':205,'model_retrained':False}
    if resume:status.update(saved)
    def update(**values):
        status.update(values);status['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(run/'status.json',status)
    try:
        cfg=yaml.safe_load((data/'data.yaml').read_text());cfg['path']=str(data)
        cfg_path=data/'data_train.yaml';cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        for name in ['summary.json','class_mapping.json','selection_criteria_ko.txt']:
            shutil.copy2(data/name,run/name)
        shutil.copy2(Path(__file__),run/'train_min100.py')
        shutil.copy2(cfg_path,run/'data_train.yaml')
        if not resume:write_json(run/'initialization.json',{'checkpoint':str(source),'sha256':sha(source),'method':'fine-tune existing 26-class weights on expanded dataset; new optimizer, not resume old dataset'})
        write_json(run/'environment.json',{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0)})
        def started(trainer):
            assert trainer.model.names==names
            update(state='training',gpu=torch.cuda.get_device_name(0))
            print('TRAINING_STARTED_26_CLASSES_MIN100',str(run),flush=True)
        def saved_epoch(trainer):
            update(state='training',epochs_completed=int(trainer.epoch+1),latest_validation_metrics={k:float(v) for k,v in trainer.metrics.items()},best_checkpoint=str(trainer.best),last_checkpoint=str(trainer.last))
            print('EPOCH_CHECKPOINT_SAVED',trainer.epoch+1,flush=True)
        model.add_callback('on_train_start',started)
        model.add_callback('on_model_save',saved_epoch)
        update(state='preparing');print('RUN_DIRECTORY',str(run),flush=True)
        if resume:model.train(resume=True,device=0,workers=2)
        else:model.train(data=str(cfg_path),epochs=100,patience=20,imgsz=640,batch=16,device=0,workers=2,seed=1398,deterministic=True,cache=False,fraction=1.0,val=True,save=True,save_period=-1,project=str(run),name='fit',exist_ok=False,plots=True,verbose=False,close_mosaic=10)
        best=run/'fit/weights/best.pt';assert best.is_file()
        update(state='evaluating_test',training_completed=True,model_retrained=True,best_checkpoint_sha256=sha(best))
        del model;gc.collect();torch.cuda.empty_cache()
        final=YOLO(str(best))
        metrics=final.val(data=str(cfg_path),split='test',imgsz=640,batch=16,device=0,workers=2,plots=True,save_json=False,project=str(run),name='test_evaluation',exist_ok=True,verbose=False)
        indices={int(c):i for i,c in enumerate(metrics.box.ap_class_index)};rows=[]
        for m in mapping:
            c=m['yolo_id'];i=indices.get(c);counts=summary['class_image_counts'][m['name']]
            row={**m,**counts,'small_test_sample':counts['test']<20}
            row.update({k:float(arr[i]) if i is not None else 0.0 for k,arr in [('precision',metrics.box.p),('recall',metrics.box.r),('ap50',metrics.box.ap50),('ap50_95',metrics.box.ap)]})
            rows.append(row)
        report={'images':205,'classes':26,'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'per_class':rows,'field_validated':False,'limitations':[x for x in summary['limitations'] if x!='No retraining performed for this update.']}
        write_json(run/'test_metrics.json',report)
        update(state='completed',evaluation_completed=True,best_checkpoint=str(best),test_metrics={k:report[k] for k in ['precision','recall','mAP50','mAP50_95']})
        print('TRAINING_AND_TEST_COMPLETE',json.dumps(status,ensure_ascii=False),flush=True)
    except BaseException as exc:
        update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
