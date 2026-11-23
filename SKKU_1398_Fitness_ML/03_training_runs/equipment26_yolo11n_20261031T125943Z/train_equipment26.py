"""Fine-tune the 26-equipment detector; persist resumable checkpoints and honest evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,shutil,zipfile,subprocess,sys,traceback,gc
import torch,yaml,ultralytics
from ultralytics import YOLO,settings
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML')
ARCHIVE=ROOT/'02_dataset/SKKU1398_equipment_v1/yolo/equipment26_20261031.zip'
EXPECTED_MANIFEST_SHA='522bafb35f7f827d2dda261f3f6739a02bfd79fa2d0a12a089dabee3ea426d3f'
BASE_RUN=ROOT/'03_training_runs/equipment24_yolo11n_20261031T122141Z'
WORK=Path('/content/equipment26_training');DATA=WORK/'equipment26_20261031'
RESUME_RUN=os.environ.get('EQUIPMENT26_RESUME_RUN','').strip()
assert torch.cuda.is_available(),'T4 GPU is required.'
settings.update({k:False for k in ('wandb','comet','mlflow','clearml','neptune','hub') if k in settings})
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN=Path(RESUME_RUN) if RESUME_RUN else ROOT/'03_training_runs'/('equipment26_yolo11n_'+tag)
if RESUME_RUN:assert RUN.parent==ROOT/'03_training_runs' and (RUN/'fit/weights/last.pt').is_file()
else:RUN.mkdir(parents=True,exist_ok=False)
status={'state':'preparing','classes':26,'epochs_requested':60,'epochs_completed':0,'run':str(RUN),'started_utc':tag,'dataset_manifest_sha256':EXPECTED_MANIFEST_SHA,'center_validation':False,'deployment_validated':False,'megamax_evaluation':'not_available_no_holdout_images'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def write_json(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def save_status(**v):
 status.update(v);status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(RUN/'status.json',status)
save_status();print('RUN_DIRECTORY',str(RUN),flush=True)
try:
 WORK.mkdir(exist_ok=True);local=WORK/ARCHIVE.name;shutil.copy2(ARCHIVE,local);save_status(dataset_archive_sha256=sha(local))
 with zipfile.ZipFile(local) as z:
  assert z.testzip() is None
  for i in z.infolist():assert (WORK/i.filename).resolve().is_relative_to(WORK.resolve())
  z.extractall(WORK)
 assert sha(DATA/'manifest.json')==EXPECTED_MANIFEST_SHA
 subprocess.run([sys.executable,str(DATA/'verify_dataset.py')],check=True)
 summary=json.loads((DATA/'summary.json').read_text());mapping=json.loads((DATA/'class_mapping.json').read_text())
 assert summary['images']==3369 and summary['splits']=={'train':2969,'val':200,'test':200}
 assert len(mapping)==26 and mapping[24]['center_id']==19 and mapping[25]['center_id']==24
 config=yaml.safe_load((DATA/'data.yaml').read_text());config['path']=str(DATA);cfg=WORK/'data_train.yaml';cfg.write_text(yaml.safe_dump(config,sort_keys=False))
 for name in ['summary.json','class_mapping.json','README_ko.txt']:shutil.copy2(DATA/name,RUN/name)
 shutil.copy2(cfg,RUN/'data_train.yaml');shutil.copy2(Path(__file__),RUN/'train_equipment26.py')
 write_json(RUN/'environment.json',{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0)})
 (RUN/'requirements_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 if RESUME_RUN:initial=RUN/'fit/weights/last.pt'
 else:
  source=BASE_RUN/'fit/weights/best.pt';assert source.is_file()
  initial=RUN/'initial_24class_best.pt';shutil.copy2(source,initial)
  write_json(RUN/'initialization.json',{'source_run':str(BASE_RUN),'source_checkpoint':str(source),'copied_checkpoint_sha256':sha(initial),'method':'transfer learned shared weights to 26-class detector; classification output resized','base_status':json.loads((BASE_RUN/'status.json').read_text())})
 model=YOLO(str(initial))
 def on_start(trainer):
  assert len(trainer.model.names)==26
  save_status(state='training',train_images=2969,val_images=200,test_images=200,gpu=torch.cuda.get_device_name(0),model_classes=trainer.model.names)
  print('TRAINING_26_CLASSES_CONFIRMED',trainer.model.names,flush=True)
 def progress(trainer):
  metrics={k:float(v) for k,v in trainer.metrics.items()}
  save_status(state='training',epochs_completed=int(trainer.epoch+1),latest_validation_metrics=metrics,best_checkpoint=str(trainer.best),last_checkpoint=str(trainer.last))
  print('CHECKPOINT_SAVED',trainer.epoch+1,json.dumps(metrics),flush=True)
 model.add_callback('on_train_start',on_start);model.add_callback('on_model_save',progress)
 if RESUME_RUN:model.train(resume=True,device=0,workers=2)
 else:model.train(data=str(cfg),epochs=60,patience=15,imgsz=640,batch=16,device=0,workers=2,seed=1398,deterministic=True,cache=False,fraction=1.0,pretrained=True,val=True,save=True,save_period=10,project=str(RUN),name='fit',exist_ok=False,plots=True,verbose=False,close_mosaic=10)
 best=RUN/'fit/weights/best.pt';assert best.is_file()
 save_status(state='evaluating_test',training_completed=True,best_checkpoint_sha256=sha(best))
 del model;gc.collect();torch.cuda.empty_cache()
 final=YOLO(str(best));metrics=final.val(data=str(cfg),split='test',imgsz=640,batch=16,device=0,workers=2,plots=True,save_json=True,project=str(RUN),name='test_evaluation',exist_ok=True,verbose=False)
 index={int(c):i for i,c in enumerate(metrics.box.ap_class_index)};rows=[]
 for m in mapping:
  c=m['yolo_id'];n=summary['class_image_counts'][m['name']]['test'];r={**m,'test_images':n,'evaluated':n>0}
  if n==0:r.update(precision=None,recall=None,ap50=None,ap50_95=None,reason='No held-out images; no performance claim is available.')
  elif c in index:
   i=index[c];r.update(precision=float(metrics.box.p[i]),recall=float(metrics.box.r[i]),ap50=float(metrics.box.ap50[i]),ap50_95=float(metrics.box.ap[i]))
  else:r.update(precision=0.0,recall=0.0,ap50=0.0,ap50_95=0.0)
  rows.append(r)
 report={'images':200,'model_classes':26,'evaluated_classes':25,'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'per_class':rows,'center_validation':False,'deployment_validated':False,'limitations':summary['limitations']}
 write_json(RUN/'test_metrics.json',report)
 pred=RUN/'prediction_examples';pred.mkdir(exist_ok=True)
 paths=sorted((DATA/'images/test').glob('added26*'))+sorted((DATA/'images/test').glob('*.jpg'))[:8]
 for p in dict.fromkeys(paths):
  result=final.predict(str(p),imgsz=640,conf=.25,device=0,verbose=False)[0];result.save(filename=str(pred/p.name))
  write_json(pred/(p.stem+'.json'),{'image_size':[result.orig_shape[1],result.orig_shape[0]],'detections':[{'class_id':int(b.cls.item()),'center_id':mapping[int(b.cls.item())]['center_id'],'class_name':final.names[int(b.cls.item())],'confidence':float(b.conf.item()),'bbox_xyxy':b.xyxy[0].tolist()} for b in result.boxes]})
 save_status(state='completed',training_completed=True,evaluation_completed=True,test_metrics={k:report[k] for k in ['precision','recall','mAP50','mAP50_95']},best_checkpoint=str(best))
 export=ROOT/'04_exports_and_reports'/RUN.name;export.mkdir(exist_ok=True)
 for p in [best,RUN/'class_mapping.json',RUN/'test_metrics.json',RUN/'status.json',RUN/'README_ko.txt']:shutil.copy2(p,export/p.name)
 write_json(export/'artifact_integrity.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in export.iterdir() if p.is_file() and p.name!='artifact_integrity.json'})
 print('TRAINING_AND_TEST_COMPLETE',json.dumps(status,ensure_ascii=False),flush=True)
except BaseException as e:
 save_status(state='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());raise
