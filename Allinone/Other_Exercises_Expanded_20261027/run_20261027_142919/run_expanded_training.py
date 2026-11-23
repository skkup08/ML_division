import os
os.environ['MPLBACKEND']='Agg'
import sys,json,tempfile,shutil,traceback,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import fitness_expanded_runtime as f
ROOT=Path('/content/drive/MyDrive/Allinone')
BASE=ROOT/'Other_Exercises_Expanded_20261027'
OUT=BASE/('run_'+time.strftime('%Y%m%d_%H%M%S'));OUT.mkdir(parents=True,exist_ok=False)
WORK=Path(tempfile.mkdtemp(prefix='fitness_temp_',dir='/content'))
class Expanded(f.Pipeline):
 def add_image(self,path,rec):
  import mediapipe as mp
  from mediapipe.tasks.python import vision,BaseOptions
  import cv2
  sha=f.sha256(path)
  if sha in self.seen:return
  if not hasattr(self,'image_detector'):
   self.image_detector=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(self.asset)),running_mode=vision.RunningMode.IMAGE,num_poses=2))
  img=cv2.imread(str(path))
  if img is None:raise ValueError('image decode failed')
  h,w=img.shape[:2]
  if max(h,w)>640:img=cv2.resize(img,(max(1,round(w*640/max(h,w))),max(1,round(h*640/max(h,w)))))
  r=self.image_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
  if len(r.pose_landmarks)!=1:raise ValueError('Expected exactly one person')
  pts=np.array([[[p.x,p.y,p.z,p.visibility] for p in r.pose_landmarks[0]]],np.float32)
  xx,ok=f.frame_features(pts)
  if not ok.all():raise ValueError('insufficient visible joints')
  self.seen[sha]=rec;self.records.append({**rec,'sha256':sha,'sampled_frames':1,'usable_frame_ratio':1.,'temporal_available':False,'temporal_exclusion':'static image; no motion inferred'})
  self.features.append(None);self.frames.append(xx);self.summaries.append(f.image_summary(xx))
 def correction_csvs(self):
  super().correction_csvs()
  for ex,data in self.csv_data.items():
   for part,(X,y,h,pts) in data.items():np.savez_compressed(self.out/f'final_ml_{ex}_{part}.npz',X=X,y=y.astype(str),hashes=h)
 def extra_manifest(self):
  import zipfile,re
  mapping={'barbell biceps curl':'bicep_curl','hammer curl':'hammer_curl','bench press':'bench_press','incline bench press':'incline_bench_press','decline bench press':'decline_bench_press','chest fly machine':'chest_fly_machine','squat':'squat','plank':'plank','leg extension':'leg_extension','russian twist':'russian_twist','tricep pushdown':'tricep_pushdown','lat pulldown':'lat_pulldown','hip thrust':'hip_thrust','pull up':'pull_up','push-up':'push_up','shoulder press':'shoulder_press','t bar row':'t_bar_row','leg raises':'leg_raises','lateral raise':'lateral_raise','tricep dips':'tricep_dips'}
  local=ROOT/'new_dataset'/'Kaggle_WorkoutFitness'
  before=len(self.records)
  if local.exists():
   for directory in sorted(local.iterdir()):
    ex=mapping.get(directory.name.lower())
    if not ex or not directory.is_dir():continue
    for path in sorted(directory.rglob('*')):
     if not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.mp4','.avi','.mov','.gif'}:continue
     rec={'source':'kaggle_existing_action_only','name':str(path.relative_to(local)),'exercise':ex,'quality':'','group':'kaggle:file:'+f.sha256(path),'split':'','errors':{}}
     try:
      if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:self.add_image(path,rec)
      elif path.suffix.lower()!='.gif':self.add_video(path,rec)
     except Exception as e:self.failed.append({**rec,'reason':str(e)})
     if len(self.records)%100==0:self.progress('extra_action_features')
  self.log('kaggle_existing_action_only','processed',f'{len(self.records)-before} usable. Quality unknown. File groups only; subject-independent generalization unverified. Existing originals preserved.')
  before=len(self.records)
  try:
   meta=self.dl.json('https://zenodo.org/api/records/17558630')
   files=meta['files'];entry=next(x for x in files if x['key'].lower().endswith('.zip'))
   url=entry['links'].get('content') or entry['links']['self']
   arc=self.dl.get(url,self.work/'zenodo.zip',entry.get('size',0))
   f.dump(self.out/'zenodo_provenance.json',{'record':'https://zenodo.org/records/17558630','license':meta.get('metadata',{}).get('license'),'file':entry['key'],'sha256':f.sha256(arc),'note':'Image-level reference only; unknown subject/video identity. Not independent subject validation.'})
   with zipfile.ZipFile(arc) as z:
    for member in z.infolist():
     path=Path(member.filename)
     if path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'} or '__MACOSX' in member.filename:continue
     names=[re.sub('[^a-z]','',part.lower()) for part in path.parts[:-1]]
     label='good' if 'good' in names else ('bad' if any(p in {'badback','badheel'} for p in names) else '')
     if not label:continue
     temp=self.work/('current_image'+path.suffix)
     try:
      with z.open(member) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
      rec={'source':'zenodo_squat_image_reference','name':member.filename,'exercise':'squat','quality':label,'group':'zenodo:file:'+f.sha256(temp),'split':'','errors':{}}
      self.add_image(temp,rec)
     except Exception as e:self.failed.append({'source':'zenodo_squat_image_reference','name':member.filename,'reason':str(e)})
     finally:temp.unlink(missing_ok=True)
     if len(self.records)%100==0:self.progress('zenodo_features')
   arc.unlink(missing_ok=True)
   self.log('zenodo_squat_image_reference','processed',f'{len(self.records)-before} usable static images. Not video/subject independent; cannot treat as temporal training.')
  except Exception as e:self.log('zenodo_squat_image_reference','failed',e)
  self.save_ml_features()
  self.save_failures()
CFG={'OUT':str(OUT),'WORK':str(WORK),'MAX_DOWNLOAD_GB':16,'MAX_VIDEOS_PER_CLASS':0,'MAX_FRAMES':96,'SAMPLE_FPS':6.,'N_TREES':300,'USE_MENDELEY':True,'USE_CORRECTION_CSV':True,'USE_UCF101':False,'REQUIRE_ALL_MENDELEY_7':True,'EXTRA_MANIFEST':'','EXISTING_VIDEO_DIR':'','MENDELEY_ZIP_URL':'','DELETE_TEMP_DATA':True}
f.dump(OUT/'config.json',CFG)
pipe=Expanded(CFG)
try:
 support=pipe.run();pipe.save_ml_features()
 notes={'status':'complete','supported':support,'limitations':['Mendeley subjects disjoint across splits but dataset used in prior experiments; no new external validation.','Kaggle and Zenodo have no trustworthy person/video IDs. Their heldout scores are image/file references, not person-independent accuracy.','Kaggle labels exercise identity only, never GOOD/BAD.','Zenodo contains static squat poses; no temporal feature is fabricated.','Existing originals preserved. Only new temporary downloads are deleted.','Deadlift/romanian deadlift excluded.'],'sources':['https://data.mendeley.com/datasets/kgbb3yn47p/3','https://github.com/NgoQuocBao1010/Exercise-Correction','https://zenodo.org/records/17558630','Existing Kaggle_WorkoutFitness on Drive']}
 f.dump(OUT/'completion.json',notes)
 # Korean concise per exercise report, preserve scope caveats.
 lines=['# 데드리프트 제외 운동 학습 결과','', '## 학습/평가',json.dumps(pipe.eval,ensure_ascii=False,indent=2),'','## 한계',*notes['limitations']]
 (OUT/'REPORT_KO.md').write_text('\n'.join(lines))
 for name in ['fitness_expanded_runtime.py','run_expanded_training.py']:shutil.copy2(BASE/name,OUT/name)
 for path in OUT.glob('*.joblib'):
  value=f.joblib.load(path);print('RELOAD_OK',path.name,flush=True)
 integrity={p.name:{'size':p.stat().st_size,'sha256':f.sha256(p)} for p in OUT.iterdir() if p.is_file()}
 f.dump(OUT/'artifact_integrity.json',integrity)
 print('ALL_OTHER_EXERCISES_COMPLETE',OUT,flush=True)
except BaseException:
 pipe.save_ml_features();pipe.save_failures();f.dump(OUT/'failure.json',{'traceback':traceback.format_exc()});raise
finally:
 if hasattr(pipe,'image_detector'):pipe.image_detector.close()
 shutil.rmtree(WORK,ignore_errors=True)
 f.dump(OUT/'cleanup.json',{'new_temporary_directory_removed':not WORK.exists(),'existing_drive_raw_data_deleted':False})
 print('TEMPORARY_RAW_DATA_REMOVED',flush=True)
