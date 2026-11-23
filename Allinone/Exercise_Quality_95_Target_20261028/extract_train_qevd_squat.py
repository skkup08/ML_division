import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,tempfile,shutil,struct,zlib,hashlib,importlib.util,traceback,time
import requests,numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
BASE=Path('/content/drive/MyDrive/Allinone/Exercise_Quality_95_Target_20261028');OLD=BASE.parent/'Other_Exercises_Expanded_20261027'/'run_20261027_142919';OUT=BASE/'qevd_squat_01';OUT.mkdir(exist_ok=True)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=module('pose_runtime',OLD/'fitness_expanded_runtime.py');feature_code=module('candidate_feature_code',BASE/'candidate_01'/'train_candidate.py')
plan=json.loads((BASE/'qevd_squat_extraction_plan.json').read_text());work=Path(tempfile.mkdtemp(prefix='quality95_raw_',dir='/content'));session=requests.Session()
if (OUT/'candidate_report.json').exists():
 print('CANDIDATE_ALREADY_FROZEN; do not repeat final test',flush=True);sys.exit(0)
records=[];frames=[];temporal=[];summaries=[];failed=[]
if (OUT/'feature_records.json').exists() and (OUT/'ml_features.npz').exists():
 z=np.load(OUT/'ml_features.npz',allow_pickle=False)
 records=json.loads(str(z['records_json'])) if 'records_json' in z.files else json.loads((OUT/'feature_records.json').read_text())
 o=z['frame_offsets'];n=min(len(records),len(z['temporal']),len(z['image_summary']),len(o)-1)
 records=records[:n];temporal=list(z['temporal'][:n]);summaries=list(z['image_summary'][:n]);frames=[z['frame_features'][o[i]:o[i+1]] for i in range(n)]
 if (OUT/'failed.json').exists():failed=json.loads((OUT/'failed.json').read_text())
def save(stage):
 offsets=np.cumsum([0]+[len(x) for x in frames]);tmp=OUT/'ml_features.part.npz'
 if records:
  np.savez_compressed(tmp,temporal=np.array(temporal),image_summary=np.array(summaries),frame_features=np.concatenate(frames),frame_offsets=offsets,records_json=np.array(json.dumps(records,ensure_ascii=False)));tmp.replace(OUT/'ml_features.npz');(OUT/'feature_records.json').write_text(json.dumps(records,ensure_ascii=False))
 (OUT/'failed.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2));(OUT/'progress.json').write_text(json.dumps({'stage':stage,'valid':len(records),'failed':len(failed),'planned':len(plan['records']),'time_utc':pd.Timestamp.now(tz='UTC').isoformat()},indent=2))
def clip(entry):
 suffix=['z01','z02','zip'][entry['disk']];url=f'https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-4/QEVD-FIT-300k-Part-4.{suffix}'
 if entry['compressed']>60*1024**2 or entry['uncompressed']>256*1024**2:raise ValueError('Clip exceeds temporary memory budget')
 for attempt in range(3):
  try:
   with session.get(url,headers={'Range':f"bytes={entry['offset']}-{entry['offset']+entry['compressed']+4095}",'Accept-Encoding':'identity'},stream=True,timeout=(25,120)) as r:
    if r.status_code!=206:raise RuntimeError(f'Expected bounded Range HTTP206; got{r.status_code}')
    raw=bytearray()
    for block in r.iter_content(1024*1024):
     raw.extend(block)
     if len(raw)>64*1024**2:raise ValueError('Range exceeded64MiB')
   if raw[:4]!=b'PK\x03\x04':raise ValueError('Invalid split local ZIP header')
   h=struct.unpack_from('<4s5H3I2H',raw);start=30+h[-2]+h[-1];payload=bytes(raw[start:start+entry['compressed']]);data=zlib.decompress(payload,-15) if entry['method']==8 else payload
   if len(data)!=entry['uncompressed']:raise ValueError('Clip size mismatch')
   if h[6] and zlib.crc32(data)!=h[6]:raise ValueError('Clip CRC mismatch')
   return data
  except Exception:
   if attempt==2:raise
   time.sleep(2*(attempt+1))
try:
 done={x['video_path'] for x in records}|{x['video_path'] for x in failed}
 for k,r in enumerate(plan['records']):
  if r['video_path'] in done:continue
  temp=work/'current.mp4'
  try:
   temp.write_bytes(clip(r['zip_member']));sha=f.sha256(temp)
   points,world,t=f.extract_pose(temp,OLD/'pose_landmarker_full.task',{'MAX_FRAMES':64,'SAMPLE_FPS':8.})
   ff,ok=f.frame_features(points)
   if ok.sum()<12 or ok.mean()<.55:raise ValueError('insufficient visible pose frames')
   try:v=f.temporal_features(world,points[:,:,3],t)
   except ValueError:v=np.full(1232,np.nan,np.float32)
   good=ff[ok];ix=np.linspace(0,len(good)-1,min(16,len(good))).astype(int)
   row={key:value for key,value in r.items() if key!='zip_member'};row.update(sha256=sha,temporal_available=bool(np.isfinite(v).all()),sampled_frames=len(t),usable_frame_ratio=float(ok.mean()))
   records.append(row);frames.append(good[ix]);temporal.append(v);summaries.append(f.image_summary(good))
  except Exception as e:failed.append({'video_path':r['video_path'],'split':r['split'],'quality':r['quality'],'worker_id':r['worker_id'],'error':repr(e)})
  finally:temp.unlink(missing_ok=True)
  if (len(records)+len(failed))%10==0:save('extracting');print('EXTRACTED',len(records),'FAILED',len(failed),flush=True)
 save('features_complete')
 m=pd.DataFrame(records);assert m.groupby('group').split.nunique().max()==1 and m.sha256.is_unique
 z=np.load(OUT/'ml_features.npz',allow_pickle=False);Xg=feature_code.geometry(z);Xs=z['image_summary'];feats={'summary':Xs,'geometry':Xg,'combined':np.concatenate([Xs,Xg],1)}
 models={'svm1':make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),SVC(C=1,class_weight='balanced')),'svm10':make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),SVC(C=10,class_weight='balanced')),'lr01':make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000)),'trees':make_pipeline(SimpleImputer(keep_empty_features=True),ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',n_jobs=1,random_state=42))}
 tr=np.flatnonzero(m.split.eq('train'));va=np.flatnonzero(m.split.eq('validation'));te=np.flatnonzero(m.split.eq('test'));y=m.quality.to_numpy();g=m.group.to_numpy();ranking=[]
 for fn,X in feats.items():
  for name,est in models.items():
   scores=[]
   for a,b in GroupKFold(5).split(X[tr],y[tr],g[tr]):
    pred=clone(est).fit(X[tr[a]],y[tr[a]]).predict(X[tr[b]]);scores.append(balanced_accuracy_score(y[tr[b]],pred))
   ranking.append({'features':fn,'model':name,'train_group_cv_balanced_accuracy':float(np.mean(scores))})
 choice=sorted(ranking,key=lambda r:(-r['train_group_cv_balanced_accuracy'],r['features'],r['model']))[0];X=feats[choice['features']];model=clone(models[choice['model']]).fit(X[tr],y[tr])
 def metric(ix):
  pred=model.predict(X[ix]);return {'valid_n':len(ix),'accuracy':float(accuracy_score(y[ix],pred)),'balanced_accuracy':float(balanced_accuracy_score(y[ix],pred)),'classes':classification_report(y[ix],pred,output_dict=True,zero_division=0),'confusion_labels':['good','bad'],'confusion_matrix':confusion_matrix(y[ix],pred,labels=['good','bad']).tolist()}
 val=metric(va);test=metric(te);planned_test=sum(r['split']=='test' for r in plan['records']);test['planned_n']=planned_test;test['pose_coverage']=len(te)/planned_test;test['strict_accuracy_counting_pose_failures_wrong']=float(np.sum(model.predict(X[te])==y[te]))/planned_test
 result={'status':'candidate2_complete','exercise':'squat_only','selection':choice,'validation':val,'new_subject_test':test,'all7_target_met':False,'squat95_met_strict':test['strict_accuracy_counting_pose_failures_wrong']>=.95,'scope':'Known squat clips only; not global23-class cascade or all7 quality tasks. QEVD no-obvious-issue vs explicit back/depth/insufficient errors.','subject_overlap':False,'no_test_tuning':True,'raw_video_files_kept':False,'original_model_replaced':False}
 joblib.dump({'model':model,'features':choice['features'],'scope':result['scope']},OUT/'qevd_squat_candidate.joblib',compress=3);(OUT/'candidate_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));pd.DataFrame(ranking).to_csv(OUT/'train_cv_search.csv',index=False);m.iloc[te].assign(predicted=model.predict(X[te])).to_csv(OUT/'new_subject_test_predictions.csv',index=False)
 save('candidate2_complete');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
except BaseException:
 save('failed');(OUT/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2));raise
finally:
 shutil.rmtree(work,ignore_errors=True);(OUT/'cleanup.json').write_text(json.dumps({'temporary_raw_directory_removed':not work.exists(),'existing_raw_or_models_deleted':False},indent=2));print('RAW_TEMP_CLEANED',flush=True)
