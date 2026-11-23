from pathlib import Path
import json,hashlib,warnings,csv,sys
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
p=Path(__file__).parent
def file_path(name):
 aliases={'features.npz':'final_ml_features.npz','manifest.csv':'video_manifest_and_splits.csv','model.joblib':'fitness_models.joblib','predictions.csv':'test_predictions.csv'}
 canonical=p/aliases.get(name,name)
 return canonical if canonical.exists() else p/name
metrics=json.loads((p/'metrics.json').read_text());integrity=json.loads((p/'artifact_integrity.json').read_text())
checks={}
for local,remote in [('features.npz','final_ml_features.npz'),('manifest.csv','video_manifest_and_splits.csv'),('metrics.json','metrics.json'),('predictions.csv','test_predictions.csv'),('model.joblib','fitness_models.joblib'),('completion.json','completion.json')]:
 h=hashlib.sha256(file_path(local).read_bytes()).hexdigest();assert h==integrity[remote]['sha256'],remote;checks[remote]=h
m=pd.read_csv(file_path('manifest.csv'),keep_default_na=False);z=np.load(file_path('features.npz'),allow_pickle=False);n=len(m)
assert n==5014 and z['temporal'].shape==(n,1232) and z['image_summary'].shape==(n,256)
o=z['frame_offsets'];assert len(o)==n+1 and o[0]==0 and o[-1]==len(z['frame_features']) and (np.diff(o)>0).all()
assert np.isfinite(z['image_summary']).all() and np.isfinite(z['frame_features']).all()
valid=np.isfinite(z['temporal']).all(1);missing=np.isnan(z['temporal']).all(1);assert (valid|missing).all();assert np.array_equal(valid,m.temporal_available.to_numpy(bool))
assert m.sha256.is_unique and m.groupby(['source','group']).split.nunique().max()==1
assert set(m.split)=={'train','validation','test'}
assert not m.exercise.str.contains('deadlift').any()
assert (missing[m.source.eq('zenodo_squat_image_reference')]).all()
with warnings.catch_warnings(record=True) as w:
 warnings.simplefilter('always');b=joblib.load(file_path('model.joblib'))
 load_warnings=sorted(set(str(a.message) for a in w))
for obj in [b['action'],b['action_frame'],*b['quality'].values(),*b['quality_image'].values()]:
 if obj is not None:obj.set_params(n_jobs=1)
print('MODEL_LOADED',flush=True)
assert len(b['quality'])==7 and len(b['quality_image'])==7 and len(b['action'].classes_)==23
rows=[];details={};reproduced={}
def evaluate(y,pred):
 return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'classes':classification_report(y,pred,labels=['good','bad'],output_dict=True,zero_division=0),'confusion_labels':['good','bad'],'confusion_matrix':confusion_matrix(y,pred,labels=['good','bad']).tolist()}
# Verify both individual quality models against saved metrics, no threshold/model tuning.
for mode,models,key,arr in [('temporal',b['quality'],'quality_',z['temporal']),('image_summary',b['quality_image'],'quality_image_',z['image_summary'])]:
 for ex,model in models.items():
  mask=m.split.eq('test') & m.exercise.eq(ex) & m.quality.isin(['good','bad'])
  if mode=='temporal':mask &= valid
  ids=np.flatnonzero(mask);pred=model.predict(arr[ids]);r=evaluate(m.quality.to_numpy()[ids],pred)
  old=metrics[key+str(ex)+'_test'];assert r['n']==old['n'] and abs(r['accuracy']-old['accuracy'])<1e-12
  reproduced[key+str(ex)+'_test']=True
  print('QUALITY_VERIFIED',mode,ex,flush=True)
  # Subject-disjoint video reference for each model, separating all static image data.
  keep=m.iloc[ids].source.eq('mendeley').to_numpy();vv=evaluate(m.quality.to_numpy()[ids][keep],pred[keep]);details[str(ex)+'_'+mode+'_mendeley']=vv
  rows.append({'exercise':str(ex),'model':mode,'evaluation':'Mendeley subject-disjoint known-exercise video test','test_videos':vv['n'],'accuracy':vv['accuracy'],'balanced_accuracy':vv['balanced_accuracy'],'good_precision':vv['classes']['good']['precision'],'good_recall':vv['classes']['good']['recall'],'good_f1':vv['classes']['good']['f1-score'],'good_support':vv['classes']['good']['support'],'bad_precision':vv['classes']['bad']['precision'],'bad_recall':vv['classes']['bad']['recall'],'bad_f1':vv['classes']['bad']['f1-score'],'bad_support':vv['classes']['bad']['support']})
# Reproduce saved automatic exercise -> quality raw cascade for every held-out sample.
preds=pd.read_csv(file_path('predictions.csv'),keep_default_na=False);test=np.flatnonzero(m.split.eq('test'));assert len(preds)==len(test)
assert preds.name.tolist()==m.iloc[test].name.tolist() and preds.source.tolist()==m.iloc[test].source.tolist()
exercise=[];quality=[]
for j,i in enumerate(test):
 if j%200==0:print('CASCADE',j,len(test),flush=True)
 if valid[i]:
  a=b['action'];pr=a.predict_proba(z['temporal'][i:i+1])[0]
 else:
  a=b['action_frame'];pr=a.predict_proba(z['frame_features'][o[i]:o[i+1]]).mean(0)
 ex=str(a.classes_[pr.argmax()]);exercise.append(ex)
 if ex in b['quality'] and valid[i]:model=b['quality'][ex];x=z['temporal'][i:i+1]
 elif ex in b['quality_image']:model=b['quality_image'][ex];x=z['image_summary'][i:i+1]
 else:quality.append('');continue
 quality.append(str(model.predict(x)[0]))
assert exercise==preds.predicted_exercise.tolist() and quality==preds.predicted_quality.tolist()
md=preds[preds.source.eq('mendeley')];joint=float(((md.true_exercise==md.predicted_exercise)&(md.true_quality==md.predicted_quality)).mean())
cleanup=json.loads((p/'cleanup.json').read_text());assert cleanup['new_temporary_directory_removed'] is True and cleanup['existing_drive_raw_data_deleted'] is False
report={'status':'final_saved_artifacts_verified','verified_utc':pd.Timestamp.now(tz='UTC').isoformat(),'samples':n,'source_counts':m.source.value_counts().to_dict(),'mendeley_subjects':int(m[m.source.eq('mendeley')].group.nunique()),'duplicate_video_or_image_hashes':0,'source_group_split_disjoint':True,'quality_models_reproduced':reproduced,'saved_cascade_predictions_reproduced':len(test),'mendeley_raw_cascade_joint_accuracy':joint,'mendeley_cascade_test_videos':len(md),'model_reload_passed':True,'local_sklearn_version':sklearn.__version__,'load_warnings':load_warnings,'hash_verified':checks,'cleanup_record':cleanup,'quality_details':details,'limitations':['Known-exercise scores do not include exercise recognition mistakes.','Mendeley subjects are split apart, but the dataset was used in earlier experiments; this is not new external validation.','Kaggle has exercise labels only. Zenodo has static image labels and no trusted person/video IDs. Neither supplies person-independent video quality accuracy.','Plank/curl correction CSVs have no person/video IDs; their scores are frame references only.','Joblib was written by sklearn1.9.1 and read locally with sklearn1.8.0; recorded quality metrics and all saved raw cascade predictions exactly reproduced, but inference environment should match training sklearn1.9.1.','Only newly downloaded temporary raw data was removed. Existing user originals and earlier models are preserved.']}
(p/'final_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (p/'quality_scorecard.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'status':report['status'],'saved_test_predictions_verified':len(test),'mendeley_cascade_joint_accuracy':joint,'mendeley_cascade_n':len(md),'scorecard':rows},ensure_ascii=False,indent=2))
