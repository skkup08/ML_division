"""Subject-disjoint model search. Legacy test is evaluated once after CV selection.
No raw data downloads; input is existing saved ML features. All samples retained.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,csv,hashlib,sys,warnings
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix

def geometry(z):
 o=z['frame_offsets'];allf=z['frame_features'];rows=[]
 joints={j:i for i,j in enumerate([11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32])}
 triples=[(11,13,15),(12,14,16),(13,11,23),(14,12,24),(11,23,25),(12,24,26),(23,25,27),(24,26,28)]
 for i in range(len(o)-1):
  f=allf[o[i]:o[i+1]];pts=f[:,:48].reshape(-1,16,3);vis=f[:,48:]>0
  cols=[]
  for dims in [2,3]:
   for a,b,c in triples:
    ia,ib,ic=[joints[j] for j in [a,b,c]];u=pts[:,ia,:dims]-pts[:,ib,:dims];v=pts[:,ic,:dims]-pts[:,ib,:dims]
    den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);x=np.arccos(np.clip((u*v).sum(1)/np.maximum(den,1e-8),-1,1))/np.pi
    x[~(vis[:,ia]&vis[:,ib]&vis[:,ic])|(den<1e-8)]=np.nan;cols.append(x)
  x=np.array(cols).T
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   vec=np.concatenate([np.nanmean(x,0),np.nanstd(x,0),np.nanpercentile(x,10,axis=0),np.nanpercentile(x,50,axis=0),np.nanpercentile(x,90,axis=0),np.nanmax(x,0)-np.nanmin(x,0)])
  rows.append(vec)
 return np.array(rows,dtype=np.float32)

def main(data,out):
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 manifest=data/'video_manifest_and_splits.csv'
 if not manifest.exists():manifest=data/'manifest.csv'
 features=data/'final_ml_features.npz'
 if not features.exists():features=data/'features.npz'
 m=pd.read_csv(manifest,keep_default_na=False);z=np.load(features,allow_pickle=False)
 md=m.source.eq('mendeley').to_numpy();ids=np.flatnonzero(md);m=m.iloc[ids].reset_index(drop=True)
 assert m.groupby('group').split.nunique().max()==1
 g=geometry(z)[ids];s=z['image_summary'][ids];t=z['temporal'][ids]
 angleidx=np.array([block*56+j for block in range(6) for j in range(48,56)]+[336+frame*56+j for frame in range(16) for j in range(48,56)])
 feats={'summary':s,'geometry':g,'combined':np.concatenate([s,g,t[:,angleidx],np.isfinite(t).all(1)[:,None]],axis=1)}
 models={'lr01':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000,random_state=42)), 'lr1':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42)), 'svm1':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),SVC(C=1,class_weight='balanced')), 'svm10':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),SVC(C=10,class_weight='balanced')), 'trees1':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,class_weight='balanced',random_state=42,n_jobs=1)), 'trees3':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=1))}
 tasks=['ACTION_7']+sorted(m.exercise.unique());search=[];bundle={};report={};testpreds=[]
 def metric(y,p):return {'n':len(y),'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'classes':classification_report(y,p,output_dict=True,zero_division=0),'confusion_labels':sorted(set(y)|set(p)),'confusion_matrix':confusion_matrix(y,p,labels=sorted(set(y)|set(p))).tolist()}
 for task in tasks:
  eligible=np.ones(len(m),bool) if task=='ACTION_7' else m.exercise.eq(task).to_numpy()
  tr=np.flatnonzero(eligible&m.split.eq('train'));va=np.flatnonzero(eligible&m.split.eq('validation'));te=np.flatnonzero(eligible&m.split.eq('test'))
  y=m.exercise.to_numpy() if task=='ACTION_7' else m.quality.to_numpy();groups=m.group.to_numpy();cv=GroupKFold(n_splits=5);rank=[]
  for fn,X in feats.items():
   for name,est in models.items():
    pred=np.empty(len(tr),object);foldscores=[]
    for a,b in cv.split(X[tr],y[tr],groups[tr]):
     assert not set(groups[tr[a]])&set(groups[tr[b]])
     model=clone(est).fit(X[tr[a]],y[tr[a]]);pred[b]=model.predict(X[tr[b]]);foldscores.append(balanced_accuracy_score(y[tr[b]],pred[b]))
    score=float(np.mean(foldscores));row={'task':task,'features':fn,'model':name,'train_group_cv_balanced_accuracy':score,'train_group_cv_accuracy':float(accuracy_score(y[tr],pred))};search.append(row);rank.append((score,fn,name))
  score,fn,name=sorted(rank,key=lambda r:(-r[0],r[1],r[2]))[0]
  fitted=clone(models[name]).fit(feats[fn][tr],y[tr]);bundle[task]={'features':fn,'model':fitted}
  validation=metric(y[va],fitted.predict(feats[fn][va]));test=metric(y[te],fitted.predict(feats[fn][te]))
  report[task]={'selection_source':'train subject-group CV only','selected_features':fn,'selected_model':name,'cv_balanced_accuracy':score,'validation':validation,'legacy_test':test,'legacy_test_previously_used':True}
  print(task,fn,name,'CV',round(score,4),'VAL',round(validation['accuracy'],4),'LEGACY_TEST',round(test['accuracy'],4),flush=True)
  pd.DataFrame(search).to_csv(out/'train_cv_search.csv',index=False)
 # Recompute full raw cascade, mandatory decision coverage100%.
 te=np.flatnonzero(m.split.eq('test'));a=bundle['ACTION_7'];action=a['model'].predict(feats[a['features']][te]);quality=[]
 for i,ex in zip(te,action):
  q=bundle[str(ex)];quality.append(str(q['model'].predict(feats[q['features']][i:i+1])[0]))
 correct=(action==m.exercise.to_numpy()[te])&(np.array(quality)==m.quality.to_numpy()[te]);joint=float(correct.mean())
 pd.DataFrame({'source':'mendeley','name':m.name.to_numpy()[te],'subject':m.group.to_numpy()[te],'true_exercise':m.exercise.to_numpy()[te],'predicted_exercise':action,'actual_quality':m.quality.to_numpy()[te],'predicted_quality':quality,'joint_correct':correct}).to_csv(out/'legacy_test_predictions.csv',index=False)
 met=all(report[t]['legacy_test']['accuracy']>=.95 for t in tasks) and joint>=.95
 result={'target':.95,'target_met_on_legacy_test':met,'tasks':report,'legacy_cascade_joint_accuracy':joint,'test_n':len(te),'coverage':1.,'raw_downloads_created':False,'selection_policy':'5-fold subject-group CV within original train partition only; no tuning on validation or legacy test. Always predicts.','limitations':['Original legacy test has been reported previously and is not new independent external validation.','Only26 subjects. Point accuracy does not establish95% population accuracy.','Action recognizer here supports the seven quality exercises only; existing23-class model is preserved.','All7 quality models evaluated on every known-exercise heldout video, including fallback cases; not directly equivalent to old temporal-only124-video subset.'],'environment':{'python':sys.version,'sklearn':sklearn.__version__},'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [features,manifest]}}
 joblib.dump({'version':'quality95_candidate1','models':bundle,'feature_spec':'summary256; frame-derived 16 angles*6stats; combined adds176 temporal angles+availability','scope':'seven quality exercises only','feature_version':'fitness_consolidated_v4_world1232_image64_summary256'},out/'candidate_models.joblib',compress=3)
 (out/'candidate_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('CANDIDATE_COMPLETE',joint,'TARGET_MET',met,flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
