import os,json,hashlib,platform,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import sklearn,joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix

def run(source,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 df=pd.read_csv(source)
 assert set(df.label.unique()) <= {'GB','LLK','PB'}
 assert df.subject.notna().all() and df.digest.notna().all()
 duplicate_count=int(df.digest.duplicated().sum())
 if df.groupby('digest').label.nunique().max()>1: raise ValueError('Conflicting duplicate labels')
 if df.groupby('digest').subject.nunique().max()>1: raise ValueError('Duplicate videos across subjects')
 df=df.drop_duplicates('digest').reset_index(drop=True)
 features=[c for c in df.columns if c not in ['digest','file','label','subject','valid_frames','valid_ratio']]
 X=df[features].replace([np.inf,-np.inf],np.nan).to_numpy();y=np.where(df.label.eq('GB'),'GOOD','BAD');g=df.subject.to_numpy()
 assert len(set(g))>=4
 candidates={'logistic_C01':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000,random_state=42)), 'svm_C1':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),SVC(C=1,class_weight='balanced',probability=False,random_state=42)), 'extra_trees':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesClassifier(n_estimators=150,min_samples_leaf=2,max_features=.7,class_weight='balanced',random_state=42,n_jobs=2))}
 cv=LeaveOneGroupOut();pred=np.empty(len(y),dtype=object);folds=[];inner_rows=[]
 def choose(indices,outer):
  scores=[]
  for name,est in candidates.items():
   pp=np.empty(len(indices),dtype=object)
   for tr,va in cv.split(X[indices],y[indices],g[indices]):
    assert not set(g[indices][tr])&set(g[indices][va])
    m=clone(est).fit(X[indices][tr],y[indices][tr]);pp[va]=m.predict(X[indices][va])
   score=float(balanced_accuracy_score(y[indices],pp));scores.append((score,name));inner_rows.append({'outer_subject':outer,'candidate':name,'inner_balanced_accuracy':score})
  return sorted(scores,key=lambda z:(-z[0],z[1]))[0][1]
 for tr,te in cv.split(X,y,g):
  subject=str(g[te][0]);name=choose(tr,subject)
  assert not set(g[tr])&set(g[te])
  pred[te]=clone(candidates[name]).fit(X[tr],y[tr]).predict(X[te])
  row={'held_out_subject':subject,'selected_model':name,'n':len(te),'accuracy':float(accuracy_score(y[te],pred[te]))};folds.append(row);print(json.dumps(row),flush=True)
 labels=['GOOD','BAD']
 metrics={'evaluation':'nested leave-one-subject-out; inner subject CV selects model; fixed default decision boundary; outer subject excluded from preprocessing and selection','n_videos':len(df),'n_subjects':len(set(g)),'duplicates_removed':duplicate_count,'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'classification_report':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),'confusion_labels':labels,'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'limitations':['Same dataset was explored by earlier V3-V5 experiments; this is not new independent external validation.','Only seven subjects; S02 has GOOD examples only.','Cached V4 pose features reused; raw-video feature extraction was not retrained or independently audited.','GOOD=GB; BAD=LLK/PB only; not all unsafe techniques are represented.','Final model fits all data; reported metrics belong to outer held-out fold models, not training accuracy.']}
 name=choose(np.arange(len(y)),'FINAL_ALL_DATA');model=clone(candidates[name]).fit(X,y)
 joblib.dump({'version':'V6','exercise':'deadlift','model':model,'features':features,'label_mapping':{'GB':'GOOD','LLK':'BAD','PB':'BAD'},'selected_model':name,'input_contract':'One row per full video; exact V4 metric aggregation columns; no IDs/labels as features'},out/'deadlift_v6.joblib')
 df.assign(actual=y,predicted=pred,outer_fold=g).to_csv(out/'nested_cv_predictions.csv',index=False)
 pd.DataFrame(folds).to_csv(out/'subject_metrics.csv',index=False);pd.DataFrame(inner_rows).to_csv(out/'model_selection.csv',index=False)
 manifest=[]
 for subject in sorted(set(g)):
  for i,row in df.iterrows():manifest.append({'outer_subject':subject,'digest':row.digest,'subject':row.subject,'file':row.file,'label':y[i],'role':'test' if row.subject==subject else 'development_inner_cv'})
 pd.DataFrame(manifest).to_csv(out/'split_manifest.csv',index=False)
 (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
 (out/'environment.json').write_text(json.dumps({'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'selected_final_model':name},indent=2))
 report=f"# 데드리프트 V6 검증 결과\n\n영상 {len(df)}개, 피험자 {len(set(g))}명. GOOD=GB, BAD=LLK/PB.\n\n피험자 단위 중첩 교차검증 정확도: {metrics['accuracy']:.2%}\n균형 정확도: {metrics['balanced_accuracy']:.2%}\n\n"
 for label in labels:
  r=metrics['classification_report'][label];report+=f"{label}: precision={r['precision']:.4f}, recall={r['recall']:.4f}, F1={r['f1-score']:.4f}, n={r['support']}\n\n"
 report+='혼동행렬 행=실제, 열=예측, 순서 GOOD/BAD: '+str(metrics['confusion_matrix'])+'\n\n모델 선택과 결측값 처리·스케일링은 매 외부 평가 피험자를 제외하고 수행했습니다. 별도 임계값 최적화 없이 기본 경계를 사용합니다.\n\n기존 V3–V5에서도 분석한 데이터이므로 새로운 독립 외부 검증이 아닙니다. 새로운 피험자/촬영 환경 검증이 필요합니다. 캐시된 V4 영상 특징을 재사용하며, 원본 영상의 특징 추출은 재학습하지 않았습니다.\n\n최종 배포 모델은 전체 자료로 재학습했습니다. 위 정확도는 최종 모델의 학습 정확도가 아니라 외부 폴드 예측 결과입니다.\n'
 (out/'REPORT_KO.md').write_text(report)
 # Reload and validate serialization against the fitted model.
 loaded=joblib.load(out/'deadlift_v6.joblib');assert np.array_equal(model.predict(X),loaded['model'].predict(X))
 (out/'verification.json').write_text(json.dumps({'reload_prediction_match':True,'outer_predictions':len(pred),'source_exists':Path(source).exists(),'utc':datetime.now(timezone.utc).isoformat()},indent=2))
 print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
 return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.source,a.out)
