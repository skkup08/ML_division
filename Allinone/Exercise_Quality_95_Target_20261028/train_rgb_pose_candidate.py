"""Frozen RGB transfer features plus preserved pose. New representation strategy.
Python3.11, CPU only. Download one licensed archive into task-local /content,
decode one video at a time, retain compressed ML embeddings only, finally delete
ALL task temporary media/cache. Never modify old source/model artifacts.
"""
import os,sys
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,hashlib,shutil,tempfile,zipfile,importlib.util,traceback
import numpy as np,pandas as pd,joblib,cv2
from PIL import Image
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE=Path('/content/drive/MyDrive/Allinone/Exercise_Quality_95_Target_20261028')
OLD=BASE.parent/'Other_Exercises_Expanded_20261027'/'run_20261027_142919'
OUT=BASE/'rgb_pose_03'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def main():
    assert sys.version_info[:2]==(3,11),'Use dedicated Python3.11 environment'
    if (OUT/'candidate_report.json').exists():
        print('RGB_CANDIDATE_ALREADY_FROZEN',flush=True);return
    q=BASE/'qevd_squat_01'
    if not (q/'candidate_report.json').exists() or not (q/'cleanup.json').exists():
        print('DEFER_RGB_UNTIL_EXISTING_QEVD_RUN_FINISHES',flush=True);return
    # No global temp cleanup: this task owns only its fresh mkdtemp directory.
    OUT.mkdir(exist_ok=True);work=Path(tempfile.mkdtemp(prefix='quality95_rgb_raw_',dir='/content'))
    os.environ['TORCH_HOME']=str(work/'torch_cache')
    records=[];embeddings=[];failed=[]
    f=module('preserved_pose_runtime',OLD/'fitness_expanded_runtime.py')
    e=module('elastic_metric_code',BASE/'elastic_alignment_02'/'train_elastic_candidate.py')
    source=pd.read_csv(OLD/'video_manifest_and_splits.csv',keep_default_na=False)
    source_ids=np.flatnonzero(source.source.eq('mendeley'));md=source.iloc[source_ids].reset_index(drop=True)
    assert md.groupby('group').split.nunique().max()==1 and md.sha256.is_unique
    def checkpoint(stage):
        if records:
            tmp=OUT/'rgb_ml_features.part.npz'
            np.savez_compressed(tmp,rgb_sequence=np.array(embeddings,dtype=np.float16),records_json=np.array(json.dumps(records,ensure_ascii=False)))
            tmp.replace(OUT/'rgb_ml_features.npz')
        write_json(OUT/'progress.json',{'stage':stage,'planned':len(md),'valid':len(records),'failed':len(failed),'time_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        write_json(OUT/'failed.json',failed)
    try:
        import torch,torchvision
        from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
        net=mobilenet_v3_large(weights=weights).features.eval()
        for p in net.parameters():p.requires_grad_(False)
        transform=weights.transforms()
        torch.save(net.state_dict(),OUT/'frozen_rgb_extractor.pt')
        if (OUT/'rgb_ml_features.npz').exists():
            prior=np.load(OUT/'rgb_ml_features.npz',allow_pickle=False)
            records=json.loads(str(prior['records_json']));embeddings=list(prior['rgb_sequence'])
            assert len(records)==len(embeddings)
            if (OUT/'failed.json').exists():failed=json.loads((OUT/'failed.json').read_text())
        checkpoint('downloading_temporary_archive')
        provenance=json.loads((OLD/'mendeley_provenance.json').read_text());entry=provenance['file']
        archive=f.Downloader(work,8).get(entry['url'],work/'source.zip',entry['size'],entry['sha'])
        done={r['name'] for r in records}|{r['name'] for r in failed}
        with zipfile.ZipFile(archive) as z:
            names=set(z.namelist())
            for index,row in md.iterrows():
                if row['name'] in done:continue
                path=work/'current.mp4'
                try:
                    if row['name'] not in names:raise ValueError('Frozen original member name not found')
                    with z.open(row['name']) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
                    if f.sha256(path)!=row.sha256:raise ValueError('Frozen original video SHA256 differs')
                    cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if count<8:cap.release();raise ValueError('Fewer than8 source frames')
                    wanted=set(np.linspace(0,count-1,8).astype(int));tensors=[]
                    try:
                        for j in range(count):
                            if not cap.grab():break
                            if j not in wanted:continue
                            ok,frame=cap.retrieve()
                            if not ok:continue
                            tensors.append(transform(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
                    finally:cap.release()
                    if len(tensors)!=8:raise ValueError('Sampled RGB frames not fully decoded')
                    with torch.inference_mode():
                        hidden=net(torch.stack(tensors));vec=torch.nn.functional.adaptive_avg_pool2d(hidden,1).flatten(1).cpu().numpy()
                    assert vec.shape==(8,960) and np.isfinite(vec).all()
                    records.append({**row.to_dict(),'original_feature_index':int(source_ids[index]),'rgb_sample_frames':8})
                    embeddings.append(vec.astype(np.float16))
                except Exception as exc:
                    failed.append({'name':row['name'],'exercise':row.exercise,'quality':row.quality,'group':row.group,'split':row.split,'error':repr(exc)})
                finally:path.unlink(missing_ok=True)
                if (len(records)+len(failed))%10==0:
                    checkpoint('rgb_features_extracting');print('RGB_FEATURES',len(records),'FAILED',len(failed),flush=True)
        archive.unlink(missing_ok=True);checkpoint('rgb_features_complete')
        m=pd.DataFrame(records);R=np.asarray(embeddings,dtype=np.float32);ids=m.original_feature_index.to_numpy()
        original=np.load(OLD/'final_ml_features.npz',allow_pickle=False)
        appearance=np.concatenate([R.mean(1),R.std(1),np.abs(np.diff(R,axis=1)).mean(1)],axis=1)
        feats={'rgb':appearance,'rgb_plus_pose':np.concatenate([appearance,original['image_summary'][ids],original['temporal'][ids]],axis=1)}
        # PCA/scaler/imputation are fitted within EACH train fold, never globally.
        models={
          'linear':make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),PCA(n_components=24,svd_solver='randomized',random_state=95),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000)),
          'rbf':make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),PCA(n_components=24,svd_solver='randomized',random_state=95),SVC(C=10,class_weight='balanced'))}
        tasks=['ACTION_7']+sorted(md.exercise.unique());bundle={};reports={};search=[];allpred=[]
        from sklearn.metrics import balanced_accuracy_score
        for task in tasks:
            eligible=np.ones(len(m),bool) if task=='ACTION_7' else m.exercise.eq(task).to_numpy()
            tr=np.flatnonzero(eligible&m.split.eq('train'));va=np.flatnonzero(eligible&m.split.eq('validation'));te=np.flatnonzero(eligible&m.split.eq('test'))
            y=m.exercise.to_numpy() if task=='ACTION_7' else m.quality.to_numpy();g=m.group.to_numpy();classes=sorted(set(y[tr]));rank=[]
            if len(tr)<32 or not len(va) or not len(te):raise RuntimeError('Insufficient frozen partition support for '+task)
            for fn,X in feats.items():
                for name,est in models.items():
                    scores=[]
                    for a,b in GroupKFold(5).split(X[tr],y[tr],g[tr]):
                        assert not set(g[tr[a]])&set(g[tr[b]])
                        fit=clone(est).fit(X[tr[a]],y[tr[a]]);pr=fit.predict(X[tr[b]])
                        scores.append(balanced_accuracy_score(y[tr[b]],pr))
                    row={'task':task,'features':fn,'model':name,'train_group_cv_balanced_accuracy':float(np.mean(scores))};rank.append(row);search.append(row)
            choice=sorted(rank,key=lambda r:(-r['train_group_cv_balanced_accuracy'],r['features'],r['model']))[0]
            X=feats[choice['features']];fit=clone(models[choice['model']]).fit(X[tr],y[tr]);bundle[task]={'model':fit,'features':choice['features']}
            reports[task]={'selection':choice,'validation':e.score(y[va],fit.predict(X[va]),g[va],classes),'legacy_test':e.score(y[te],fit.predict(X[te]),g[te],classes)}
            planned_n=int(((md.split=='test')&((md.exercise==task) if task!='ACTION_7' else True)).sum())
            reports[task]['legacy_test'].update(planned_n=planned_n,rgb_coverage=len(te)/planned_n,strict_accuracy_counting_rgb_failures_wrong=float(np.sum(fit.predict(X[te])==y[te])/planned_n))
            for part,ix in [('validation',va),('legacy_test',te)]:
                for i,pr in zip(ix,fit.predict(X[ix])):allpred.append({'task':task,'evaluation':part,'name':m.iloc[i]['name'],'actual':y[i],'predicted':str(pr),'group':g[i]})
            checkpoint('training_'+task);print('RGB_TRAINED',task,choice,'legacy',reports[task]['legacy_test']['accuracy'],flush=True)
        te=np.flatnonzero(m.split.eq('test'));a=bundle['ACTION_7'];ap=a['model'].predict(feats[a['features']][te])
        qp=np.array([bundle[ex]['model'].predict(feats[bundle[ex]['features']][i:i+1])[0] for i,ex in zip(te,ap)])
        cascade=m.iloc[te].copy();cascade['predicted_exercise']=ap;cascade['predicted_quality']=qp
        actual=(cascade.exercise+'|'+cascade.quality).to_numpy();predicted=(cascade.predicted_exercise+'|'+cascade.predicted_quality).to_numpy()
        joint=e.score(actual,predicted,cascade.group.to_numpy(),sorted(set(actual)|set(predicted)))
        # Original23 recognizer stays frozen; all misroutes/unsupported quality are errors.
        oldbundle=joblib.load(OLD/'fitness_models.joblib');oldaction=[];oldquality=[]
        for i in te:
            oi=ids[i];t=original['temporal'][oi]
            if np.isfinite(t).all():est=oldbundle['action'];prob=est.predict_proba(t[None])[0]
            else:
                est=oldbundle['action_frame'];o=original['frame_offsets'];prob=est.predict_proba(original['frame_features'][o[oi]:o[oi+1]]).mean(0)
            ex=str(est.classes_[prob.argmax()]);oldaction.append(ex)
            oldquality.append(str(bundle[ex]['model'].predict(feats[bundle[ex]['features']][i:i+1])[0]) if ex in bundle else 'unavailable')
        cascade['frozen23_exercise']=oldaction;cascade['frozen23_candidate_quality']=oldquality
        combined=np.array([ex+'|'+q for ex,q in zip(oldaction,oldquality)]);joint23=e.score(actual,combined,cascade.group.to_numpy(),sorted(set(actual)|set(combined)))
        planned_test=md.split.eq('test').sum();joint['planned_test_n']=int(planned_test);joint['rgb_coverage']=len(te)/planned_test
        joint['strict_accuracy_counting_rgb_failures_wrong']=float((actual==predicted).sum()/planned_test)
        joint23['strict_accuracy_counting_rgb_failures_wrong']=float((actual==combined).sum()/planned_test)
        result={'status':'rgb_pose_candidate_complete','target':.95,'independent95_verified':False,'tasks':reports,'legacy_action7_cascade':joint,'frozen23_recognizer_cascade':joint23,
          'scope':'Frozen ImageNet RGB feature extraction8frames960dimensions, RGB+pose candidate quality for7 exercises. Legacy subject split retained. Not new independent data.',
          'limitations':['Legacy test has been used before; never a new independent target verification.','RGB may exploit background, clothing and camera cues; fresh subject/location tests required.','Uniform8frame sampling may miss short form errors; ImageNet pretraining is not exercise quality supervision.','Original excluded raw videos are outside frozen829 feature-source sample universe.','All failed RGB extractions are recorded and count wrong in strict cascade.'],
          'original_models_replaced':False,'deployment_status':'research_only_no_automatic_replacement','selection_policy':'Original TRAIN subject GroupKFold only; transforms fit inside each fold; no tuning on validation/test.',
          'environment':{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__},
          'pretrained_weights_source':weights.url,'source':provenance}
        joblib.dump({'models':bundle,'feature_spec':'RGB mean960+std960+mean_abs_temporal_delta960; optionally image_summary256+temporal1232','frame_extractor':'frozen_rgb_extractor.pt; torchvision MobileNetV3-Large .features plus adaptive_avg_pool2d1; IMAGENET1K_V2.transforms','scope':result['scope']},OUT/'candidate_models.joblib',compress=3)
        m.to_csv(OUT/'split_manifest.csv',index=False);pd.DataFrame(search).to_csv(OUT/'train_cv_search.csv',index=False);pd.DataFrame(allpred).to_csv(OUT/'task_predictions.csv',index=False);cascade.to_csv(OUT/'cascade_predictions.csv',index=False)
        write_json(OUT/'candidate_report.json',result)
        # Verify disk read-back and model reload without retraining or selecting again.
        saved=joblib.load(OUT/'candidate_models.joblib');s=saved['models']['ACTION_7'];assert np.array_equal(ap,s['model'].predict(feats[s['features']][te]))
        reread=pd.read_csv(OUT/'cascade_predictions.csv');assert np.isclose(((reread.exercise==reread.predicted_exercise)&(reread.quality==reread.predicted_quality)).mean(),joint['accuracy'])
        write_json(OUT/'integrity.json',{'reread_verified':True,'sha256':{p.name:f.sha256(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['integrity.json','progress.json']}})
        checkpoint('rgb_candidate_complete');print('RGB_POSE_CANDIDATE_COMPLETE',joint23['strict_accuracy_counting_rgb_failures_wrong'],flush=True)
    except BaseException:
        checkpoint('failed');write_json(OUT/'failure.json',{'traceback':traceback.format_exc()});raise
    finally:
        shutil.rmtree(work,ignore_errors=True);write_json(OUT/'cleanup.json',{'task_temporary_raw_and_weight_cache_removed':not work.exists(),'existing_models_or_originals_deleted':False});print('RGB_RAW_TEMP_CLEANED',flush=True)

if __name__=='__main__':main()
