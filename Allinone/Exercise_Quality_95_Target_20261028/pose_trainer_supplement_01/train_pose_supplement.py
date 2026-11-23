"""Acquire/validate already downloaded public OpenPose ML arrays and evaluate
two supplemental quality models. All unidentifiable source people stay in ONE
group. Source-only results are file/frame references, never subject accuracy.
"""
from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
import train_elastic_candidate as e

HERE=Path(__file__).resolve().parent

def source_sequences(plan,temporary):
    mapping={11:5,12:2,13:6,14:3,15:7,16:4,23:11,24:8,25:12,26:9,27:13,28:10}
    out=[];records=[]
    for r in plan['records']:
        path=temporary/r['name'];raw=path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha']
        p=np.load(path,allow_pickle=False);assert p.ndim==3 and p.shape[1:]==(18,3)
        cols=[];available=[]
        for a,b,c in e.TRIPLES:
            ia,ib,ic=[mapping[j] for j in (a,b,c)];u=p[:,ia,:2]-p[:,ib,:2];v=p[:,ic,:2]-p[:,ib,:2]
            den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
            ok=(np.min(p[:,[ia,ib,ic],2],axis=1)>=.3)&np.isfinite(den)&(den>1e-8)
            angle=np.arccos(np.clip((u*v).sum(1)/np.maximum(den,1e-8),-1,1))/np.pi
            angle=np.interp(np.arange(len(p)),np.flatnonzero(ok),angle[ok]) if ok.any() else np.full(len(p),.5)
            cols.append(np.interp(np.linspace(0,len(p)-1,16),np.arange(len(p)),angle));available.append(ok.mean())
        seq=np.stack(cols,axis=1).astype(np.float32);assert np.isfinite(seq).all()
        out.append(seq);records.append({**r,'input_sha256':hashlib.sha256(raw).hexdigest(),'frames_in_source_features':len(p),'mean_angle_joint_coverage':float(np.mean(available))})
    return np.array(out),records

def reference_score(y,p):
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    return {'evaluation_type':'frame/file reference only; source subjects unavailable',
            'n':len(y),'accuracy':float(accuracy_score(y,p)),
            'classes':classification_report(y,p,labels=['good','bad'],output_dict=True,zero_division=0),
            'confusion_labels':['good','bad'],'confusion_matrix':confusion_matrix(y,p,labels=['good','bad']).tolist(),
            'pose_from_original_video_coverage':'unknown; provider skeleton arrays supplied',
            'no_person_independent_claim':True}

def main():
    out=HERE/'pose_trainer_01';out.mkdir(exist_ok=False)
    temp=HERE/'pose_trainer_temp';plan=json.loads((HERE/'pose_trainer_plan.json').read_text())
    Xpt,records=source_sequences(plan,temp)
    mdroot=HERE.parent/'final_verification';manifest=pd.read_csv(mdroot/'manifest.csv',keep_default_na=False);z=np.load(mdroot/'features.npz',allow_pickle=False)
    ids=np.flatnonzero(manifest.source.eq('mendeley'));md=manifest.iloc[ids].reset_index(drop=True)
    Xmd=e.sequences(z,ids)['angles2d'];pt=pd.DataFrame(records)
    m=pd.concat([md,pt],ignore_index=True);X=np.concatenate([Xmd,Xpt]);assert m.groupby('group').split.nunique().max()==1
    selection={};models={};predictions=[]
    for task in ['bicep_curl','shoulder_press']:
        eligible=m.exercise.eq(task).to_numpy();g=m.group.to_numpy();y=m.quality.to_numpy()
        oldtr=np.flatnonzero(eligible&m.split.eq('train'));supp=np.flatnonzero(eligible&m.split.eq('supplemental_train_only'))
        tr=np.r_[oldtr,supp];va=np.flatnonzero(eligible&m.split.eq('validation'));te=np.flatnonzero(eligible&m.split.eq('test'))
        involved=np.r_[oldtr,supp,va,te];D=np.full((len(m),len(m)),np.inf,np.float32)
        D[np.ix_(involved,involved)]=e.dtw_distances(X[involved],X[involved]);classes=['good','bad']
        def choose(train):
            rank=[]
            for k in (1,3,5):
                scores=[]
                for a,b in GroupKFold(5).split(train,y[train],g[train]):
                    assert not set(g[train[a]])&set(g[train[b]])
                    p=e.neighbor_predict(D[np.ix_(train[b],train[a])],y[train[a]],classes,k)
                    scores.append(balanced_accuracy_score(y[train[b]],p))
                rank.append({'neighbors':k,'train_group_cv_balanced_accuracy':float(np.mean(scores))})
            return sorted(rank,key=lambda r:(-r['train_group_cv_balanced_accuracy'],r['neighbors']))[0],rank
        # Source holdout evaluated BEFORE adding any of its files to training.
        frozen_old,oldrank=choose(oldtr)
        oldpred=e.neighbor_predict(D[np.ix_(supp,oldtr)],y[oldtr],classes,frozen_old['neighbors'])
        new,newrank=choose(tr)
        predict=lambda ix:e.neighbor_predict(D[np.ix_(ix,tr)],y[tr],classes,new['neighbors'])
        selection[task]={'pre_supplement_selection':frozen_old,'pre_supplement_train_cv':oldrank,
                         'heldout_pose_trainer_file_reference_before_augmentation':reference_score(y[supp],oldpred),
                         'supplemented_selection':new,'supplemented_train_group_cv':newrank,
                         'validation':e.score(y[va],predict(va),g[va],classes),
                         'legacy_test':e.score(y[te],predict(te),g[te],classes)}
        models[task]={'neighbors':new['neighbors'],'training_indices':tr.tolist(),'training_labels':y[tr].tolist(),'classes':classes}
        for ix,p,phase in [(supp,oldpred,'heldout_source_frame_file_reference'),(va,predict(va),'validation'),(te,predict(te),'legacy_test')]:
            for i,pr in zip(ix,p):predictions.append({'exercise':task,'name':m.iloc[i]['name'],'actual_quality':y[i],'predicted_quality':pr,'group':g[i],'evaluation':phase})
        print(task,'additional',len(supp),'legacy accuracy',selection[task]['legacy_test']['accuracy'],flush=True)
    np.savez_compressed(out/'final_ml_sequences.npz',angles2d=X,records_json=np.array(m.to_json(orient='records')))
    m.to_csv(out/'split_and_source_manifest.csv',index=False);(out/'models.json').write_text(json.dumps(models,indent=2))
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    report={'status':'two_exercise_supplement_complete','new_feature_sequences':len(pt),
            'source_support':pt.groupby(['exercise','quality']).size().to_dict().__str__(),
            'tasks':selection,'all7_or_full_pipeline_95_met':False,'original_model_replaced':False,
            'selection_policy':'train-only group CV; entire unidentified source is one indivisible group, original Mendeley subjects retain frozen splits',
            'limitations':['Person IDs absent for all52 new source files. Source reference scores are not person-independent.','Source labels are original filenames good/bad. No original raw video coverage is known.','OpenPose pixel coordinates and MediaPipe normalized image coordinates differ, so this cross-source representation may suffer aspect-ratio/domain differences.','Legacy Mendeley test has been reported before and is not independent new evaluation.','Unknown cross-source subject overlap cannot be excluded; no95percent independent validation claim.','No3D coordinates invented. Only common2D joint angles are used.'],
            'deployment_status':'research_only_no_automatic_replacement'}
    (out/'candidate_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    shutil.copyfile(HERE/'POSE_TRAINER_LICENSE.txt',out/'SOURCE_LICENSE_GPL3.txt')
    shutil.copyfile(HERE/'pose_trainer_plan.json',out/'source_manifest.json')
    # Re-read final compact features and reports before removing temporary arrays.
    packed=np.load(out/'final_ml_sequences.npz',allow_pickle=False);assert np.array_equal(packed['angles2d'],X)
    predictions_read=pd.read_csv(out/'predictions.csv');saved=json.loads((out/'candidate_report.json').read_text())
    for task in selection:
        q=predictions_read[(predictions_read.exercise==task)&(predictions_read.evaluation=='legacy_test')]
        assert np.isclose((q.actual_quality==q.predicted_quality).mean(),saved['tasks'][task]['legacy_test']['accuracy'])
    shutil.rmtree(temp)
    (out/'cleanup.json').write_text(json.dumps({'temporary_downloaded_npy_directory_removed':not temp.exists(),'new_raw_videos_or_images_downloaded':False,'existing_user_data_or_models_deleted':False},indent=2))
    (out/'integrity.json').write_text(json.dumps({'reread_verified':True,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}},indent=2))
    print('POSE_TRAINER_SUPPLEMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
