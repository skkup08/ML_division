"""Independent temporal-alignment experiment on preserved ML features.

No raw media. All choices use original TRAIN subject-group CV only. Legacy
validation/test are diagnostics, not new external evidence. No automatic rollout.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

BODY = [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
TRIPLES = [(11,13,15),(12,14,16),(13,11,23),(14,12,24),
           (11,23,25),(12,24,26),(23,25,27),(24,26,28)]

def sequences(z, ids):
    """Pure per-video transformation; no population estimates or labels."""
    offsets=z['frame_offsets']; result=[]
    for i in ids:
        f=z['frame_features'][offsets[i]:offsets[i+1]]
        p=f[:,:48].reshape(-1,16,3); visible=f[:,48:]>0
        features=[]
        for dims in (2,3):
            for a,b,c in TRIPLES:
                ia,ib,ic=[BODY.index(j) for j in (a,b,c)]
                u=p[:,ia,:dims]-p[:,ib,:dims];v=p[:,ic,:dims]-p[:,ib,:dims]
                den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
                angle=np.arccos(np.clip((u*v).sum(1)/np.maximum(den,1e-8),-1,1))/np.pi
                ok=visible[:,ia]&visible[:,ib]&visible[:,ic]&(den>1e-8)
                # Missing coordinates never use another video's information.
                angle=np.interp(np.arange(len(f)),np.flatnonzero(ok),angle[ok]) if ok.any() else np.full(len(f),.5)
                features.append(angle)
        torso=(p[:,0]+p[:,1]-p[:,6]-p[:,7])/2
        norm=np.maximum(np.linalg.norm(torso,axis=1),1e-8)
        features.extend([np.abs(torso[:,0])/norm,np.abs(torso[:,2])/norm])
        arr=np.stack(features,axis=1)
        seq=np.stack([np.interp(np.linspace(0,len(f)-1,16),np.arange(len(f)),arr[:,j]) for j in range(arr.shape[1])],axis=1)
        result.append(seq)
    allseq=np.asarray(result,dtype=np.float32)
    assert np.isfinite(allseq).all()
    return {'angles2d':allseq[:,:,:8], 'angles_torso':allseq,
            'angles_torso_motion':np.concatenate([allseq,np.diff(allseq,axis=1,prepend=allseq[:,:1])],axis=2)}

def dtw_distances(query, reference, radius=4):
    """Banded DTW, vectorized across references. No third-party model/code."""
    output=np.empty((len(query),len(reference)),np.float32)
    steps=query.shape[1]
    for k,q in enumerate(query):
        cost=cdist(q,reference.reshape(-1,reference.shape[-1]),metric='sqeuclidean')
        cost=cost.reshape(steps,len(reference),steps).transpose(1,0,2)/q.shape[-1]
        d=np.full((len(reference),steps+1,steps+1),np.inf,np.float32);d[:,0,0]=0
        for a in range(1,steps+1):
            for b in range(max(1,a-radius),min(steps,a+radius)+1):
                d[:,a,b]=cost[:,a-1,b-1]+np.minimum(np.minimum(d[:,a-1,b],d[:,a,b-1]),d[:,a-1,b-1])
        output[k]=d[:,-1,-1]/steps
    return output

def neighbor_predict(dist, labels, classes, k):
    order=np.argsort(dist,axis=1,kind='stable')[:,:min(k,len(labels))]
    selected=np.take_along_axis(dist,order,axis=1)
    weights=1/np.maximum(selected,1e-5)
    scores=np.stack([(weights*(labels[order]==c)).sum(1) for c in classes],axis=1)
    return np.asarray(classes)[scores.argmax(1)]

def wilson(correct,n):
    if n==0:return [None,None]
    p=correct/n;zz=1.959963984540054**2;center=(p+zz/(2*n))/(1+zz/n)
    half=np.sqrt(zz*(p*(1-p)/n+zz/(4*n*n)))/(1+zz/n)
    return [float(center-half),float(center+half)]

def score(y,p,groups,classes):
    correct=np.asarray(y)==np.asarray(p);rng=np.random.default_rng(9503);ug=np.unique(groups)
    chunks=[correct[groups==g] for g in ug]
    boot=[np.concatenate([chunks[j] for j in rng.integers(len(chunks),size=len(chunks))]).mean() for _ in range(1000)]
    return {'n':len(y),'subjects':len(ug),'accuracy':float(correct.mean()),
            'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'accuracy_wilson95_descriptive_clip_independence':wilson(int(correct.sum()),len(y)),
            'subject_cluster_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'classes':classification_report(y,p,labels=classes,output_dict=True,zero_division=0),
            'confusion_labels':classes,'confusion_matrix':confusion_matrix(y,p,labels=classes).tolist(),
            'decision_coverage_within_preserved_features':1.0}

def main(data,out):
    data=Path(data);out=Path(out);out.mkdir(exist_ok=False,parents=True)
    mp=data/'manifest.csv';fp=data/'features.npz'
    if not mp.exists():mp=data/'video_manifest_and_splits.csv'
    if not fp.exists():fp=data/'final_ml_features.npz'
    source=pd.read_csv(mp,keep_default_na=False);ids=np.flatnonzero(source.source.eq('mendeley'))
    m=source.iloc[ids].reset_index(drop=True);z=np.load(fp,allow_pickle=False)
    assert m.groupby('group').split.nunique().max()==1
    assert m.sha256.is_unique
    seq=sequences(z,ids);distances={}
    for name,X in seq.items():
        distances[name]=dtw_distances(X,X)
        assert np.allclose(distances[name],distances[name].T,atol=1e-6)
        print('DISTANCES_COMPLETE',name,flush=True)
    groups=m.group.to_numpy();bundles={};reports={};search=[]
    tasks=['ACTION_7']+sorted(m.exercise.unique())
    for task in tasks:
        eligible=np.ones(len(m),bool) if task=='ACTION_7' else m.exercise.eq(task).to_numpy()
        tr=np.flatnonzero(eligible&m.split.eq('train'));va=np.flatnonzero(eligible&m.split.eq('validation'));te=np.flatnonzero(eligible&m.split.eq('test'))
        y=m.exercise.to_numpy() if task=='ACTION_7' else m.quality.to_numpy();classes=sorted(set(y[tr]));rank=[]
        for name,D in distances.items():
            for k in (1,3,5):
                foldscore=[]
                for a,b in GroupKFold(5).split(tr,y[tr],groups[tr]):
                    assert not set(groups[tr[a]])&set(groups[tr[b]])
                    p=neighbor_predict(D[np.ix_(tr[b],tr[a])],y[tr[a]],classes,k)
                    foldscore.append(balanced_accuracy_score(y[tr[b]],p))
                row={'task':task,'sequence':name,'neighbors':k,'train_group_cv_balanced_accuracy':float(np.mean(foldscore))}
                rank.append(row);search.append(row)
        selected=sorted(rank,key=lambda r:(-r['train_group_cv_balanced_accuracy'],r['sequence'],r['neighbors']))[0]
        D=distances[selected['sequence']]
        predict=lambda ix:neighbor_predict(D[np.ix_(ix,tr)],y[tr],classes,selected['neighbors'])
        bundles[task]={'selection':selected,'train_ids':tr,'train_labels':y[tr],'classes':classes}
        reports[task]={'selection':selected,'validation':score(y[va],predict(va),groups[va],classes),
                       'legacy_test':score(y[te],predict(te),groups[te],classes)}
        print(task,'CV',selected['train_group_cv_balanced_accuracy'],'legacy',reports[task]['legacy_test']['accuracy'],flush=True)
    te=np.flatnonzero(m.split.eq('test'))
    def predict_task(task,ix):
        b=bundles[task];s=b['selection'];return neighbor_predict(distances[s['sequence']][np.ix_(ix,b['train_ids'])],b['train_labels'],b['classes'],s['neighbors'])
    action=predict_task('ACTION_7',te);quality=np.array([predict_task(ex,[i])[0] for i,ex in zip(te,action)])
    p=m.iloc[te].copy();p['predicted_exercise']=action;p['predicted_quality']=quality
    p['joint_correct']=(action==p.exercise.to_numpy())&(quality==p.quality.to_numpy())
    actual_joint=(p.exercise+'|'+p.quality).to_numpy()
    predicted_joint=(p.predicted_exercise+'|'+p.predicted_quality).to_numpy()
    joint=score(actual_joint,predicted_joint,p.group.to_numpy(),sorted(set(actual_joint)|set(predicted_joint)))
    result={'status':'elastic_candidate_complete','target':.95,'all7_plus_cascade_95_point_met_legacy':all(reports[t]['legacy_test']['accuracy']>=.95 for t in tasks) and joint['accuracy']>=.95,
            'scope':'Seven supported quality exercises only, with separate action_7 recognizer. Not full23-class recognition. Legacy data reused, not new independent validation.',
            'tasks':reports,'legacy_cascade':joint,'original_models_replaced':False,
            'selection_policy':'5-fold subject GroupKFold on frozen original TRAIN subjects only, deterministic sequence transformation; no test/validation tuning.',
            'limitations':['Only26 subjects,6 legacy test subjects. Confidence intervals are uncertain with few subjects.','Failed/previously excluded raw videos not present in preserved ML features are outside this evaluation denominator.','No new source subjects; this cannot establish the user target on a new independent test.','Image-space estimated joint depth can be noisy. Missing per-video joint trajectories interpolated; full prediction coverage within saved features.'],
            'deployment_status':'research_only_no_automatic_replacement',
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (mp,fp)}}
    (out/'candidate_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    pd.DataFrame(search).to_csv(out/'train_group_cv.csv',index=False);p.to_csv(out/'legacy_test_predictions.csv',index=False)
    m.to_csv(out/'manifest.csv',index=False)
    np.savez_compressed(out/'sequence_features.npz',**seq)
    # Models are safe JSON indices and training labels, paired with the fixed sequences.
    (out/'models.json').write_text(json.dumps({t:{'selection':b['selection'],'train_ids':b['train_ids'].tolist(),'train_labels':b['train_labels'].tolist(),'classes':b['classes']} for t,b in bundles.items()},indent=2))
    check=json.loads((out/'candidate_report.json').read_text());pred=pd.read_csv(out/'legacy_test_predictions.csv')
    assert np.isclose(pred.joint_correct.mean(),check['legacy_cascade']['accuracy'])
    for task in tasks[1:]:
        known=np.flatnonzero(m.split.eq('test')&m.exercise.eq(task));assert len(known)==check['tasks'][task]['legacy_test']['n']
    (out/'cleanup.json').write_text(json.dumps({'new_raw_media_created':False,'existing_models_or_raw_deleted':False},indent=2))
    (out/'integrity.json').write_text(json.dumps({'reread_verified':True,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}},indent=2))
    print('ELASTIC_CANDIDATE_COMPLETE',joint['accuracy'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('data');parser.add_argument('out');args=parser.parse_args();main(args.data,args.out)
