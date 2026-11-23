"""Read-only model/prediction audit; no selection, fitting or threshold changes."""
from pathlib import Path
import json,hashlib,warnings,sys
import numpy as np,pandas as pd,joblib,sklearn
import train_candidate as feature_code
import train_elastic_candidate as metric_code

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'qevd_squat_01_verified'
plan=json.loads((ROOT/'qevd_squat_extraction_plan.json').read_text())['records']
records=json.loads((OUT/'feature_records.json').read_text());failed=json.loads((OUT/'failed.json').read_text())
m=pd.DataFrame(records);p=pd.DataFrame(plan)
z=np.load(OUT/'ml_features.npz',allow_pickle=False)
assert len(m)==len(z['temporal'])==len(z['image_summary'])==len(z['frame_offsets'])-1
assert z['frame_offsets'][-1]==len(z['frame_features'])
assert m.groupby('worker_id').split.nunique().max()==1 and p.groupby('worker_id').split.nunique().max()==1
assert m.video_path.is_unique and m.sha256.is_unique
assert set(m.video_path).isdisjoint({r['video_path'] for r in failed})
assert set(m.video_path)|{r['video_path'] for r in failed}==set(p.video_path)
lookup=p.set_index('video_path')
for r in records:
    ref=lookup.loc[r['video_path']]
    assert str(r['worker_id'])==str(ref.worker_id) and r['quality']==ref.quality and r['split']==ref.split
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always');bundle=joblib.load(OUT/'qevd_squat_candidate.joblib')
model=bundle['model'];name=bundle['features'];geometry=feature_code.geometry(z)
X={'geometry':geometry,'summary':z['image_summary'],'combined':np.concatenate([z['image_summary'],geometry],axis=1)}[name]
report=json.loads((OUT/'candidate_report.json').read_text());pred=pd.read_csv(OUT/'new_subject_test_predictions.csv',dtype={'video_path':str,'worker_id':str})
te=np.flatnonzero(m.split.eq('test'));test=m.iloc[te].copy();prediction=model.predict(X[te])
assert pred.video_path.tolist()==test.video_path.tolist() and pred.predicted.tolist()==prediction.tolist()
assert np.isclose((prediction==test.quality.to_numpy()).mean(),report['new_subject_test']['accuracy'])
test['predicted']=prediction
new_metric=metric_code.score(test.quality.to_numpy(),prediction,test.group.to_numpy(),['good','bad'])
planned_test=p[p.split.eq('test')].copy();planned_test['predicted']=planned_test.video_path.map(dict(zip(test.video_path,prediction))).fillna('unavailable')
strict=metric_code.score(planned_test.quality.to_numpy(),planned_test.predicted.to_numpy(),planned_test.group.to_numpy(),['good','bad','unavailable'])
strict['coverage']=len(test)/len(planned_test)
assert np.isclose(strict['accuracy'],report['new_subject_test']['strict_accuracy_counting_pose_failures_wrong'])
for part,key in [('validation','validation'),('test','new_subject_test')]:
    ix=np.flatnonzero(m.split.eq(part));pr=model.predict(X[ix])
    assert np.isclose((pr==m.quality.to_numpy()[ix]).mean(),report[key]['accuracy'])
cleanup=json.loads((OUT/'cleanup.json').read_text());assert cleanup['temporary_raw_directory_removed'] and not cleanup['existing_raw_or_models_deleted']
failmap={r['video_path']:r['error'] for r in failed};sha=dict(zip(m.video_path,m.sha256))
manifest=p.drop(columns=['zip_member']).copy();manifest['processing_status']=manifest.video_path.map(lambda v:'failed' if v in failmap else 'features_saved');manifest['failure_reason']=manifest.video_path.map(failmap).fillna('');manifest['sha256']=manifest.video_path.map(sha).fillna('')
manifest.to_json(OUT/'planned_split_and_label_manifest.json.gz',orient='records',force_ascii=False,compression='gzip')
audit={'status':'saved_model_and_predictions_verified','new_source':'QEVD explicit squat issue labels only','planned_clips':len(p),'usable_feature_clips':len(m),'failed_clips':len(failed),'all_source_coverage':len(m)/len(p),
       'planned_workers':int(p.worker_id.nunique()),'test_workers':int(planned_test.worker_id.nunique()),'worker_split_overlap':False,'duplicate_sha256':0,
       'test_valid':new_metric,'test_strict_failures_count_wrong':strict,'pose_failure_denominator_verified':True,'cleanup':cleanup,'all7_or_full23_cascade_95_verified':False,
       'limitations':['Known squat clips only; no exercise recognition stage tested.','This particular frozen51clip worker holdout has now been evaluated and cannot become a fresh test in later experiments.','Clip-level Wilson interval assumes independent clips; worker bootstrap interval better reflects clustering, but few workers limit precision.','GOOD means QEVD no obvious issue; BAD limited to explicit back/depth/insufficient annotations, not every possible squat issue.'],
       'local_verification_environment':{'python':sys.version,'sklearn':sklearn.__version__,'load_warnings':sorted(set(str(w.message) for w in ws))},
       'selection_or_threshold_changed':False}
(OUT/'verification_report.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
saved=json.loads((OUT/'verification_report.json').read_text());assert saved['test_strict_failures_count_wrong']['n']==51
(OUT/'verification_integrity.json').write_text(json.dumps({'reread_verified':True,'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name!='verification_integrity.json'}},indent=2))
print(json.dumps({k:audit[k] for k in ['planned_clips','usable_feature_clips','failed_clips','planned_workers','test_workers']},ensure_ascii=False))
print('new worker strict test',strict['accuracy'],'worker bootstrap95',strict['subject_cluster_bootstrap95'],'Wilson95',strict['accuracy_wilson95_descriptive_clip_independence'],flush=True)
