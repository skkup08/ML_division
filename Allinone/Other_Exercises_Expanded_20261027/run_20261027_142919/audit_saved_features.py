"""Validate saved compact ML features without modifying any running training."""
import json,hashlib,csv
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
base=Path(__file__).parent
metrics_path=base/'metrics.json'
if not metrics_path.exists():metrics_path=base/'metrics_snapshot.json'
metrics=json.loads(metrics_path.read_text())
rows=[]
for name,key,expected_dims in [('plank','error_plank',64),('curl','error_bicep_curl',32)]:
 canonical='plank' if name=='plank' else 'bicep_curl'
 paths={part:(base/f'final_ml_{canonical}_{part}.npz' if (base/f'final_ml_{canonical}_{part}.npz').exists() else base/f'{name}_{part}.npz') for part in ['train','test']}
 data={part:np.load(path,allow_pickle=False) for part,path in paths.items()}
 a,b=data['train'],data['test']
 assert all(len(z['X'])==len(z['y'])==len(z['hashes']) for z in data.values())
 assert all(z['X'].ndim==2 and z['X'].shape[1]==expected_dims for z in data.values())
 assert all(np.isfinite(z['X']).all() for z in data.values())
 train_features=set(row.tobytes() for row in a['X'])
 raw_overlap=len(np.intersect1d(a['hashes'],b['hashes']))
 feature_overlap=sum(row.tobytes() in train_features for row in b['X'])
 assert raw_overlap==0 and feature_overlap==0
 report=metrics[key]['publisher_test_frame_reference_only']
 assert len(a['X'])==metrics[key]['train_rows'] and len(b['X'])==report['n']
 row={'exercise':name,'train_rows':len(a['X']),'test_rows':len(b['X']),'feature_dimensions':expected_dims,'finite_values':True,'train_test_coordinate_hash_overlap':raw_overlap,'train_test_exact_feature_overlap':feature_overlap,'publisher_frame_reference_accuracy':report['accuracy'],'publisher_frame_reference_macro_f1':report['macro_f1'],'subject_independent_validation':False,'train_labels':dict(zip(*[x.tolist() for x in np.unique(a['y'],return_counts=True)])),'files':{part:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for part,p in paths.items()}}
 rows.append(row)
out={'status':'partial_data_audit_passed','utc':datetime.now(timezone.utc).isoformat(),'run':'run_20261027_142919','full_video_training_complete':False,'checks':rows,'limitations':['Exact duplicate checks do not rule out near-duplicate adjacent frames or the same person in both partitions.','No person/video identifiers in source correction CSVs; publisher frame reference scores are not person-independent GOOD/BAD accuracy.','Plank covers correct/hips high/hips low; curl covers correct/torso lean back only.','Audit does not interrupt or modify the continuing video training; final raw-data cleanup is not yet verified.']}
(base/'partial_feature_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with (base/'partial_validation_scorecard.csv').open('w',newline='') as f:
 fields=['exercise','train_rows','test_rows','feature_dimensions','train_test_coordinate_hash_overlap','train_test_exact_feature_overlap','publisher_frame_reference_accuracy','publisher_frame_reference_macro_f1','subject_independent_validation']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({'status':out['status'],'checks':[{k:v for k,v in r.items() if k!='files'} for r in rows]},ensure_ascii=False,indent=2))
