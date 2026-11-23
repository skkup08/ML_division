import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
p=Path(__file__).parent
f=p/'final_ml_features.npz'
if not f.exists():f=p/'checkpoint_features.npz'
mf=p/'feature_manifest.csv'
if not mf.exists():mf=p/'checkpoint_manifest.csv'
z=np.load(f,allow_pickle=False);m=pd.read_csv(mf);n=len(m)
assert z['temporal'].shape==(n,1232)
assert z['image_summary'].shape==(n,256)
assert z['frame_features'].ndim==2 and z['frame_features'].shape[1]==64
assert z['frame_offsets'].shape==(n+1,)
o=z['frame_offsets'];assert o[0]==0 and o[-1]==len(z['frame_features']) and (np.diff(o)>0).all()
assert np.isfinite(z['image_summary']).all() and np.isfinite(z['frame_features']).all()
finite=np.isfinite(z['temporal']);valid=finite.all(axis=1);missing=np.isnan(z['temporal']).all(axis=1)
assert (valid|missing).all(), 'Partially missing or infinite temporal row'
assert np.array_equal(valid,m.temporal_available.to_numpy(dtype=bool))
assert m.sha256.notna().all() and m.sha256.is_unique
assert m.group.notna().all()
assert set(m.exercise).issubset({'bent_over_row','bicep_curl','push_up','shoulder_press','side_bend','squat','triceps_extension'})
assert set(m.quality).issubset({'good','bad'})
assert m.split.isna().all() or m.split.fillna('').eq('').all()
counts=[{'exercise':str(ex),'quality':str(q),'videos':int(v)} for (ex,q),v in m.groupby(['exercise','quality']).size().items()]
out={'status':'checkpoint_integrity_passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'run':'run_20261027_142919','saved_snapshot_videos':n,'saved_snapshot_subject_groups':int(m.group.nunique()),'temporal_available':int(valid.sum()),'image_fallback_only':int(missing.sum()),'frame_feature_rows':int(len(z['frame_features'])),'duplicate_video_hashes':0,'row_alignment_and_offsets_valid':True,'all_expected_feature_values_finite':True,'video_split_assigned':False,'full_training_complete':False,'new_raw_cleanup_verified':False,'video_class_counts':counts,'artifacts':{'final_ml_features.npz':{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size},'feature_manifest.csv':{'sha256':hashlib.sha256(mf.read_bytes()).hexdigest(),'bytes':mf.stat().st_size}},'note':'This is an earlier saved feature snapshot of continuing extraction, not final model evaluation. NaN-only temporal rows are intentional fallback markers; image/frame features are finite. Subject-disjoint split validation follows after extraction.'}
(p/'video_checkpoint_integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
