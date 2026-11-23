"""Predict with the saved feature format (not raw images or videos).
Usage: python predict_saved_features.py RUN_DIRECTORY OUTPUT.csv
Use the matching sklearn 1.9.1 environment and only load your trusted model.
For raw videos, fitness_expanded_runtime.py provides --predict/--bundle.
"""
from pathlib import Path
import sys
import numpy as np,pandas as pd,joblib,sklearn
if sklearn.__version__!='1.9.1':raise RuntimeError('Use scikit-learn==1.9.1 to match training')
p=Path(sys.argv[1]);b=joblib.load(p/'fitness_models.joblib');z=np.load(p/'final_ml_features.npz',allow_pickle=False);m=pd.read_csv(p/'video_manifest_and_splits.csv',keep_default_na=False)
assert len(m)==len(z['temporal'])==len(z['image_summary'])
o=z['frame_offsets'];rows=[]
for i,r in m.iterrows():
 temporal=np.isfinite(z['temporal'][i]).all()
 if temporal and b['action'] is not None:
  a=b['action'];pr=a.predict_proba(z['temporal'][i:i+1])[0]
 else:
  a=b['action_frame'];pr=a.predict_proba(z['frame_features'][o[i]:o[i+1]]).mean(0)
 ex=str(a.classes_[pr.argmax()]);row={'source':r.source,'name':r['name'],'exercise':ex,'action_score':float(pr.max()),'quality':'','quality_score':None,'mode':'unsupported_quality'}
 if temporal and ex in b['quality']:q=b['quality'][ex];X=z['temporal'][i:i+1];mode='temporal'
 elif ex in b['quality_image']:q=b['quality_image'][ex];X=z['image_summary'][i:i+1];mode='image_summary'
 else:rows.append(row);continue
 pr=q.predict_proba(X)[0];row.update(quality=str(q.classes_[pr.argmax()]),quality_score=float(pr.max()),mode=mode);rows.append(row)
pd.DataFrame(rows).to_csv(sys.argv[2],index=False)
# Raw argmax reference only: scores are not calibrated safety probabilities.
