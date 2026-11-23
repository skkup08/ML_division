"""Predict from V4 video-level pose features; not raw frames or videos.
Only load a trusted joblib model. Usage: python predict_deadlift_v6.py model.joblib features.csv predictions.csv
"""
import sys
import joblib
import numpy as np
import pandas as pd
bundle=joblib.load(sys.argv[1])
df=pd.read_csv(sys.argv[2]);missing=set(bundle['features'])-set(df.columns)
if missing: raise ValueError('Missing video feature columns: '+str(sorted(missing)))
X=df[bundle['features']].replace([np.inf,-np.inf],np.nan)
out=df[[c for c in ['digest','file','subject'] if c in df.columns]].copy()
out['predicted']=bundle['model'].predict(X.to_numpy())
out.to_csv(sys.argv[3],index=False)
