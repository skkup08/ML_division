import os
os.environ['MPLBACKEND']='Agg'
import re,json,hashlib,sys,importlib.util
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np,pandas as pd,joblib,cv2,mediapipe as mp
from PIL import Image,ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,classification_report

ROOT=Path('/content/drive/MyDrive/Allinone')
BASE=ROOT/'Hybrid_Training_Results_Py311'/'20261027_080653'
SRC=ROOT/'scripts'/'train_hybrid_kaggle_py311.py'
OUT=ROOT/'Labeled_Upgrade_Results'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True,exist_ok=True)
SEARCH=[ROOT/'BAD_Validation_Input', ROOT/'new_dataset', ROOT/'Mendeley', ROOT/'mendeley_dataset']
EXCLUDE={'Kaggle_WorkoutFitness','Zenodo_Squat','CFRep'}
MEDIA={'.mp4','.mov','.avi','.mkv','.webm','.gif','.jpg','.jpeg','.png','.bmp','.webp'}
IMAGE={'.jpg','.jpeg','.png','.bmp','.webp'}
BAD={'bad','wrong','incorrect','faulty','improper','poor','unsafe','invalid','error','mistake','나쁜','잘못된','불량','오류'}
GOOD={'good','correct','proper','normal','valid','정상','올바른'}
ALIAS={
 'bench_press':['bench press','bench_press','flat bench'],
 'deadlift':['deadlift','barbell deadlift'],
 'lat_pulldown':['lat pulldown','lat_pulldown'],
 'shoulder_press':['shoulder press','shoulder_press','dumbbell shoulder press'],
 'leg_press':['leg press','leg_press','seated leg press','power leg press'],
 'squat':['squat','back squat','barbell squat']
}

def toks(x):return re.findall(r'[a-z0-9가-힣]+',str(x).lower())
def match_exercise(path):
    parts=path.parts
    words=' '.join(toks(' '.join(parts)))
    hits=[]
    for name,aliases in ALIAS.items():
        for alias in aliases:
            seq=' '.join(toks(alias))
            if re.search(r'(?<!\w)'+re.escape(seq)+r'(?!\w)',words):
                hits.append((len(seq),name))
    return max(hits)[1] if hits else None

def label(path):
    # labels appear in name or a near parent folder; don't interpret arbitrary ancestor directory
    terms=[]
    for part in path.parts[-5:]:terms.extend(toks(part))
    neg=bool(set(terms)&BAD); pos=bool(set(terms)&GOOD)
    if neg and pos:return 'AMBIGUOUS'
    if neg:return 'BAD'
    if pos:return 'GOOD'
    return 'UNLABELED'

assert SRC.exists(),f'복구된 특징 추출 스크립트가 없습니다: {SRC}'
assert BASE.exists(),'기존 22개 모델 폴더가 없습니다'
spec=importlib.util.spec_from_file_location('pose_features_saved',SRC)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Images use a dedicated single-frame feature space; do not feed images into old video models.
def image_feature(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((960,960))
        rgb=np.asarray(im)
    with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=1,min_detection_confidence=0.5) as pose:
        result=pose.process(np.ascontiguousarray(rgb))
        if result.pose_landmarks is None:return None
        feat=mod.get_pose_features(result.pose_landmarks.landmark)
        return np.asarray(feat,dtype=np.float32) if np.isfinite(feat).all() else None

files=set()
for folder in SEARCH:
    if not folder.exists():continue
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in MEDIA and not (set(p.parts)&EXCLUDE):files.add(p)
files=sorted(files)
print('검사할 신규 자료:',len(files),flush=True)
records=[];skipped=[];seen=set()
for i,p in enumerate(files,1):
    ex=match_exercise(p);lab=label(p)
    if not ex or lab not in ('GOOD','BAD'):
        skipped.append({'file':str(p),'reason':'UNKNOWN_EXERCISE' if not ex else lab});continue
    try:
        # exact duplicates do not count as independent samples
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            skipped.append({'file':str(p),'reason':'DUPLICATE'});continue
        feat=image_feature(p) if p.suffix.lower() in IMAGE else mod.extract(p)
        if feat is None:
            skipped.append({'file':str(p),'reason':'POSE_EXTRACTION_FAILED'});continue
        seen.add(digest)
        records.append({'file':str(p),'exercise':ex,'label':lab,'kind':'IMAGE' if p.suffix.lower() in IMAGE else 'VIDEO_GIF','feature':feat,'sha256':digest})
    except Exception as e:skipped.append({'file':str(p),'reason':str(e)[:250]})
    if i%25==0:print(f'처리 {i}/{len(files)} / 유효 {len(records)}',flush=True)

report=[]
if records:
    df=pd.DataFrame(records)
    for (ex,kind),g in df.groupby(['exercise','kind']):
        counts=Counter(g.label)
        r={'exercise':ex,'media_type':kind,'GOOD':counts['GOOD'],'BAD':counts['BAD']}
        if min(counts['GOOD'],counts['BAD'])<5:
            r['status']='NEEDS_5_PER_CLASS';report.append(r);continue
        X=np.stack(g.feature.to_numpy());y=g.label.to_numpy()
        ix1,ix2=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=42)
        test_model=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=42,n_jobs=-1)
        test_model.fit(X[ix1],y[ix1]);pred=test_model.predict(X[ix2])
        cm=confusion_matrix(y[ix2],pred,labels=['GOOD','BAD']).tolist()
        final=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=42,n_jobs=-1)
        final.fit(X,y)
        path=OUT/f'{ex}_{kind}_GOOD_BAD.joblib'
        joblib.dump({'exercise':ex,'mode':'BINARY','media_type':kind,'model':final,'feature_version':'pose_single_frame_v1' if kind=='IMAGE' else 'pose_summary_v2','note':'Random split; correlated subjects may inflate accuracy'},path)
        r.update(status='TRAINED',holdout_balanced_accuracy=float(balanced_accuracy_score(y[ix2],pred)),confusion_matrix=cm,model_path=str(path))
        report.append(r)
        print('학습 완료:',ex,kind,'GOOD',counts['GOOD'],'BAD',counts['BAD'],flush=True)

pd.DataFrame(report,columns=['exercise','media_type','GOOD','BAD','status','holdout_balanced_accuracy','confusion_matrix','model_path']).to_csv(OUT/'report.csv',index=False,encoding='utf-8-sig')
(OUT/'skipped.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='feature'} for r in records],ensure_ascii=False,indent=2),encoding='utf-8')
summary={'candidate_files':len(files),'usable':len(records),'label_counts':dict(Counter(r['label'] for r in records)),'new_binary_models':sum(r['status']=='TRAINED' for r in report),'old_models_untouched':True,'results':str(OUT),'skips':dict(Counter(x['reason'] for x in skipped))}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n===== 최종 결과 =====',flush=True)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if report:print(pd.DataFrame(report)[['exercise','media_type','GOOD','BAD','status']].to_string(index=False),flush=True)
