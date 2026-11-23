import pathlib,json,zipfile,collections,re,io,hashlib,shutil
import numpy as np, cv2, requests, mediapipe as mp, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
w=pathlib.Path('/content/fitness_merged_audit'); out=pathlib.Path('/content/drive/MyDrive/Exercise_AI_New_20261117_RF_GroupHoldout_v2');out.mkdir(exist_ok=True)
rows=json.loads((w/'manifest.json').read_text());z=zipfile.ZipFile(w/'source.zip')
model=w/'pose_lite.task'
if not model.exists():
 r=requests.get('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',timeout=60);r.raise_for_status();model.write_bytes(r.content)
opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1)
det=mp.tasks.vision.PoseLandmarker.create_from_options(opts)
uniq={};rejected=[]
def canonical(s):return s.replace('BrenchPress','BenchPress').replace('Abormal','Abnormal').replace('_','').lower()
for r in sorted(rows,key=lambda r:({'valid':0,'test':1,'train':2}[r['split']],r['path'])):
 if len(r['labels'])==1 and canonical(r['group'].rsplit('_',1)[0])==canonical(r['labels'][0]):uniq.setdefault(r['frame'],r)
 else:rejected.append(r)
groups=collections.defaultdict(list)
for r in uniq.values():groups[r['group']].append(r)
features=[];quality=[];exercises=[];groupids=[];frameids=[];audit=[]
for g,rr in sorted(groups.items()):
 rr=sorted(rr,key=lambda r:r['frame']);good=0
 for i in np.linspace(0,len(rr)-1,min(24,len(rr)),dtype=int):
  r=rr[i];b=np.frombuffer(z.read(r['path']),np.uint8);img=cv2.imdecode(b,cv2.IMREAD_COLOR);img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);res=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=img))
  if not res.pose_landmarks:continue
  lm=res.pose_landmarks[0];pts=np.array([[p.x,p.y] for p in lm]);vis=np.array([p.visibility for p in lm])
  if np.min(vis[[11,12,23,24]])<.3:continue
  hip=(pts[23]+pts[24])/2;shoulder=(pts[11]+pts[12])/2;scale=np.linalg.norm(shoulder-hip)
  if scale<.03:continue
  feat=((pts-hip)/scale).reshape(-1);lab=r['labels'][0];exercise=lab.split('_',1)[1].replace('BrenchPress','BenchPress'); features.append(feat);quality.append(int(lab.startswith('Normal_')));exercises.append(exercise);groupids.append(g);frameids.append(r['frame']);good+=1
 audit.append({'group':g,'selected':min(24,len(rr)),'accepted':good}); print('POSE',g,good,flush=True)
det.close();X=np.array(features);y=np.array(quality);ex=np.array(exercises);gg=np.array(groupids);results=[]
for e in sorted(set(ex)):
 ix=np.flatnonzero(ex==e);gs=sorted(set(gg[ix]));hold=set()
 for label in [0,1]:
  lg=sorted(set(gg[ix[y[ix]==label]]))
  if len(lg)<2:break
  hold.add(lg[-1])
 else:
  # Keep matching numeric video IDs together even across quality classes.
  ids={g.rsplit('_',1)[-1] for g in hold};te=ix[[g.rsplit('_',1)[-1] in ids for g in gg[ix]]];tr=ix[[g.rsplit('_',1)[-1] not in ids for g in gg[ix]]]
  if len(set(y[tr]))<2 or len(set(y[te]))<2:results.append({'exercise':e,'status':'SKIP_SPLIT'});continue
  clf=RandomForestClassifier(n_estimators=200,max_depth=5,class_weight='balanced',random_state=42,n_jobs=-1);clf.fit(X[tr],y[tr]);pred=clf.predict(X[te]);dest=out/(e+'_EXPERIMENTAL.joblib');joblib.dump({'model':clf,'feature':'pose33_2d_hip_center_torso_scale_v1','labels':{0:'source_Abnormal',1:'source_Normal'},'not_validated_for_deployment':True},dest,compress=3);assert np.array_equal(joblib.load(dest)['model'].predict(X[te]),pred)
  result={'exercise':e,'status':'EXPERIMENTAL_ONLY','train_frames':len(tr),'test_frames':len(te),'train_groups':sorted(set(gg[tr])),'test_groups':sorted(set(gg[te])),'balanced_accuracy':float(balanced_accuracy_score(y[te],pred)),'confusion_matrix':confusion_matrix(y[te],pred).tolist()};results.append(result);print('MODEL',e,round(result['balanced_accuracy'],3),flush=True)
  continue
 results.append({'exercise':e,'status':'SKIP_TOO_FEW_POSE_GROUPS'})
(out/'rejected_label_records.json').write_text(json.dumps(rejected)); (out/'retained_frame_manifest.json').write_text(json.dumps(list(uniq.values()))); (out/'metrics.json').write_text(json.dumps(results,indent=2));(out/'pose_audit.json').write_text(json.dumps(audit,indent=2))
for f in ['audit.json','overlap_metadata.json']:shutil.copy2(w/f,out/f)
(out/'provenance.json').write_text(json.dumps({'source':'https://universe.roboflow.com/fitness-training/fitness_merged/dataset/4','license_as_supplied':'CC BY 4.0','zip_sha256':hashlib.sha256((w/'source.zip').read_bytes()).hexdigest(),'original_frames':3803,'source_filename_groups':61,'original_split_leaks':61,'subjects':'unknown','quality_labels':'unverified source labels inherited on frames','original_creator_rights':'not independently verified','allinone':'No data or model reused; 49 metadata files had no source/name match; renamed content overlap unknown','scope':'15 dataset movement categories, NOT all 26 equipment','evaluation':'small video-identifier holdout, not subject-independent, not real-world accuracy'},indent=2))
print('DONE_MODELS',len(list(out.glob('*.joblib'))),'OUT',out,flush=True)
