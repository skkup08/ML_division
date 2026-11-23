"""Reproducible raw preservation and conservative candidate preparation; never auto-approves labels."""
from pathlib import Path
import fcntl, os
_BUILD_LOCK=open("/content/SKKU1398_equipment_v1.lock","w")
try: fcntl.flock(_BUILD_LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise SystemExit("A build is already running; use its log.")
import json, csv, re, math, hashlib, shutil, zipfile, time, collections
from PIL import Image, ImageOps, ImageDraw
import yaml, imagehash
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML/02_dataset/SKKU1398_equipment_v1')
WORK=Path('/content/SKKU1398_equipment_v1_work')
for name in ['raw_sources','manifests','review','reports','yolo']:(ROOT/name).mkdir(parents=True,exist_ok=True)
WORK.mkdir(exist_ok=True)
SPECS=[('fitfuel','all-gym-equipment',2),('rizzlabzz','gymbro',2),('trainers-workspace','gym-equipment-t6kck-fkuoc',1),('gymze','gymze',1),('gym-qknei','gym-shkud',5),('dans-new-workspace','home-gym-equipment-detection',6),('hn-2kixi','healnion-f4l32',1),('hydraws','exercise-identifier',2),('shazils-workspace','gym-0vnct-2p6bi',1),('xuejun-hong','gym-equipment-iv6k9',1)]
NAMES=['treadmill','recumbent_bike','upright_bike','elliptical','seated_chest_press','pectoral_fly','lat_pulldown','shoulder_press','seated_cable_row','arm_curl','abdominal_machine','assisted_chin_dip','rotary_torso','leg_extension','leg_curl','seated_leg_press','cable_crossover','smith_machine','megamax','chin_dip_hipflexor','power_leg_press','power_rack','situp_bench','double_twist','hyperextension','incline_decline_bench_press']
KO=['트레드밀','리컴번트 바이크','업라이트 바이크','일립티컬','체스트 프레스','펙토럴 플라이','하이플라이(랫풀다운 대응)','숄더 프레스','롱풀','암컬','앱도미널','친업','로터리 토르소','레그 익스텐션','레그 컬','시티드 레그프레스','케이블 크로스오버','스미스 머신','메가맥스','치닝·디핑·힙플렉서','파워 레그프레스','파워랙','싯업','더블 트위스트','하이퍼 익스텐션','슈퍼 벤치프레스 인클라인/디클라인']
BASE={1,5,8,16,18}
ALIASES={
1:['treadmill'],2:['recumbent bike'],3:['stationary bike','spin bike','exercise bike'],4:['elliptical','elliptical trainer','eliptical cycle'],
5:['chest press machine','chest press','chestpress'],6:['chest fly machine','chest fly','pec deck machine','pec fly machine'],
7:['lat pull down machine','lat pulldown machine','lat pulldown','cable lat pulldown'],8:['shoulder press machine','shoulder press','overhead shoulder press','shoulderpress'],
9:['seated cable row','seated row machine','seated row','low row'],10:['arm curl','armcurl','arm curl machine','arm machine','preacher curl','preacher curl machine','curl machine'],
11:['ab crunch machine','abdominal machine','seated ab crunch','seated abdominal curl','ab crunch'],12:['dip chin assist','assisted pull up and dip machine','assisted pull up machine','assisted chin dip','pull up bar'],
13:['torso rotation machine','torso machine'],14:['leg extension','legextension','leg extension machine'],15:['leg curl','legcurl','leg curl machine','lying down leg curl','seated leg curl','hamstring curl machine','seated leg curl extension'],
16:['leg press','leg press machine','leg 90'],17:['cable crossover machine','cable crossover','pulley machine','functional trainer','cable machine'],18:['smith machine'],
20:['leg raise tower','power tower','chin up dipping machine','chin dip'],21:['powerlegpress','leg 45'],22:['squat rack','squat stand'],23:['abdominal bench','ab bench'],25:['back extension machine','back extension','roman chair'],26:['incline bench','adjustable bench','adjustable weight bench','declined bench press','incline bench press']}
def norm(s):return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
LOOKUP={norm(a):n for n,aa in ALIASES.items() for a in aa}
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvout(path,rows,fields):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def status(stage,**kw):
 v={'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw};dump(ROOT/'reports/build_status.json',v);print(json.dumps(v),flush=True)
registry=[{'center_id':i,'name':n,'name_ko':KO[i-1],'scope':'existing5' if i in BASE else 'additional21','exact_center_shape_verified':False} for i,n in enumerate(NAMES,1)]
dump(ROOT/'manifests/class_registry.json',registry)
status('preserving_raw')
sources=[]; allrows=[]; mappings=[]; annotations=[]
for ws,proj,ver in SPECS:
 root=next((p for p in [Path('/content/equipment_counts')/ws,Path('/content/equipment_search2')/ws] if (p/'data.yaml').exists()),None)
 if root is None:raise RuntimeError('Missing source: '+ws)
 key=f'{ws}__{proj}__v{ver}'
 archive=ROOT/'raw_sources'/(key+'.zip')
 if archive.exists():
  try:
   with zipfile.ZipFile(archive) as check: assert check.testzip() is None
  except (zipfile.BadZipFile,AssertionError,EOFError):
   archive.rename(archive.with_suffix('.incomplete_'+str(int(time.time()))))
 if not archive.exists():
  tmp=WORK/(key+'.zip')
  with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED) as z:
   for p in sorted(root.rglob('*')):
    if p.is_file() and p.suffix.lower()!='.zip':z.write(p,p.relative_to(root))
  partial=archive.with_suffix('.zip.partial')
  shutil.copyfile(tmp,partial)
  assert sha(tmp)==sha(partial)
  partial.replace(archive)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  files=len(z.infolist())
 source={'source_id':key,'workspace':ws,'project':proj,'version':ver,'url':f'https://universe.roboflow.com/{ws}/{proj}/dataset/{ver}','license_reported':'MIT' if ws=='dans-new-workspace' else 'CC BY 4.0','license_evidence':'Prior investigation and bundled README; original photographer rights unverified','original_photo_origin':'unknown','archive':str(archive.relative_to(ROOT)),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'archive_files':files,'local_root':str(root)}
 sources.append(source);dump(ROOT/'manifests/sources.json',sources)
 print('RAW_VERIFIED',key,source['archive_bytes'],flush=True)
 cfg=yaml.safe_load((root/'data.yaml').read_text());names=cfg['names'];names=dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
 for cid,name in names.items():mappings.append({'source_id':key,'source_class_id':cid,'source_class':name,'candidate_center_id':LOOKUP.get(norm(name),''),'status':'candidate_only_not_semantic_approval'})
 cache=ROOT/"manifests"/(key+".jsonl")
 if cache.exists():
  cached=[json.loads(line) for line in cache.read_text().splitlines()]
  if cached and all("automated_valid" in r and "phash" in r for r in cached):
   allrows.extend(cached);print("RESUMED_INDEX",key,len(cached),flush=True);continue
 source_rows=[]
 for sp in ['train','valid','val','test']:
  for p in sorted((root/sp/'images').glob('*')):
   if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}:continue
   rid=hashlib.sha256((key+'/'+str(p.relative_to(root))).encode()).hexdigest()[:20]
   row={'id':rid,'source_id':key,'source_workspace':ws,'source_split':sp,'source_image':str(p),'source_relative':str(p.relative_to(root)),'source_url':source['url'],'license':source['license_reported'],'original_photo_origin':'unknown','sha256':sha(p),'width':0,'height':0,'exif_orientation':1,'issues':[],'source_classes':[],'target_ids':[],'boxes':[],'phash':[],'review_status':'pending_visual','semantic_reviewed':False}
   lab=root/sp/'labels'/(p.stem+'.txt');row['source_label']=str(lab)
   try:
    with Image.open(p) as im:
     im.load();row['width'],row['height']=im.size;row['exif_orientation']=im.getexif().get(274,1)
     if row['exif_orientation']!=1:row['issues'].append('exif_orientation_requires_label_transform')
     if min(im.size)<80:row['issues'].append('low_resolution_review')
   except Exception as e:row['issues'].append('corrupt_image:'+type(e).__name__)
   if not lab.exists():row['issues'].append('missing_label')
   else:
    for li,line in enumerate(lab.read_text(errors='replace').splitlines(),1):
     if not line.strip():continue
     try:
      v=list(map(float,line.split()));assert all(math.isfinite(a) for a in v)
      assert v[0].is_integer() and int(v[0]) in names
      cid=int(v[0]);coords=v[1:];name=names[cid]
      row['source_classes'].append(name)
      target=LOOKUP.get(norm(name))
      if target:row['target_ids'].append(target)
      assert all(0<=a<=1 for a in coords)
      converted=False
      if len(coords)==4:x,y,w,h=coords
      else:
       assert len(coords)>=6 and len(coords)%2==0
       xs,ys=coords[::2],coords[1::2];w=max(xs)-min(xs);h=max(ys)-min(ys);x=(min(xs)+max(xs))/2;y=(min(ys)+max(ys))/2;converted=True
      assert w>0 and h>0
      assert x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6
      box={'source_class_id':cid,'source_class':name,'target_id':target,'xywh':[x,y,w,h],'polygon_to_box':converted,'source_line':li}
      row['boxes'].append(box)
     except (AssertionError,ValueError,IndexError):row['issues'].append('invalid_label_line:'+str(li))
   row['target_ids']=sorted(set(row['target_ids']));row['source_classes']=sorted(set(row['source_classes']))
   row['label_signature']=hashlib.sha256(json.dumps(sorted([(b['target_id'],b['xywh']) for b in row['boxes'] if b['target_id']]),sort_keys=True).encode()).hexdigest()
   fatal=any(s!='low_resolution_review' for s in row['issues'])
   if fatal:row['review_status']='needs_file_or_box_repair'
   elif not row['target_ids']:row['review_status']='outside_target_taxonomy'
   elif any(t in {3,9,10,12,15,16,17,20,22,23,25,26} for t in row['target_ids']):row['review_status']='needs_type_and_box_review'
   if row['target_ids'] and not fatal:
    with Image.open(p) as im:
     im=im.convert('RGB');row['phash']=[str(imagehash.phash(im)),str(imagehash.phash(ImageOps.mirror(im)))]
   row['automated_valid']=bool(row['target_ids']) and not fatal
   source_rows.append(row)
 allrows.extend(source_rows)
 with (ROOT/'manifests'/(key+'.jsonl')).open('w') as f:
  for r in source_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 status('source_indexed',source=key,source_images=len(source_rows),total_indexed=len(allrows))
dump(ROOT/'manifests/source_class_mapping.json',mappings)
# Union groups prevent detected duplicates entering different splits. Near matches are not claimed to be identical.
parent=list(range(len(allrows)))
def find(x):
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[max(a,b)]=min(a,b)
class BK:
 def __init__(self):self.root=None
 def add(self,h,i):
  if self.root is None:self.root=[h,i,{}];return
  n=self.root
  while True:
   d=(n[0]^h).bit_count()
   if d==0:return
   if d not in n[2]:n[2][d]=[h,i,{}];return
   n=n[2][d]
 def query(self,h,t=4):
  stack=[self.root] if self.root else [];out=[]
  while stack:
   n=stack.pop();d=(n[0]^h).bit_count()
   if d<=t:out.append((n[1],d))
   stack.extend(c for k,c in n[2].items() if d-t<=k<=d+t)
  return out
exact={}; filenames={};tree=BK();edges=[];hash_labels=collections.defaultdict(set)
for i,r in enumerate(allrows):
 h=r['sha256'];hash_labels[h].add(r['label_signature'])
 if h in exact:union(i,exact[h]);r['exact_duplicate_of']=allrows[exact[h]]['id']
 else:exact[h]=i;r['exact_duplicate_of']=''
 # Within-source file groups only; generic filenames across publishers do not establish identity.
 stem=Path(r['source_image']).stem.split('.rf.')[0];stem=re.sub(r'[-_]mirrored','',stem,flags=re.I).lower();fk=(r['source_id'],stem)
 if fk in filenames:union(i,filenames[fk])
 else:filenames[fk]=i
 for hv in r['phash']:
  hval=int(hv,16)
  for j,d in tree.query(hval):
   union(i,j)
   if len(edges)<200000:edges.append({'a':r['id'],'b':allrows[j]['id'],'distance':d,'kind':'phash_or_mirror_candidate'})
  tree.add(hval,i)
 if i%5000==0:print('GROUPING',i,len(allrows),flush=True)
groups=collections.defaultdict(list)
for i,r in enumerate(allrows):groups[find(i)].append(r)
for rs in groups.values():
 gid=min(r['id'] for r in rs);v=int(hashlib.sha256(('1398:'+gid).encode()).hexdigest()[:8],16)%100
 sp='train' if v<80 else ('val' if v<90 else 'test')
 for r in rs:r['duplicate_group']=gid;r['proposed_split']=sp;r['group_images']=len(rs)
 for r in rs:
  if len(hash_labels[r['sha256']])>1:r['issues'].append('same_image_conflicting_annotations');r['review_status']='needs_annotation_conflict_review'
# One technically valid representative per conservative group and target combination. Not an independent-original guarantee.
representatives={}
for r in allrows:
 if not r['automated_valid']:continue
 key=(r['duplicate_group'],tuple(r['target_ids']))
 score=(len(r['issues'])==0,r['source_split']!='train',r['width']*r['height'],r['id'])
 if key not in representatives or score>representatives[key][0]:representatives[key]=(score,r)
selected={r['id'] for _,r in representatives.values()}
for r in allrows:r['selected_for_review']=r['id'] in selected
with (ROOT/'manifests/images.jsonl').open('w') as f:
 for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
csvout(ROOT/'manifests/review_queue.csv',[{**r,'target_ids':','.join(map(str,r['target_ids'])),'issues':';'.join(r['issues'])} for r in allrows if r['selected_for_review']],['id','source_workspace','source_relative','target_ids','duplicate_group','proposed_split','review_status','issues','source_url','license'])
csvout(ROOT/'manifests/near_duplicate_candidates.csv',edges,['a','b','distance','kind'])
counts=[]
for c in registry:
 n=c['center_id'];rr=[r for r in allrows if n in r['target_ids']];ok=[r for r in rr if r['automated_valid']];sel=[r for r in ok if r['selected_for_review']]
 counts.append({**c,'candidate_files':len(rr),'technical_valid_files':len(ok),'unique_sha256':len({r['sha256'] for r in ok}),'conservative_groups':len({r['duplicate_group'] for r in ok}),'review_representatives':len(sel),'visually_approved':0,'train':0,'val':0,'test':0})
csvout(ROOT/'reports/class_counts.csv',counts,list(counts[0]));dump(ROOT/'reports/class_counts.json',counts)
# Compact review pack retains image and converted proposals; canonical training output is gated by explicit review decisions.
pack=WORK/'review_pack';(pack/'images').mkdir(parents=True,exist_ok=True);(pack/'labels').mkdir(exist_ok=True);(pack/'contact_sheets').mkdir(exist_ok=True)
for r in allrows:
 if not r['selected_for_review']:continue
 shutil.copyfile(r['source_image'],pack/'images'/(r['id']+Path(r['source_image']).suffix.lower()))
 (pack/'labels'/(r['id']+'.txt')).write_text('\n'.join(str(b['target_id']-1)+' '+' '.join(f'{v:.8f}' for v in b['xywh']) for b in r['boxes'] if b['target_id'])+'\n')
# Sheets use persistent ids and all target boxes. No visual approval is inferred from generation.
sheet_index=[]
for cid in [2,4,25,17]+[i for i in range(1,27) if i not in [2,4,25,17]]:
 rr=[r for r in allrows if r['selected_for_review'] and cid in r['target_ids']]
 for start in range(0,len(rr),24):
  subset=rr[start:start+24];canvas=Image.new('RGB',(1440,1680),'#eeeeee');draw=ImageDraw.Draw(canvas)
  for pos,r in enumerate(subset):
   im=Image.open(r['source_image']).convert('RGB');d=ImageDraw.Draw(im);w,h=im.size
   for b in r['boxes']:
    if b['target_id']:
     x,y,bw,bh=b['xywh'];d.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='#ef2222',width=max(2,w//200))
   im.thumbnail((350,240));xx=(pos%4)*360;yy=(pos//4)*280;canvas.paste(im,(xx,yy));draw.text((xx+3,yy+241),f'{start+pos:04d} {r["id"][:10]}',fill='black');draw.text((xx+3,yy+258),r['source_workspace'],fill='black')
  name=f'{cid:02d}_{NAMES[cid-1]}_{start//24:03d}.jpg';canvas.save(pack/'contact_sheets'/name,quality=90)
  sheet_index.append({'file':name,'center_id':cid,'ids':[r['id'] for r in subset]})
dump(pack/'sheet_index.json',sheet_index)
shutil.copyfile(ROOT/'manifests/review_queue.csv',pack/'review_queue.csv');shutil.copyfile(ROOT/'manifests/class_registry.json',pack/'class_registry.json')
(pack/'README.txt').write_text('UNREVIEWED CANDIDATES. Labels use center_id-1 (26 registry slots). Do not train until reviewed. Images are one representative per conservative duplicate group and candidate class combination. Original photos and all annotations remain in raw_sources archives. No independent-original count or center accuracy is asserted.\n')
archive_path=Path(shutil.make_archive(str(WORK/'review_pack_v1'),'zip',pack));shutil.copyfile(archive_path,ROOT/'review/review_pack_v1.zip');assert sha(archive_path)==sha(ROOT/'review/review_pack_v1.zip')
shutil.copyfile(pack/'sheet_index.json',ROOT/'review/sheet_index.json')
(ROOT/'yolo/README.txt').write_text('Only explicitly visually approved images will be exported here. Pending images are in review/review_pack_v1.zip. The original 5-class model/dataset remains separate.\n')
summary={'raw_sources':len(sources),'raw_images':len(allrows),'raw_archive_bytes':sum(s['archive_bytes'] for s in sources),'raw_unique_sha256':len(exact),'technical_valid_candidate_images':sum(r['automated_valid'] for r in allrows),'review_representative_images':len(selected),'duplicate_groups_all':len(groups),'candidate_target_classes':sum(c['candidate_files']>0 for c in counts),'candidate_additional_classes':sum(c['candidate_files']>0 and c['center_id'] not in BASE for c in counts),'visually_approved_images':0,'training_images_exported':0,'raw_archive_integrity':'ZIP CRC plus SHA256 copy verification passed','review_pack_sha256':sha(ROOT/'review/review_pack_v1.zip'),'limitations':['No automatic semantic approval','pHash/mirror and filename grouping do not prove independent capture sessions','No pretrained model training performed','Mixed classes and missing labels require per-image review','Center shape equivalence unverified','Existing 5-class evaluation split is not reused']}
dump(ROOT/'reports/automated_build_summary.json',summary)
status('automated_build_completed_visual_review_pending',**summary)
print('REVIEW_PACK_LOCAL',str(pack),flush=True)
