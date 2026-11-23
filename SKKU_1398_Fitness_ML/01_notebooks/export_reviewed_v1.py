"""Export only recorded image-level visual decisions. Pending and rejected images cannot enter training."""
from pathlib import Path
import json, collections, shutil, hashlib, math, zipfile, time
import yaml
from PIL import Image
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML/02_dataset/SKKU1398_equipment_v1')
WORK=Path('/content/SKKU1398_equipment_v1_work')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows={r['id']:r for r in map(json.loads,(ROOT/'manifests/images.jsonl').read_text().splitlines())}
 decisions=json.loads((ROOT/'review/visual_decisions.json').read_text())
 registry=json.loads((ROOT/'manifests/class_registry.json').read_text());names={r['center_id']:r['name'] for r in registry}
 latest={}
 for d in decisions:
  assert d['id'] in rows
  assert d['decision'] in ['approve','hold','reject']
  assert d.get('reviewer') and d.get('evidence') and d.get('reason')
  latest[d['id']]=d
 accepted=[]
 for rid,d in latest.items():
  if d['decision']!='approve':continue
  r=rows[rid]
  assert r['automated_valid'] and r['selected_for_review']
  assert d.get('all_visible_target_equipment_checked') is True
  assert d.get('boxes_checked') is True
  boxes=d.get('corrected_boxes',[{'target_id':b['target_id'],'xywh':b['xywh']} for b in r['boxes'] if b['target_id']])
  assert boxes
  for b in boxes:
   assert b['target_id'] in names and b['target_id'] not in {19,24}
   x,y,w,h=b['xywh'];assert all(math.isfinite(v) for v in [x,y,w,h]) and w>0 and h>0
   assert x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6
  accepted.append((r,boxes,d))
 assert accepted, 'No visually approved images; refusing an empty training export'
 active=sorted({b['target_id'] for _,bb,_ in accepted for b in bb});remap={cid:i for i,cid in enumerate(active)}
 out=WORK/'approved_yolo_v1'
 if out.exists():raise RuntimeError('Output exists; create a new version to preserve prior export')
 for sp in ['train','val','test']:
  for kind in ['images','labels']:(out/sp/kind).mkdir(parents=True,exist_ok=True)
 frozen=[];unique=set();group_split={};counts=collections.defaultdict(collections.Counter)
 for r,boxes,d in accepted:
  if r['sha256'] in unique:continue
  unique.add(r['sha256']);sp=r['proposed_split'];g=r['duplicate_group']
  assert group_split.setdefault(g,sp)==sp
  source=Path(r['source_image']);suffix=source.suffix.lower();target=out/sp/'images'/(r['id']+suffix)
  shutil.copyfile(source,target);assert sha(target)==r['sha256']
  (out/sp/'labels'/(r['id']+'.txt')).write_text('\n'.join(str(remap[b['target_id']])+' '+' '.join(f'{v:.8f}' for v in b['xywh']) for b in boxes)+'\n')
  for cid in {b['target_id'] for b in boxes}:counts[cid][sp]+=1
  frozen.append({'id':r['id'],'image':str(target.relative_to(out)),'sha256':r['sha256'],'duplicate_group':g,'split':sp,'source_url':r['source_url'],'source_image':r['source_relative'],'license':r['license'],'source_id':r['source_id'],'original_photo_origin':'unknown','review':d})
 (out/'data.yaml').write_text(yaml.safe_dump({'path':'.','train':'train/images','val':'val/images','test':'test/images','names':{remap[c]:names[c] for c in active}},sort_keys=False))
 (out/'class_mapping.json').write_text(json.dumps([{'yolo_id':remap[c],'center_id':c,'name':names[c]} for c in active],indent=2))
 (out/'manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
 (out/'README.txt').write_text('Visually reviewed public-data pilot. Use an absolute path in data.yaml after extraction, or pass a copy with path set to the extracted directory. Not center validation. All source archives and image provenance are preserved. pHash/filename groups reduce but do not prove absence of scene leakage. Only active classes are listed; this is not a 26-class completed dataset. Do not combine with old holdout without regrouping.\n')
 # Validate exported bytes, label ids, and duplicate-group/hash split isolation.
 for sp in ['train','val','test']:
  imgs=list((out/sp/'images').glob('*'));labs=list((out/sp/'labels').glob('*.txt'));assert len(imgs)==len(labs)
  for p in imgs:
   with Image.open(p) as im:im.verify()
   for line in (out/sp/'labels'/(p.stem+'.txt')).read_text().splitlines():
    v=line.split();assert len(v)==5 and 0<=int(v[0])<len(active)
 archive=Path(shutil.make_archive(str(WORK/'approved_yolo_v1'),'zip',out))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 dest=ROOT/'yolo/approved_yolo_v1.zip';assert not dest.exists();shutil.copyfile(archive,dest);assert sha(dest)==sha(archive)
 summary={'exported_images':len(frozen),'active_classes':len(active),'class_counts':{names[k]:dict(v) for k,v in counts.items()},'splits':dict(collections.Counter(r['split'] for r in frozen)),'reviewed_images':len(latest),'decisions':dict(collections.Counter(d['decision'] for d in latest.values())),'duplicate_group_cross_split':0,'exact_hash_cross_split':0,'zip_sha256':sha(dest),'center_validation':False,'training_run':False}
 (ROOT/'reports/reviewed_export_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 shutil.copyfile(out/'manifest.json',ROOT/'manifests/approved_manifest.json')
 shutil.copyfile(out/'class_mapping.json',ROOT/'yolo/class_mapping.json')
 # Keep automated source records immutable and publish a current decision overlay.
 exported_ids={x['id'] for x in frozen}
 with (ROOT/'manifests/review_state.jsonl').open('w') as f:
  for rid,r in rows.items():
   d=latest.get(rid)
   f.write(json.dumps({'id':rid,'review_status_current':d['decision'] if d else r['review_status'],'decision':d,'exported':rid in exported_ids},ensure_ascii=False)+'\n')
 class_table=json.loads((ROOT/'reports/class_counts.json').read_text())
 for c in class_table:
  cid=c['center_id'];c['visually_approved']=sum(counts[cid].values())
  for sp in ['train','val','test']:c[sp]=counts[cid][sp]
  c['reviewed_representatives']=sum(cid in rows[rid]['target_ids'] for rid in latest)
 (ROOT/'reports/class_counts_with_review.json').write_text(json.dumps(class_table,ensure_ascii=False,indent=2))
 import csv
 with (ROOT/'reports/class_counts_with_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(class_table[0]));w.writeheader();w.writerows(class_table)
 with (ROOT/'README_KO.txt').open('a',encoding='utf-8') as f:
  f.write('\n최신 시각 검수 및 학습셋 수량: reports/reviewed_export_summary.json 및 class_counts_with_review.csv 참조. 승인본: yolo/approved_yolo_v1.zip. 미검수 자료는 학습 승인본에 포함되지 않습니다.\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
