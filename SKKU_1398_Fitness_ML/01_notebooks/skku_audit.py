from pathlib import Path
import json, csv, shutil, random, hashlib, math
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageOps
import yaml, imagehash
import skku1398_ml as ml
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML')
WORK=Path('/content/skku1398_workspace')
SRC=WORK/'raw_source/fitfuel_v2'
OUT=WORK/'grouped_source_v1'
REPORT=ROOT/'02_dataset/fitfuel_v2_grouped_v1'
REPORT.mkdir(parents=True,exist_ok=True)
names=ml.parse_class_names(yaml.safe_load((SRC/'data.yaml').read_text())['names'])
mapping=ml.source_to_target(names)
rows=[]; issues=[]; negatives=[]; raw_counts={}
for split in ['train','valid','test']:
    cnt=Counter()
    for img in sorted((SRC/split/'images').glob('*')):
        lab=SRC/split/'labels'/(img.stem+'.txt')
        for line in lab.read_text().splitlines():
            if line.strip(): cnt[names[int(line.split()[0])]]+=1
        boxes,errs=ml.read_yolo_boxes(lab,mapping)
        if errs:
            issues.append({'image':str(img),'issues':errs}); continue
        if not boxes:
            negatives.append(img); continue
        im=ImageOps.exif_transpose(Image.open(img)).convert('RGB')
        hashes=[int(str(imagehash.phash(im.rotate(a,expand=True))),16) for a in [0,90,180,270]]
        pixels=hashlib.sha256(im.tobytes()).hexdigest()
        rows.append({'image':str(img),'source_split':split,'stem':img.stem.split('.rf.')[0], 'boxes':boxes,'hashes':hashes,'pixel_sha':pixels})
    raw_counts[split]=dict(cnt)
print('POSITIVE_IMAGES',len(rows),'INVALID_IMAGES',len(issues),'NEGATIVE_IMAGES',len(negatives),flush=True)
# Conservative union: same original filename OR rotation-invariant near-duplicate pHash.
# Generic filename collisions may over-group; keeping related material together takes priority.
parent=list(range(len(rows)))
def find(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]];i=parent[i]
    return i
def union(i,j):
    a,b=find(i),find(j)
    if a!=b:parent[b]=a
links=[]
for i,a in enumerate(rows):
    for j in range(i):
        b=rows[j]
        same=a['stem']==b['stem']
        distance=min((h^k).bit_count() for h in a['hashes'] for k in b['hashes'])
        if same or distance<=4:
            union(i,j)
            links.append({'a':a['image'],'b':b['image'],'reason':'filename' if same else 'rotation_phash','distance':distance})
groups=defaultdict(list)
for i in range(len(rows)):groups[find(i)].append(i)
group_ids=sorted(groups)
G=np.array([[sum(sum(b[0]==c for b in rows[i]['boxes']) for i in groups[g]) for c in range(5)] for g in group_ids])
total=G.sum(0)
rng=np.random.default_rng(42)
best=None
for attempt in range(10000):
    assign=rng.choice(3,size=len(group_ids),p=[.7,.15,.15])
    counts=np.array([G[assign==s].sum(0) for s in range(3)])
    if (counts<5).any():continue
    score=float(((counts/total-np.array([.7,.15,.15])[:,None])**2).sum())
    if best is None or score<best[0]:best=(score,assign.copy(),counts)
if best is None:raise RuntimeError('Cannot obtain all-class held-out split under conservative grouping')
assignment={g:['train','val','test'][int(s)] for g,s in zip(group_ids,best[1])}
if OUT.exists():raise FileExistsError('Grouped split already exists; do not overwrite frozen split')
manifest=[]
for g,indices in groups.items():
    sp=assignment[g]
    for i in indices:
        r=rows[i];img=Path(r['image']);stem=hashlib.sha256(str(img).encode()).hexdigest()[:16]
        dst=OUT/sp/'images'/(stem+img.suffix);dst.parent.mkdir(parents=True,exist_ok=True)
        label=OUT/sp/'labels'/(stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(img,dst)
        label.write_text('\n'.join(' '.join(map(str,b)) for b in r['boxes'])+'\n')
        manifest.append({'group':g,'split':sp,'source':str(img),'source_split':r['source_split'],'file':str(dst.relative_to(OUT)),'classes':sorted(set(b[0] for b in r['boxes']))})
# Negative scenes stay in an auxiliary pool, excluded from this first controlled baseline.
(OUT/'data.yaml').write_text(yaml.safe_dump({'path':str(OUT),'train':'train/images','val':'val/images','test':'test/images','names':ml.CLASS_NAMES}))
PREP=WORK/'prepared/grouped_v1'
report=ml.prepare_yolo_dataset(OUT,PREP,mode='baseline',seed=42)
report.update({'source_url':'https://universe.roboflow.com/fitfuel/all-gym-equipment/dataset/2','license':'CC BY 4.0','attribution':'FitFuel, All Gym Equipment v2, Roboflow Universe, 2024; filtered/remapped and regrouped in this project','official_source_counts':raw_counts,'source_class_count':len(names),'grouping':'Same pre-.rf filename or rotation-invariant pHash distance <=4; seed 42; frozen before training','groups':len(groups),'largest_group_images':max(map(len,groups.values())),'split_selection_label_only_objective':best[0],'negative_images_excluded':len(negatives),'invalid_images_excluded':len(issues),'scope':'New grouped holdout of public proxy dataset, NOT official test benchmark, NOT center validation','independence_limit':'Conservative grouping is not proof of different photographers, source websites or gym sessions. Augmented source images remain; residual similarity can exist.'})
(REPORT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(REPORT/'frozen_split_manifest.json').write_text(json.dumps(manifest,indent=2))
(REPORT/'group_links.json').write_text(json.dumps(links,indent=2))
(REPORT/'excluded_label_issues.json').write_text(json.dumps(issues,indent=2))
shutil.copytree(PREP/'reports',REPORT/'prepared_audit',dirs_exist_ok=True)
archive=shutil.make_archive(str(WORK/'fitfuel_v2_original'),'zip',SRC)
shutil.copy2(archive,REPORT/'fitfuel_v2_original.zip')
archive=shutil.make_archive(str(WORK/'five_class_grouped_v1'),'zip',PREP)
shutil.copy2(archive,REPORT/'five_class_grouped_v1.zip')
for sp in ['train','val','test']:
    # class-balanced label previews, no model selection based on held-out images.
    ims=list((PREP/sp/'images').glob('*'));chosen=[]
    for c in range(5):
        opts=[p for p in ims if any(int(l.split()[0])==c for l in (PREP/sp/'labels'/(p.stem+'.txt')).read_text().splitlines() if l.strip())]
        if opts:chosen.append(opts[0])
    from PIL import ImageDraw
    canvas=Image.new('RGB',(1250,300),'white')
    for k,p in enumerate(chosen):
        im=Image.open(p).convert('RGB');d=ImageDraw.Draw(im);w,h=im.size
        for l in (PREP/sp/'labels'/(p.stem+'.txt')).read_text().splitlines():
            c,x,y,bw,bh=map(float,l.split());xy=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
            d.rectangle(xy,outline='red',width=4);d.text((max(0,xy[0]),max(0,xy[1])),ml.CLASS_NAMES[int(c)],fill='red')
        im.thumbnail((250,260));canvas.paste(im,(k*250,20));ImageDraw.Draw(canvas).text((k*250,280),ml.CLASS_NAMES[k],fill='black')
    canvas.save(REPORT/(sp+'_label_review.jpg'))
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
print('DATA_READY',str(PREP),flush=True)
