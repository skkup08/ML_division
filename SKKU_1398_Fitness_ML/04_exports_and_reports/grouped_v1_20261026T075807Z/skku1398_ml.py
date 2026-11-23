"""SKKU 1398 Fitness Center: YOLO dataset preparation and audit utilities.

This code prepares a public *proxy* dataset for five provisional center classes.
It never asserts the public class names perfectly match the installed machines.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image, ImageDraw, ImageOps

CLASS_NAMES = [
    "treadmill",
    "seated_chest_press",
    "shoulder_press",
    "seated_leg_press",
    "smith_machine",
]

# The public names can be broader than the equipment listed by the center.
SOURCE_ALIASES = {
    "treadmill": ["Treadmill", "Tredmill"],
    "seated_chest_press": ["Chest Press Machine", "Chest Press", "Seated Chest Press"],
    "shoulder_press": ["Shoulder Press Machine", "Shoulder Press"],
    "seated_leg_press": ["Leg Press Machine", "Leg Press", "Seated Leg Press"],
    "smith_machine": ["Smith Machine", "Smith"],
}
MATCH_NOTE = {
    "treadmill": "generic treadmill vs center treadmill: appearance unverified",
    "seated_chest_press": "generic chest press may include other press configurations",
    "shoulder_press": "generic shoulder press vs exact installed machine unverified",
    "seated_leg_press": "generic leg press may include horizontal/plate-loaded variants",
    "smith_machine": "generic smith machine vs exact installed unit unverified",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def parse_class_names(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, dict):
        indexed = {int(k): str(v) for k, v in raw.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Class IDs in YAML must start at 0 and be contiguous")
        return [indexed[i] for i in range(len(indexed))]
    raise ValueError("data.yaml must have a list/dictionary called names")


def source_to_target(source_names: list[str]) -> dict[int, int]:
    alias_lookup = {}
    for target_id, canonical in enumerate(CLASS_NAMES):
        for alias in [canonical] + SOURCE_ALIASES[canonical]:
            norm = normalize_name(alias)
            if norm in alias_lookup and alias_lookup[norm] != target_id:
                raise ValueError(f"Ambiguous class alias {alias}")
            alias_lookup[norm] = target_id
    found = {}
    for src_id, name in enumerate(source_names):
        norm = normalize_name(name)
        if norm in alias_lookup:
            found[src_id] = alias_lookup[norm]
    return found


def read_yolo_boxes(label_file: Path, map_ids: dict[int, int]) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Return selected valid boxes and issue messages. Unmapped classes are ignored."""
    boxes, errors = [], []
    if not label_file.exists():
        return boxes, ["missing label file"]
    for line_num, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {line_num}: expected 5 values")
            continue
        try:
            fcls = float(fields[0])
            if not fcls.is_integer():
                raise ValueError("non-integer class ID")
            src_cls = int(fcls)
            vals = [float(v) for v in fields[1:]]
        except ValueError:
            errors.append(f"line {line_num}: failed to parse numeric box")
            continue
        if src_cls not in map_ids:
            continue
        cx, cy, bw, bh = vals
        if not all(map(lambda x: float('-inf') < x < float('inf'), vals)):
            errors.append(f"line {line_num}: NaN/Inf box")
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            errors.append(f"line {line_num}: values outside YOLO normalized range")
            continue
        if not (0 <= cx-bw/2 and cx+bw/2 <= 1 and 0 <= cy-bh/2 and cy+bh/2 <= 1):
            errors.append(f"line {line_num}: box crosses image boundary")
            continue
        boxes.append((map_ids[src_cls], cx, cy, bw, bh))
    return boxes, errors


def _candidate_image_dirs(root: Path, config: dict, split: str) -> list[Path]:
    cands = []
    cfgvalue = config.get(split)
    if split == 'val':
        cfgvalue = cfgvalue or config.get('valid') or config.get('validation')
    if cfgvalue:
        if isinstance(cfgvalue, list):
            if len(cfgvalue) != 1:
                raise ValueError(f"split {split}: multiple datasets not supported in baseline")
            cfgvalue = cfgvalue[0]
        candidate = Path(str(cfgvalue))
        if candidate.is_absolute():
            cands.append(candidate)
        else:
            cands.append(root / candidate)
            cands.append(root / Path(str(cfgvalue).replace("../", "")))
    for synonym in SPLIT_ALIASES[split]:
        cands += [root / synonym / "images", root / synonym]
    return cands


def detect_image_dir(root: Path, config: dict, split: str) -> Path | None:
    for candidate in _candidate_image_dirs(root, config, split):
        if candidate.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir() if p.is_file()):
            return candidate.resolve()
    return None


def find_yaml(root: Path) -> Path:
    direct = root / "data.yaml"
    if direct.exists():
        return direct
    found = [p for p in root.rglob("data.yaml") if len(p.relative_to(root).parts) <= 3]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one data.yaml within {root}; found {len(found)}")
    return found[0]


def _label_path(image_path: Path) -> Path:
    if image_path.parent.name.lower() == "images":
        return image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    return image_path.parent / "labels" / (image_path.stem + ".txt")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('L').resize((9,8))
        arr = list(image.get_flattened_data()) if hasattr(image, 'get_flattened_data') else list(image.getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | int(arr[y*9+x] > arr[y*9+x+1])
    return result


def _stratified_pick(rows: list[dict], limit: int | None, seed: int = 42) -> list[dict]:
    """Select examples with every available class represented before filling randomly."""
    if limit is None or len(rows) <= limit:
        return sorted(rows, key=lambda r: r['source_image'])
    rnd = random.Random(seed)
    order = rows.copy()
    rnd.shuffle(order)
    picked, seen = [], set()
    for class_id in range(len(CLASS_NAMES)):
        possible = [r for r in order if any(b[0] == class_id for b in r['boxes']) and r['source_image'] not in seen]
        if possible and len(picked) < limit:
            row = possible[0]
            picked.append(row)
            seen.add(row['source_image'])
    for row in order:
        if len(picked) >= limit:
            break
        if row['source_image'] not in seen:
            picked.append(row)
            seen.add(row['source_image'])
    return picked


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, '') for k in fields} for r in rows])


def prepare_yolo_dataset(source_root: Path | str, dest_root: Path | str, *,
                         mode: str = "smoke", max_train: int = 200,
                         max_val: int = 60, max_test: int = 40,
                         seed: int = 42) -> dict:
    """Prepare five-class YOLO data retaining source splits, auditing exact leakage.

    Training and validation may be augmented source images. A separate held-out
    center-photo evaluation is still required; this routine never claims otherwise.
    """
    source_root, dest_root = Path(source_root).resolve(), Path(dest_root).resolve()
    if mode not in {"smoke", "baseline"}:
        raise ValueError("mode must be 'smoke' or 'baseline'")
    config_path = find_yaml(source_root)
    src_root = config_path.parent
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    source_names = parse_class_names(config['names'])
    mapping = source_to_target(source_names)
    absent = [name for idx, name in enumerate(CLASS_NAMES) if idx not in set(mapping.values())]
    if absent:
        raise ValueError(f"Source dataset has no matching classes: {absent}; source names: {source_names}")
    if dest_root.exists():
        raise FileExistsError(f"Destination already exists: {dest_root}; choose a fresh directory")
    dest_root.mkdir(parents=True)
    (dest_root / 'reports').mkdir()
    manifest, issues, near = [], [], []
    counts = {}
    # Process more trustworthy held-out files before train to remove exact split leakage.
    seen_sha: dict[str, dict] = {}
    seen_hashes: list[tuple[int, str, str]] = []
    for split in ('test', 'val', 'train'):
        image_dir = detect_image_dir(src_root, config, split)
        if image_dir is None:
            if split == 'test':
                raise FileNotFoundError('No test/images found in source. Reserve an independent test split first.')
            raise FileNotFoundError(f'Missing {split}/images in {src_root}')
        all_rows, negatives = [], []
        for img in sorted(image_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            labels, errors = read_yolo_boxes(_label_path(img), mapping)
            if errors:
                issues.append({'split':split,'image':str(img),'issue':'; '.join(errors)})
            if labels:
                all_rows.append({'source_image':str(img),'boxes':labels})
            elif not errors and _label_path(img).exists():
                negatives.append({'source_image':str(img),'boxes':[]})
        if not all_rows:
            raise ValueError(f'No selected target boxes in split {split}')
        limit = {'train':max_train,'val':max_val,'test':max_test}[split] if mode=='smoke' else None
        pos_limit = max(5,int(limit*0.85)) if limit is not None else None
        chosen = _stratified_pick(all_rows, pos_limit, seed=seed + len(split))
        # Include a modest number of negative scenes containing only non-target classes.
        neg_limit = min(len(negatives), max(1, (limit or len(chosen))//8))
        random.Random(seed+42).shuffle(negatives)
        chosen += negatives[:neg_limit]
        rows_kept=0
        for row in chosen:
            img=Path(row['source_image'])
            sha=_sha256(img)
            if sha in seen_sha:
                issues.append({'split':split,'image':str(img),'issue':f'exact duplicate dropped; first in {seen_sha[sha]["split"]}'})
                continue
            # Validate the image, not only filename and annotation text.
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as ex:
                issues.append({'split':split,'image':str(img),'issue':f'image decode failed: {type(ex).__name__}'})
                continue
            prefix = hashlib.sha256(str(img.relative_to(src_root)).encode()).hexdigest()[:14]
            new_stem = prefix+'_'+img.stem[:50]
            out_image=dest_root/split/'images'/f'{new_stem}{img.suffix.lower()}'
            out_label=dest_root/split/'labels'/f'{new_stem}.txt'
            out_image.parent.mkdir(parents=True,exist_ok=True)
            out_label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(img,out_image)
            out_label.write_text(('\n'.join(' '.join([str(b[0])]+[f'{n:.6f}' for n in b[1:]]) for b in row['boxes'])+'\n') if row['boxes'] else '',encoding='utf-8')
            seen_sha[sha]={'split':split,'image':str(img)}
            h=dhash(img)
            seen_hashes.append((h,split,str(img)))
            rows_kept+=1
            manifest.append({'split':split,'image':str(out_image.relative_to(dest_root)),
                             'source_image':str(img),'sha256':sha,'dhash':f'{h:016x}',
                             'classes':','.join(CLASS_NAMES[c] for c in sorted(set(b[0] for b in row['boxes']))),
                             'box_count':len(row['boxes'])})
        counts[split]={'available_positive_images':len(all_rows),'available_negative_images':len(negatives),'selected_before_exact_dedup':len(chosen),'kept_images':rows_kept}
        if rows_kept<1:
            raise ValueError(f'No valid images kept for {split}')
    # Report suspicious near-identical images across splits (do NOT auto-delete).
    if len(seen_hashes)<=6000:
        for i,(a,sa,ia) in enumerate(seen_hashes):
            for b,sb,ib in seen_hashes[i+1:]:
                if sa==sb:
                    continue
                d=(a^b).bit_count()
                if d<=3:
                    near.append({'split_a':sa,'image_a':ia,'split_b':sb,'image_b':ib,'dhash_distance':d})
                    if len(near)>=250:
                        break
            if len(near)>=250:break
    _write_csv(dest_root/'reports'/'manifest.csv',manifest,['split','image','source_image','sha256','dhash','classes','box_count'])
    _write_csv(dest_root/'reports'/'label_issues.csv',issues,['split','image','issue'])
    _write_csv(dest_root/'reports'/'potential_near_duplicates.csv',near,['split_a','image_a','split_b','image_b','dhash_distance'])
    target_yaml={'path':str(dest_root),'train':'train/images','val':'val/images','test':'test/images',
                 'names':{k:v for k,v in enumerate(CLASS_NAMES)}}
    (dest_root/'data.yaml').write_text(yaml.safe_dump(target_yaml,allow_unicode=True,sort_keys=False),encoding='utf-8')
    class_boxes=defaultdict(Counter)
    for r in manifest:
        labels=(dest_root/Path(r['image']).parent.parent/'labels'/(Path(r['image']).stem+'.txt'))
        for line in labels.read_text(encoding='utf-8').splitlines():
            if line.strip():
                class_boxes[r['split']][CLASS_NAMES[int(line.split()[0])]]+=1
    summary={
        'classes':CLASS_NAMES,'source_root':str(src_root), 'source_classes':source_names,
        'id_mapping':{str(i):{'source_name':source_names[i], 'target':CLASS_NAMES[j]} for i,j in mapping.items()},
        'match_caveats':MATCH_NOTE, 'source_split_counts':counts,
        'box_counts_by_split':{s:dict(class_boxes[s]) for s in ('train','val','test')},
        'label_issue_count':len(issues),'near_duplicate_warning_count':len(near),
        'limitations':['Public 29-class labels are proxy labels, not installed-equipment confirmation',
                       'Near duplicates must be manually reviewed; hash check is not proof of independence',
                       'Evaluate separately on real SKKU 1398 photos from independent capture sessions']}
    (dest_root/'reports'/'dataset_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def make_annotation_preview(prepared_root: str|Path, output: str|Path, *,max_images:int=9, seed:int=42) -> Path:
    root=Path(prepared_root)
    images=list((root/'train'/'images').glob('*'))
    images=[p for p in images if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise FileNotFoundError('No prepared train images')
    images.sort()
    random.Random(seed).shuffle(images)
    images=images[:max_images]
    tiles=[]
    for img in images:
        with Image.open(img) as raw:
            im=ImageOps.exif_transpose(raw).convert('RGB')
        lab=root/'train'/'labels'/f'{img.stem}.txt'
        draw=ImageDraw.Draw(im)
        for line in lab.read_text().splitlines():
            c,x,y,w,h=map(float,line.split())
            W,H=im.size
            xy=[(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H]
            draw.rectangle(xy,outline=(255,168,17),width=max(2,W//180))
            draw.text((xy[0]+4,max(0,xy[1]-18)),CLASS_NAMES[int(c)],fill=(255,168,17))
        im.thumbnail((430,315))
        tile=Image.new('RGB',(450,345),'white')
        tile.paste(im,((450-im.width)//2,10))
        tiles.append(tile)
    cols=min(3,len(tiles))
    rows=(len(tiles)+cols-1)//cols
    collage=Image.new('RGB',(cols*450,rows*345),(244,244,244))
    for i,tile in enumerate(tiles):
        collage.paste(tile,((i%cols)*450,(i//cols)*345))
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    collage.save(out)
    return out


def predictions_to_json(result, *, threshold: float=0.25) -> dict:
    boxes=[]
    for box in result.boxes:
        score=float(box.conf.item())
        if score<threshold:
            continue
        cls=int(box.cls.item())
        x1,y1,x2,y2=[round(float(t),2) for t in box.xyxy[0].tolist()]
        boxes.append({'class_id':cls,'class_name':str(result.names[cls]),'confidence':round(score,4),
                      'bbox_xyxy_pixels':[x1,y1,x2,y2]})
    return {'status':'ok' if boxes else 'no_detections',
            'image_height':int(result.orig_shape[0]),'image_width':int(result.orig_shape[1]),
            'detections':boxes}
