"""Reconstruct all 26 classes from the existing Drive archives plus this small update."""
import argparse,hashlib,shutil,subprocess,sys,zipfile
from pathlib import Path
DRIVE=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML/02_dataset/SKKU1398_equipment_v1/yolo')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--base-zip',type=Path,default=DRIVE/'equipment24_expanded_min100_20261031.zip');p.add_argument('--previous-zip',type=Path,default=DRIVE/'equipment26_additions_labeled_20261101.zip');p.add_argument('--update-zip',type=Path,default=DRIVE/'equipment26_min100_v2_update.zip');p.add_argument('--output',type=Path,default=Path('/content/equipment26_min100_v2'));a=p.parse_args()
 for f,h in [(a.base_zip,'587491a3e2d6ea98310f9c7de489a12b3a139c94de74b765499cd77e92a24f41'),(a.previous_zip,'7357a197939ab83a600b1a8bffac2031ae131b929335af65cef6631e1fc983b5')]:
  if not f.is_file():raise SystemExit(f'Missing required archive: {f}')
  if sha(f)!=h:raise SystemExit(f'Unexpected archive contents: {f}')
 if not a.update_zip.is_file():raise SystemExit(f'Missing update: {a.update_zip}')
 if a.output.exists() and any(a.output.iterdir()):raise SystemExit('Output must be empty. Choose a new --output folder; existing data is preserved.')
 a.output.mkdir(parents=True,exist_ok=True)
 for archive,rootname,images_only in [(a.base_zip,None,True),(a.previous_zip,'equipment26_20261101',True),(a.update_zip,'equipment26_min100_v2',False)]:
  with zipfile.ZipFile(archive) as z:
   for item in z.infolist():
    parts=Path(item.filename).parts
    if item.is_dir() or len(parts)<2 or (rootname and parts[0]!=rootname):continue
    rel=Path(*parts[1:])
    if images_only and rel.parts[0] not in ('images','labels'):continue
    dest=a.output/rel
    if not dest.resolve().is_relative_to(a.output.resolve()):raise ValueError('Unsafe archive path')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with z.open(item) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
 subprocess.run([sys.executable,str(a.output/'verify_dataset.py')],check=True)
 print('Verified dataset:',a.output.resolve())
 print('YOLO data:',(a.output/'data.yaml').resolve())
 print('Model training has not been started by this script.')
if __name__=='__main__':main()
