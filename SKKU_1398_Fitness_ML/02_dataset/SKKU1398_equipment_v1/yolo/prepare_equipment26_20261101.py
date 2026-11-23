"""Reconstruct the updated 26-class dataset without another large Drive ZIP."""
import argparse,json,zipfile,shutil,subprocess,sys,hashlib
from pathlib import Path
DRIVE=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML/02_dataset/SKKU1398_equipment_v1/yolo')
def main():
 p=argparse.ArgumentParser();p.add_argument('--base-zip',type=Path,default=DRIVE/'equipment24_expanded_min100_20261031.zip');p.add_argument('--delta-zip',type=Path,default=DRIVE/'equipment26_additions_labeled_20261101.zip');p.add_argument('--output',type=Path,default=Path('/content/equipment26_20261101'));a=p.parse_args()
 h=hashlib.sha256()
 with a.base_zip.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()=='587491a3e2d6ea98310f9c7de489a12b3a139c94de74b765499cd77e92a24f41','Wrong 24-class base archive'
 if a.output.exists() and any(a.output.iterdir()):raise SystemExit('Output must be empty; existing data is preserved. Choose a new --output directory.')
 a.output.mkdir(parents=True,exist_ok=True)
 for archive,base_only in [(a.base_zip,True),(a.delta_zip,False)]:
  with zipfile.ZipFile(archive) as z:
   for item in z.infolist():
    parts=Path(item.filename).parts
    if item.is_dir() or len(parts)<2:continue
    if not base_only and parts[0]!='equipment26_20261101':continue
    rel=Path(*parts[1:])
    if base_only and rel.parts[0] not in ('images','labels'):continue
    dest=a.output/rel
    if not dest.resolve().is_relative_to(a.output.resolve()):raise ValueError('Unsafe archive member')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with z.open(item) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
 subprocess.run([sys.executable,str(a.output/'verify_dataset.py')],check=True)
 print('Dataset ready:',a.output.resolve())
 print('Training has not been started. Existing model weights are unchanged.')
if __name__=='__main__':main()
