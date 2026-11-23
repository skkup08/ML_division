
from pathlib import Path
import json, shutil, hashlib, sys
from ultralytics import YOLO
import ultralytics, torch
work=Path('/content/equipment26_tflite_export')
out=Path(sys.argv[1])
model=YOLO(str(work/'equipment26_fp32.pt'))
assert len(model.names)==26
print('EXPORT_START',ultralytics.__version__,torch.__version__,flush=True)
exported=Path(model.export(format='litert',imgsz=640,batch=1,device='cpu',nms=False))
assert exported.is_file() and exported.suffix=='.tflite'
target=out/'equipment26_fp32.tflite'
shutil.copy2(exported,target)
(out/'labels.txt').write_text(chr(10).join(model.names[i] for i in range(26))+chr(10),encoding='utf-8')
report={'model':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'ultralytics':ultralytics.__version__,'torch':torch.__version__,'source_sha256':'7af2ceb79edad610f40369bdac110f2aeb80fae40c2921ec95a70eb081bc585b','nms_included':False}
(out/'conversion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('TFLITE_SAVED',str(target),json.dumps(report),flush=True)
