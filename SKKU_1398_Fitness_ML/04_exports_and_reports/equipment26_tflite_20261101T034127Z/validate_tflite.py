
from pathlib import Path
import sys,json,hashlib,zipfile,shutil,time
import numpy as np,torch,cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ai_edge_litert.interpreter import Interpreter
torch.set_num_threads(2)
out=Path(sys.argv[1]); work=Path('/content/equipment26_tflite_export')
f=out/'equipment26_fp32.tflite'
interpreter=Interpreter(model_path=str(f),num_threads=2)
interpreter.allocate_tensors()
def describe(d):
    return {'name':d['name'],'index':int(d['index']),'shape':d['shape'].tolist(),'dtype':np.dtype(d['dtype']).name,'quantization':[float(d['quantization'][0]),int(d['quantization'][1])]}
inputs=interpreter.get_input_details(); outputs=interpreter.get_output_details()
spec={'inputs':[describe(d) for d in inputs],'outputs':[describe(d) for d in outputs],'nms_included':False}
(out/'input_output_spec.json').write_text(json.dumps(spec,indent=2))
print('TENSOR_SPEC',json.dumps(spec),flush=True)
assert tuple(inputs[0]['shape'])==(1,3,640,640)
assert tuple(outputs[0]['shape'])==(1,30,8400)
assert inputs[0]['dtype']==np.float32
pt=YOLO(str(work/'equipment26_fp32.pt'))
pt.model.float().eval()
images=sorted(Path('/content/equipment26_min100_v2/images/test').glob('*'))
assert len(images)==205
checks=[]
for p in images[:5]:
    im=cv2.imread(str(p)); im=LetterBox(new_shape=(640,640),auto=False)(image=im)
    x=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.0
    interpreter.set_tensor(inputs[0]['index'],x); interpreter.invoke(); actual=interpreter.get_tensor(outputs[0]['index'])
    with torch.inference_mode(): expected=pt.model(torch.from_numpy(x))
    if isinstance(expected,(tuple,list)): expected=expected[0]
    expected=expected.numpy(); expected[:,:4,:]/=640.0
    checks.append({'image':p.name,'max_abs_error':float(np.max(np.abs(actual-expected))),'mean_abs_error':float(np.mean(np.abs(actual-expected)))})
print('RAW_PARITY',json.dumps(checks),flush=True)
report={'test_images':205,'raw_output_comparison':checks,'device':'Colab CPU; not a phone','validation_settings':{'imgsz':640,'batch':1,'rect':False,'conf':0.001,'iou':0.7},'models':{}}
for name,path in [('pytorch',work/'equipment26_fp32.pt'),('tflite',f)]:
    print('EVALUATE',name,flush=True)
    m=YOLO(str(path),task='detect').val(data='/content/equipment26_min100_v2/data_train.yaml',split='test',imgsz=640,batch=1,rect=False,device='cpu',workers=0,plots=False,save_json=False,verbose=False,conf=0.001,iou=0.7,project=str(work/'validation'),name=name,exist_ok=True)
    report['models'][name]={'precision':float(m.box.mp),'recall':float(m.box.mr),'mAP50':float(m.box.map50),'mAP50_95':float(m.box.map),'speed_ms':{k:float(v) for k,v in m.speed.items()}}
    (out/'validation_report.json').write_text(json.dumps(report,indent=2))
    print('METRICS',name,json.dumps(report['models'][name]),flush=True)
print('VALIDATION_COMPLETE',json.dumps(report),flush=True)
