from pathlib import Path
import json,shutil,sys,time,hashlib
import numpy as np,torch
from ultralytics import YOLO
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML')
RUN=Path((ROOT/'04_exports_and_reports/latest_run.txt').read_text().strip())
DEST=ROOT/'04_exports_and_reports'/RUN.name;DEST.mkdir(exist_ok=True)
status=json.loads((RUN/'status.json').read_text());status['mobile']='exporting';(RUN/'status.json').write_text(json.dumps(status,indent=2))
try:
    local=Path('/content/skku1398_mobile');local.mkdir(exist_ok=True)
    shutil.copy2(RUN/'baseline/weights/best.pt',local/'best.pt')
    model=YOLO(str(local/'best.pt'))
    exported=Path(model.export(format='litert',imgsz=640,batch=1,device='cpu'))
    assert exported.is_file() and exported.stat().st_size>0
    shutil.copy2(exported,DEST/'skku1398_yolo11n_fp32.tflite')
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=str(exported),num_threads=2);it.allocate_tensors()
    inp=it.get_input_details();out=it.get_output_details()
    def detail(x):return {'name':x['name'],'shape':x['shape'].tolist(),'dtype':str(x['dtype']),'quantization':list(x['quantization'])}
    contract={'inputs':[detail(x) for x in inp],'outputs':[detail(x) for x in out],'classes':model.names,'preprocessing':'EXIF upright RGB, letterbox 640x640 padding 114, float32 /255; use actual input tensor layout','scope':'public_proxy_not_center','nms':'external class-aware NMS; confidence 0.25, IoU 0.7','flutter_device_tested':False}
    sys.path.insert(0,str(ROOT/'01_notebooks'))
    from skku_inference import preprocess,decode,predict_photo
    samples=sorted(Path('/content/skku1398_workspace/prepared/grouped_v1/test/images').glob('*'))[:3]
    x,meta=preprocess(samples[0]);feed=x if tuple(inp[0]['shape'])==(1,3,640,640) else x.transpose(0,2,3,1).copy()
    it.set_tensor(inp[0]['index'],feed);it.invoke();raw=it.get_tensor(out[0]['index'])
    reference=YOLO(str(local/'best.pt')).model.eval().float()
    with torch.no_grad():pt=reference(torch.from_numpy(x))[0].numpy()
    # Compare coordinate units against same PyTorch raw output, without inferring from example scores alone.
    direct=float(np.mean(np.abs(raw[:,:4]-pt[:,:4])))
    norm=float(np.mean(np.abs(raw[:,:4]*640-pt[:,:4])))
    normalized=norm<direct
    adjusted=raw.copy()
    if normalized:adjusted[:,:4]*=640
    contract['bbox_units']='normalized to 640x640 canvas' if normalized else 'pixels in 640x640 letterboxed canvas'
    contract['output_interpretation']='[batch,9,8400]: cx,cy,w,h,5 class probabilities; no separate objectness; transpose anchors, choose max class, threshold and class-aware NMS, then undo letterbox'
    check={'input_shape':list(feed.shape),'output_shape':list(raw.shape),'raw_box_mean_abs_error_vs_pytorch_pixels':float(np.mean(np.abs(adjusted[:,:4]-pt[:,:4]))),'raw_class_max_abs_error_vs_pytorch':float(np.max(np.abs(raw[:,4:]-pt[:,4:]))),'normalized_coordinates':normalized,'test_photos':[]}
    for p in samples:
        x,meta=preprocess(p);feed=x if tuple(inp[0]['shape'])==(1,3,640,640) else x.transpose(0,2,3,1).copy()
        it.set_tensor(inp[0]['index'],feed);start=time.perf_counter();it.invoke();elapsed=(time.perf_counter()-start)*1000
        result=decode(it.get_tensor(out[0]['index']),meta,normalized=normalized)
        result['inference_cpu_ms']=elapsed;result['source']=p.name
        (DEST/(p.stem+'_litert.json')).write_text(json.dumps(result,indent=2));check['test_photos'].append(result)
    check['python_photo_function_test']=predict_photo(samples[0],local/'best.pt')
    (DEST/'tensor_contract.json').write_text(json.dumps(contract,indent=2))
    (DEST/'conversion_verification.json').write_text(json.dumps(check,indent=2))
    shutil.copy2(ROOT/'01_notebooks/skku_inference.py',DEST/'skku_inference.py')
    (DEST/'labels.txt').write_text('\n'.join(model.names[i] for i in range(5)))
    files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in DEST.iterdir() if p.is_file()}
    (DEST/'artifact_integrity.json').write_text(json.dumps(files,indent=2))
    status['mobile']='exported_and_interpreter_invoked';status['flutter_device_tested']=False
    (RUN/'status.json').write_text(json.dumps(status,indent=2))
    print('MOBILE_SUCCESS',str(DEST),json.dumps(contract),json.dumps(check),flush=True)
except Exception as e:
    status['mobile']='failed';status['mobile_error']=type(e).__name__+': '+str(e);(RUN/'status.json').write_text(json.dumps(status,indent=2));raise
