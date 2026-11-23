"""Five-class inference helpers. LiteRT contract is verified against the exported model."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
NAMES=['treadmill','seated_chest_press','shoulder_press','seated_leg_press','smith_machine']
def preprocess(path,size=640):
    im=ImageOps.exif_transpose(Image.open(path)).convert('RGB');w,h=im.size
    r=min(size/w,size/h);nw,nh=round(w*r),round(h*r)
    left,top=round((size-nw)/2-.1),round((size-nh)/2-.1)
    import cv2
    arr=cv2.resize(np.asarray(im),(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((size,size,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=arr
    return np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)/255,{'width':w,'height':h,'ratio':r,'left':left,'top':top}
def decode(raw,meta,confidence=.25,iou=.7,normalized=False):
    pred=np.asarray(raw)[0]
    if pred.shape[0]==9:pred=pred.T
    if pred.shape[1]!=9:raise ValueError('Expected 4 box channels + 5 class channels')
    cls=pred[:,4:].argmax(1);scores=pred[np.arange(len(pred)),4+cls]
    mask=scores>=confidence;pred=pred[mask];cls=cls[mask];scores=scores[mask]
    coords=pred[:,:4].copy()
    if normalized:coords*=640
    xy=np.column_stack((coords[:,0]-coords[:,2]/2,coords[:,1]-coords[:,3]/2,coords[:,0]+coords[:,2]/2,coords[:,1]+coords[:,3]/2))
    order=np.argsort(-scores);keep=[]
    while len(order) and len(keep)<300:
        i=order[0];keep.append(i);rest=order[1:]
        lt=np.maximum(xy[i,:2],xy[rest,:2]);rb=np.minimum(xy[i,2:],xy[rest,2:]);wh=np.maximum(rb-lt,0)
        inter=wh[:,0]*wh[:,1];area=lambda b:np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
        overlap=inter/(area(xy[i:i+1])[0]+area(xy[rest])-inter+1e-7)
        order=rest[(cls[rest]!=cls[i])|(overlap<=iou)]
    out=[]
    for i in keep:
        b=xy[i];b[[0,2]]=(b[[0,2]]-meta['left'])/meta['ratio'];b[[1,3]]=(b[[1,3]]-meta['top'])/meta['ratio']
        b[[0,2]]=np.clip(b[[0,2]],0,meta['width']);b[[1,3]]=np.clip(b[[1,3]],0,meta['height'])
        out.append({'class_id':int(cls[i]),'class_name':NAMES[cls[i]],'confidence':float(scores[i]),'bbox_xyxy_pixels':b.tolist()})
    return {'status':'ok' if out else 'no_detections','image_width':meta['width'],'image_height':meta['height'],'detections':out}
def predict_photo(image_path,model_path,confidence=.25):
    from ultralytics import YOLO
    r=YOLO(str(model_path)).predict(str(image_path),imgsz=640,conf=confidence,verbose=False)[0]
    return {'image_width':r.orig_shape[1],'image_height':r.orig_shape[0],'detections':[{'class_id':int(b.cls.item()),'class_name':r.names[int(b.cls.item())],'confidence':float(b.conf.item()),'bbox_xyxy_pixels':b.xyxy[0].tolist()} for b in r.boxes]}
def predict_litert(image_path,model_path,confidence=.25,normalized=True):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=str(model_path));it.allocate_tensors();inp=it.get_input_details()[0];out=it.get_output_details()[0]
    x,meta=preprocess(image_path)
    if tuple(inp['shape'])==(1,640,640,3):x=x.transpose(0,2,3,1).copy()
    it.set_tensor(inp['index'],x);it.invoke()
    return decode(it.get_tensor(out['index']),meta,confidence,normalized=normalized)
