from pathlib import Path
import argparse,torch
from ultralytics import YOLO
def main():
    p=argparse.ArgumentParser();p.add_argument('weights');p.add_argument('directory');args=p.parse_args()
    dst=Path(args.directory);dst.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    m=YOLO(args.weights).model.float().fuse().eval()
    head=m.model[-1];head.export=True;head.format='onnx';head.dynamic=False
    x=torch.zeros(1,3,640,640)
    with torch.inference_mode():
        m(x)
        torch.onnx.export(m,x,str(dst/'detector.onnx'),opset_version=17,input_names=['images'],output_names=['detections'],dynamo=False,do_constant_folding=True)
    import onnx,onnxslim
    onnx.save(onnxslim.slim(str(dst/'detector.onnx')),str(dst/'slim.onnx'))
if __name__=='__main__':main()
