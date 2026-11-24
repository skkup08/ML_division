import sys
from pathlib import Path
import numpy as np
import onnx2tf
import tensorflow as tf
directory=Path(sys.argv[1]).resolve()
# Let the converter check its layer plumbing with 20 training images.
import os
from PIL import Image
training_paths=sorted(Path(sys.argv[2]).resolve().glob('*.jpg'))[:20]
if len(training_paths)!=20:raise ValueError('Expected 20 or more training JPEGs')
os.chdir(directory)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy',np.stack([np.asarray(Image.open(p).convert('RGB').resize((128,128)),dtype=np.float32)/255. for p in training_paths]))
core=onnx2tf.convert(input_onnx_file_path=str(directory/'slim.onnx'),output_folder_path=str(directory/'tfcore'),not_use_onnxsim=True,verbosity='error',output_signaturedefs=True,enable_batchmatmul_unfold=True,disable_model_save=True)
print('CORE I/O',core.input_shape,core.output_shape,flush=True)
class Equipment(tf.Module):
    def __init__(self,inner):
        super().__init__();self.core=inner
    @tf.function(input_signature=[tf.TensorSpec([1,None,None,3],tf.uint8,name='image')])
    def rank(self,image):
        # Preserve aspect ratio and match the 114-grey padding used by the detector.
        shape=tf.shape(image)[1:3]; dims=tf.cast(shape,tf.float32)
        size=tf.maximum(1,tf.cast(tf.round(dims*640./tf.reduce_max(dims)),tf.int32))
        first=(640-size)//2;last=640-size-first
        scaled=tf.round(tf.image.resize(tf.cast(image,tf.float32),size,method='bilinear'))
        padded=tf.pad(scaled,[[0,0],[first[0],last[0]],[first[1],last[1]],[0,0]],constant_values=114.)/255.
        padded=tf.reshape(padded,[1,640,640,3])
        raw=self.core(padded,training=False)
        if isinstance(raw,(tuple,list)):raw=raw[0]
        if raw.shape[1]!=30:raw=tf.transpose(raw,[0,2,1])
        scores=raw[0,4:,:];idx=tf.argmax(scores,axis=1,output_type=tf.int32)
        xywh=tf.gather(tf.transpose(raw[0,:4,:]),idx)
        center,extent=tf.split(xywh,2,axis=1)
        # Normalize to the original photograph: [left, top, right, bottom].
        shift=tf.cast(tf.reverse(first,[0]),tf.float32); divisor=tf.cast(tf.reverse(size,[0]),tf.float32)
        boxes=tf.concat([(center-extent/2.-shift)/divisor,(center+extent/2.-shift)/divisor],1)
        return {'class_scores':tf.reduce_max(scores,axis=1),'best_boxes':tf.clip_by_value(boxes,0,1)}
wrapper=Equipment(core)
concrete=wrapper.rank.get_concrete_function()
conv=tf.lite.TFLiteConverter.from_concrete_functions([concrete],wrapper)
conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
out=directory/'equipment26_anysize.tflite';out.write_bytes(conv.convert())
from add_signature import add_signature
add_signature(out)
print(out,out.stat().st_size,flush=True)
