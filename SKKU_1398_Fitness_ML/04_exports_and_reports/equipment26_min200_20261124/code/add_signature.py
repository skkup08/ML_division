from pathlib import Path
import sys,flatbuffers
from tensorflow.lite.python import schema_py_generated as sch

def add_signature(path):
    data=path.read_bytes()
    model=sch.ModelT.InitFromObj(sch.Model.GetRootAsModel(data,0))
    sub=model.subgraphs[0]
    def entry(name,index):
        m=sch.TensorMapT();m.name=name;m.tensorIndex=int(index);return m
    sig=sch.SignatureDefT();sig.signatureKey='rank';sig.subgraphIndex=0
    sig.inputs=[entry('image',sub.inputs[0])]
    outputs={tuple(sub.tensors[i].shape):i for i in sub.outputs}
    sig.outputs=[entry('class_scores',outputs[(26,)]),entry('best_boxes',outputs[(26,4)])]
    model.signatureDefs=[sig]
    b=flatbuffers.Builder(len(data));off=model.Pack(b);b.Finish(off,file_identifier=b'TFL3')
    path.write_bytes(b.Output())
if __name__=='__main__':add_signature(Path(sys.argv[1]))
