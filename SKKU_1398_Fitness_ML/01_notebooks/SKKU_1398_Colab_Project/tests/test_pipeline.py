from pathlib import Path
import sys
import csv
import json
import tempfile
import unittest

import yaml
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'src'))
from skku1398_ml import (
    CLASS_NAMES,normalize_name,parse_class_names,source_to_target,
    read_yolo_boxes, prepare_yolo_dataset, make_annotation_preview,
)


class DatasetPipelineTests(unittest.TestCase):
    def test_names_and_mapping(self):
        self.assertEqual(normalize_name('Seated-Leg_Press'), 'seatedlegpress')
        self.assertEqual(parse_class_names({0:'Treadmill',1:'Smith Machine'}),['Treadmill','Smith Machine'])
        with self.assertRaises(ValueError):parse_class_names({'0':'a','2':'b'})
        mapping=source_to_target(['Bench','Treadmill','Chest Press Machine','Smith Machine'])
        self.assertEqual(mapping,{1:0,2:1,3:4})

    def test_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'test.txt'
            f.write_text('0 .5 .5 .4 .2\n0 .9 .9 .4 .4\n0 .5 .5 -1 .5\n1 .5 .5 .2 .2\n',encoding='utf-8')
            boxes,errors=read_yolo_boxes(f,{0:0})
            self.assertEqual(len(boxes),1)
            self.assertEqual(len(errors),2)
            self.assertEqual(read_yolo_boxes(Path(tmp)/'missing.txt',{0:0})[0],[])

    def test_full_data_prep_and_leak_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); src=base/'source'; dest=base/'processed'
            (src/'data.yaml').parent.mkdir(parents=True,exist_ok=True)
            names=['Treadmill','Chest Press Machine','Shoulder Press Machine','Leg Press Machine','Smith Machine','Bench']
            (src/'data.yaml').write_text(yaml.safe_dump({'names':names,'train':'../train/images','val':'../valid/images','test':'../test/images'}))
            for i,split in enumerate(('train','valid','test')):
                (src/split/'images').mkdir(parents=True)
                (src/split/'labels').mkdir()
                for k in range(7):
                    img=Image.new('RGB',(120,90),((k*29+i*5)%255,(i*71+k*9)%255,130))
                    filename=f'{split}_{k}.png'
                    img.save(src/split/'images'/filename)
                    cl=k%6
                    (src/split/'labels'/f'{split}_{k}.txt').write_text(f'{cl} 0.5 0.5 0.4 0.4\n')
            # Ensure exact image reuse is removed even when filenames differ.
            image_a=src/'test'/'images'/'test_1.png'
            target=src/'train'/'images'/'train_1.png'
            target.write_bytes(image_a.read_bytes())
            (src/'train'/'labels'/'train_1.txt').write_text('1 0.5 0.5 0.3 0.3\n')
            report=prepare_yolo_dataset(src,dest,mode='baseline')
            self.assertEqual(report['classes'],CLASS_NAMES)
            self.assertTrue((dest/'data.yaml').exists())
            self.assertGreaterEqual(report['label_issue_count'],1)
            self.assertEqual(len(list((dest/'test'/'images').glob('*'))),7)
            self.assertGreater(len(list((dest/'train'/'images').glob('*'))),0)
            data=yaml.safe_load((dest/'data.yaml').read_text())
            self.assertEqual(len(data['names']),5)
            hashes=[]
            for split in ('train','val','test'):
                for f in (dest/split/'images').iterdir():
                    import hashlib
                    hashes.append(hashlib.sha256(f.read_bytes()).hexdigest())
            self.assertEqual(len(hashes),len(set(hashes)))
            preview=make_annotation_preview(dest,base/'preview.jpg')
            self.assertTrue(preview.exists())
            with self.assertRaises(FileExistsError):
                prepare_yolo_dataset(src,dest,mode='baseline')

if __name__=='__main__':unittest.main()
