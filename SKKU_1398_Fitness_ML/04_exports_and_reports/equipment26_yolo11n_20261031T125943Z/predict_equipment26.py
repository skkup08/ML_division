"""26-class detector: EXIF/RGB conversion, aspect-preserving 640px inference."""
from pathlib import Path
import argparse
import json
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO


class EquipmentDetector:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else Path(__file__).with_name('best.pt')
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = YOLO(str(path))
        if len(self.model.names) != 26:
            raise ValueError('Expected the 26-class equipment model.')
        mapping_path = path.with_name('class_mapping.json')
        self.mapping = {m['yolo_id']: m for m in json.loads(mapping_path.read_text())}
        if any(self.mapping[i]['name'] != name for i, name in self.model.names.items()):
            raise ValueError('Class mapping does not match the checkpoint.')

    def predict(self, image_path, conf=0.25, annotated_path=None, device=None):
        if not 0 <= conf <= 1:
            raise ValueError('conf must be between 0 and 1.')
        try:
            with Image.open(image_path) as source:
                source_size = source.size
                oriented = ImageOps.exif_transpose(source)
                if 'A' in oriented.getbands() or 'transparency' in oriented.info:
                    rgba = oriented.convert('RGBA')
                    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, rgba).convert('RGB')
                else:
                    image = oriented.convert('RGB')
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError('Unsupported or damaged image. Convert it to JPEG, PNG or WebP.') from exc
        # Ultralytics letterboxes without stretching, normalizes and maps boxes back.
        result = self.model.predict(image, imgsz=640, rect=False, conf=conf,
                                    device=device, verbose=False, save=False)[0]
        width, height = image.size
        assert tuple(result.orig_shape) == (height, width)
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                'class_id': class_id,
                'center_id': self.mapping[class_id]['center_id'],
                'class_name': self.model.names[class_id],
                'confidence': float(box.conf.item()),
                'bbox_xyxy': xyxy,
                'bbox_xyxy_normalized': [xyxy[0]/width, xyxy[1]/height,
                                        xyxy[2]/width, xyxy[3]/height],
            })
        if annotated_path:
            result.save(filename=str(annotated_path))
        return {'source_size': list(source_size), 'image_size': [width, height],
                'coordinate_system': 'EXIF-oriented image; origin top-left; xyxy pixels',
                'detections': detections}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--model')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--output', help='Optional JSON output path')
    parser.add_argument('--annotated', help='Optional annotated image path')
    args = parser.parse_args()
    payload = EquipmentDetector(args.model).predict(args.image, args.conf, args.annotated)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)
