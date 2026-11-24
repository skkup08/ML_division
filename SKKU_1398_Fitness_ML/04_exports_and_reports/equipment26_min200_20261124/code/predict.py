"""Rank equipment classes for one photograph with the dynamic-input TFLite model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from ai_edge_litert.interpreter import Interpreter

def predict(photo, top_k=3, min_score=0.0, model=None):
    root = Path(__file__).resolve().parents[1]
    classes = json.loads((root / 'model' / 'class_mapping.json').read_text(encoding='utf-8'))
    if not 1 <= top_k <= len(classes):
        raise ValueError('top_k must be from 1 to 26')
    if not 0 <= min_score <= 1:
        raise ValueError('min_score must be from 0 to 1')
    with Image.open(photo) as src:
        rgb = np.asarray(ImageOps.exif_transpose(src).convert('RGB'), dtype=np.uint8)
    engine = Interpreter(model_path=str(model or root / 'model' / 'equipment26_anysize.tflite'), num_threads=2)
    signature = engine.get_signature_runner()
    result = signature(image=rgb[None, ...])
    scores = result['class_scores']
    ranked = np.argsort(-scores, kind='stable')[:top_k]
    return [
        {'rank': i + 1, 'id': int(k), 'equipment': classes[k]['name_ko'],
         'score': round(float(scores[k]), 5),
         'box_xyxy': [round(float(x), 5) for x in result['best_boxes'][k]]}
        for i, k in enumerate(ranked) if scores[k] >= min_score
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('photo')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--min-score', type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(predict(args.photo, args.top_k, args.min_score), ensure_ascii=False, indent=2))
