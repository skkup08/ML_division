"""Fine-tune the 26-class detector on the original split plus the reviewed training images."""
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(6)
    model = YOLO(args.weights)
    model.train(data=str(Path(args.data).resolve()), epochs=8, patience=4, imgsz=416,
                batch=12, device=args.device, workers=2, freeze=23, optimizer='AdamW',
                lr0=0.00055, lrf=0.10, warmup_epochs=1.0, weight_decay=0.0005,
                mosaic=0.35, close_mosaic=2, translate=0.08, scale=0.3, fliplr=0.5,
                seed=1398, deterministic=True, plots=False, amp=False, cache=False,
                project=str(Path(args.output).resolve()), name='min200', save=True, verbose=False)

if __name__ == '__main__':
    main()
