"""Rebuild the training data from the three original archives plus the additions shipped here."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import shutil
import yaml

ARCHIVES = [
    ('equipment24_expanded_min100_20261031.zip', 'equipment24_expanded_20261031', '587491a3e2d6ea98310f9c7de489a12b3a139c94de74b765499cd77e92a24f41'),
    ('equipment26_additions_labeled_20261101.zip', 'equipment26_20261101', '7357a197939ab83a600b1a8bffac2031ae131b929335af65cef6631e1fc983b5'),
    ('equipment26_min100_v2_update.zip', 'equipment26_min100_v2', '0356d013ba2d8cbca0b511150665350f6c7e76f2ca2544f0f9490add75a42ee2')
]

def restore(source, output):
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / 'additions' / 'summary.json').read_text())
    config = None
    counts = Counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    for filename, _, expected in ARCHIVES:
        with (source / filename).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                raise ValueError(f'Original archive checksum mismatch: {filename}')
    for filename, prefix, _ in ARCHIVES:
        with ZipFile(source / filename) as z:
            config = yaml.safe_load(z.read(f'{prefix}/data.yaml'))
            for name in z.namelist():
                parts = name.split('/')
                if len(parts) != 4 or parts[0] != prefix or parts[1] not in {'images','labels'} or parts[2] not in summary:
                    continue
                if Path(parts[3]).name != parts[3] or not parts[3] or parts[3] in {'.','..'}:
                    continue
                dest = output.joinpath(*parts[1:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, dest.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    for kind in ('images', 'labels'):
        for path in (root / 'additions' / kind / 'train').glob('*'):
            shutil.copyfile(path, output / kind / 'train' / path.name)
    for split, spec in summary.items():
        labels = list((output / 'labels' / split).glob('*.txt'))
        images = {p.stem for p in (output / 'images' / split).glob('*')}
        if images != {p.stem for p in labels} or len(images) != spec['images']:
            raise ValueError(f'Split mismatch: {split}')
        per_class = Counter()
        for label in labels:
            per_class.update({line.split()[0] for line in label.read_text().splitlines() if line})
        if dict(per_class) != spec['per_class']:
            raise ValueError(f'Class counts mismatch: {split}')
        counts[split] = len(images)
    config['path'] = str(output)
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(dict(counts))

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('original_zip_folder',type=Path);p.add_argument('output',type=Path)
    a = p.parse_args();restore(a.original_zip_folder, a.output)
