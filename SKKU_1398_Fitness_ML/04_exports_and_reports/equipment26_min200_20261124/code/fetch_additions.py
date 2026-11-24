"""Recreate the 516 training JPEGs from the published sources and verify each SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

def fetch(record, folder):
    destination = folder / record['filename']
    expected = record['sha256']
    if destination.exists() and sha256(destination.read_bytes()).hexdigest() == expected:
        return None
    ident = record['source_id']
    url = record['image_source']
    is_roboflow = ident.startswith('rf_')
    if not is_roboflow:
        parsed = urlsplit(url)
        params = dict(parse_qsl(parsed.query))
        params['width'] = '768'
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(params), ''))
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; gym equipment dataset research)'})
    with urlopen(request, timeout=25) as source:
        content = source.read(12_000_000)
    if not is_roboflow:
        with Image.open(BytesIO(content)) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
        im.thumbnail((896, 896))
        extra_situp = ident.startswith('21-') and int(ident.split('-')[1]) >= 501
        buffer = BytesIO()
        im.save(buffer, 'JPEG', quality=90 if extra_situp else 91)
        content = buffer.getvalue()
    if sha256(content).hexdigest() != expected:
        raise ValueError('SHA-256 mismatch')
    with NamedTemporaryFile(dir=folder, delete=False) as temp:
        temp.write(content)
    Path(temp.name).replace(destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'additions'
    records = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    folder = root / 'images' / 'train'
    folder.mkdir(parents=True, exist_ok=True)
    def one(record):
        try:
            fetch(record, folder)
        except Exception as exc:
            return f"{record['filename']} ({exc})"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = [message for message in pool.map(one, records) if message]
    print(f'{len(records) - len(failed)} / {len(records)} files verified')
    if failed:
        print('\n'.join(failed))
        raise SystemExit(1)

if __name__ == '__main__':
    main()
