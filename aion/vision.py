"""Read-only perception of supplied local images; no game control or screen hooks."""
import csv
import hashlib
import io
import math
from pathlib import Path
import subprocess
from dataclasses import dataclass, asdict
import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class Region:
    kind: str
    label: str
    confidence: float | None
    box: tuple[int, int, int, int]
    source: str


def read_image(path):
    from PIL import Image
    with Image.open(Path(path)) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError('image exceeds pixel budget')
        return np.array(image.convert('RGB'))


def region(kind, label, score, box, source, image):
    h, w = image.shape[:2]
    if not all(math.isfinite(float(v)) for v in box):
        raise ValueError('nonfinite box')
    x1, y1, x2, y2 = map(int, box)
    box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    if score is not None and not 0 <= float(score) <= 1:
        raise ValueError('invalid confidence')
    return Region(kind, str(label)[:512], None if score is None else float(score), box, source)


def contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    found, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted((cv2.boundingRect(c) for c in found), key=lambda b: b[2]*b[3], reverse=True)
    return [region('region', 'unclassified', None, (x,y,x+w,y+h), 'opencv', image)
            for x,y,w,h in boxes[:64] if w*h >= 100]


def tesseract(image):
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('PNG encoding failed')
    result = subprocess.run(['tesseract','stdin','stdout','-l','eng','--psm','11','tsv'],
                            input=encoded.tobytes(), capture_output=True, timeout=30, check=True)
    out = []
    for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')), delimiter='\t'):
        if not row['text'].strip() or float(row['conf']) < 0:
            continue
        x,y,w,h = (int(row[k]) for k in ('left','top','width','height'))
        r = region('text', row['text'], float(row['conf'])/100, (x,y,x+w,y+h), 'tesseract', image)
        if r is not None:
            out.append(r)
    return out[:256]


def encode(image, regions):
    """192 RGB grid values + 48 region occupancy + 16 hashed OCR features."""
    pixels = cv2.resize(image, (8,8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255
    features = np.zeros(64, dtype=np.float32)
    h,w = image.shape[:2]
    for r in regions:
        x1,y1,x2,y2 = r.box
        x = min(7, int(8*(x1+x2)/(2*w)))
        y = min(5, int(6*(y1+y2)/(2*h)))
        features[y*8+x] += 1
        if r.kind == 'text':
            bucket = hashlib.sha256(r.label.casefold().encode()).digest()[0] % 16
            features[48+bucket] += 1
    return torch.from_numpy(np.concatenate((pixels, np.tanh(features)))).reshape(1,256)


def observe(image, *, ocr=True, detector=None, bank=None):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not 0 < image.shape[0]*image.shape[1] <= 16_000_000:
        raise ValueError('expected bounded RGB uint8 image')
    regions = contours(image)
    if ocr:
        regions += tesseract(image)
    if detector is not None:
        regions += detector(image)
    regions = [r for r in regions if r is not None]
    vector = encode(image, regions)
    report = {'shape': list(image.shape), 'regions': [asdict(r) for r in regions], 'features': vector.tolist()[0], 'ocr_enabled': ocr, 'game_control': False}
    if bank is not None:
        output, routing = bank.infer(vector, candidates=['N001','N002'])
        report['neural'] = {'route': routing, 'output': output.tolist()[0], 'finite': bool(torch.isfinite(output).all()), 'trained': False}
    return report
