"""Optional, read-only visual adapters. External weights and runtimes are explicit."""
import base64
import hashlib
import http.client
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from .vision import region


def verify_file(path, expected):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('a regular file and expected SHA-256 are required')
    with path.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected.lower():
        raise ValueError('SHA-256 mismatch: ' + str(path))
    return path


def omni_regions(image, result):
    """OmniParser V2 normalized xyxy; captions are hypotheses, not labels of truth."""
    h, w = image.shape[:2]
    out = []
    for entry in result['parsed_content_list'][:512]:
        box = entry['bbox']
        if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box):
            raise ValueError('expected normalized OmniParser box')
        kind = 'text' if entry['type'] == 'text' else 'icon'
        r = region(kind, entry.get('content') or 'unclassified', None,
                   [box[0]*w, box[1]*h, box[2]*w, box[3]*h], 'omniparser_v2', image)
        if r is not None:
            out.append(r)
    return out


class OmniParser:
    """Connect only to a user-started local official OmniParser V2 server."""
    def __init__(self, port=8000, timeout=60):
        if not 1 <= port <= 65535 or not 0 < timeout <= 300:
            raise ValueError('invalid local server settings')
        self.port, self.timeout = port, timeout

    def __call__(self, image):
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format='PNG')
        payload = json.dumps({'base64_image': base64.b64encode(buffer.getvalue()).decode('ascii')})
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
        try:
            connection.request('POST', '/parse/', payload, {'Content-Type':'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError('OmniParser HTTP status ' + str(response.status))
            raw = response.read(32*1024**2+1)
            if len(raw) > 32*1024**2:
                raise ValueError('OmniParser response too large')
            return omni_regions(image, json.loads(raw))
        finally:
            connection.close()


def paddle_regions(image, page):
    texts, scores, polys = (page[k] for k in ('rec_texts','rec_scores','rec_polys'))
    if not len(texts) == len(scores) == len(polys):
        raise ValueError('PaddleOCR result lengths differ')
    out = []
    for text, score, poly in zip(texts, scores, polys):
        points = np.asarray(poly, dtype=float)
        if points.shape != (4,2) or not np.isfinite(points).all():
            raise ValueError('invalid OCR polygon')
        x1,y1 = points.min(axis=0)
        x2,y2 = points.max(axis=0)
        r = region('text', text, score, (x1,y1,x2,y2), 'paddleocr', image)
        if r is not None:
            out.append(r)
    return out[:512]


class PaddleOCR:
    """PaddleOCR 3.x with explicit local detection/recognition model directories."""
    def __init__(self, det_dir, rec_dir):
        from paddleocr import PaddleOCR as Engine
        dirs = [Path(p).resolve(strict=True) for p in (det_dir,rec_dir)]
        if not all(p.is_dir() and any(p.iterdir()) for p in dirs):
            raise ValueError('local PaddleOCR model directories required')
        self.engine = Engine(text_detection_model_dir=str(dirs[0]), text_recognition_model_dir=str(dirs[1]),
                             use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False, device='cpu')

    def __call__(self, image):
        out = []
        for page in self.engine.predict(np.ascontiguousarray(image[:,:,::-1])):
            out.extend(paddle_regions(image, page))
        return out[:512]


class SAM2:
    """Prompted masks only; a mask is not an object classification."""
    def __init__(self, checkpoint, sha256, device='cpu'):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        path = verify_file(checkpoint, sha256)
        self.predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(path), device=device))

    def segment(self, image, boxes):
        if not 0 < len(boxes) <= 16:
            raise ValueError('one to sixteen box prompts required')
        self.predictor.set_image(image)
        results = []
        for box in boxes:
            r = region('region','prompt',None,box,'sam2_prompt',image)
            if r is None:
                raise ValueError('empty prompt')
            masks, scores, _ = self.predictor.predict(box=np.asarray(r.box,dtype=np.float32), multimask_output=False)
            mask = np.asarray(masks[0],dtype=bool)
            if mask.shape != image.shape[:2] or not np.isfinite(scores[0]):
                raise ValueError('invalid SAM2 result')
            results.append({'mask':mask,'quality':float(scores[0]),'prompt':r.box})
        return results


class ByteTrack:
    """Supervision 0.26.1 ByteTrack; one instance per ordered frame sequence."""
    def __init__(self, fps=30):
        import supervision as sv
        self.sv = sv
        self.tracker = sv.ByteTrack(frame_rate=fps)

    def update(self, regions):
        # Do not fabricate confidences for contours or unscored captions.
        candidates = [r for r in regions if r.kind in ('object','icon') and r.confidence is not None]
        boxes = np.asarray([r.box for r in candidates],dtype=np.float32).reshape(-1,4)
        scores = np.asarray([r.confidence for r in candidates],dtype=np.float32)
        detections = self.sv.Detections(xyxy=boxes, confidence=scores)
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return []
        return [{'track_id':int(i),'box':[float(x) for x in b],'confidence':float(s)}
                for i,b,s in zip(tracked.tracker_id,tracked.xyxy,tracked.confidence)]
