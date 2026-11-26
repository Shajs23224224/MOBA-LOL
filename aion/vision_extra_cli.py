"""Read-only diagnostic CLI for optional vision backends."""
import argparse
import json
from .vision import read_image, observe
from .vision_extra import OmniParser, PaddleOCR, SAM2, ByteTrack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('images', nargs='+')
    p.add_argument('--omni-port', type=int)
    p.add_argument('--paddle', nargs=2, metavar=('DET_DIR','REC_DIR'))
    p.add_argument('--sam', nargs=2, metavar=('CHECKPOINT','SHA256'))
    p.add_argument('--box', nargs=4, type=float, action='append')
    p.add_argument('--rtdetr-dir')
    p.add_argument('--track', action='store_true')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--neural', action='store_true')
    a = p.parse_args()
    if bool(a.sam) != bool(a.box):
        p.error('--sam requires --box and vice versa')
    if a.track and not a.rtdetr_dir:
        p.error('--track requires scored detections via --rtdetr-dir')
    detectors = []
    if a.omni_port is not None:
        detectors.append(OmniParser(a.omni_port))
    if a.paddle:
        detectors.append(PaddleOCR(*a.paddle))
    if a.rtdetr_dir:
        from .vision_models import RTDETR
        detectors.append(RTDETR(a.rtdetr_dir))
    segmenter = SAM2(*a.sam) if a.sam else None
    tracker = ByteTrack(a.fps) if a.track else None
    bank = None
    if a.neural:
        from .neural import ExpertBank
        bank = ExpertBank()
    for path in a.images:
        image = read_image(path)
        regions = [r for detector in detectors for r in detector(image)]
        report = observe(image, ocr=False, detector=lambda _: regions, bank=bank)
        if segmenter:
            report['masks'] = [{'prompt':m['prompt'],'quality':m['quality'],'pixels':int(m['mask'].sum())} for m in segmenter.segment(image,a.box)]
        if tracker:
            report['tracks'] = tracker.update(regions)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
