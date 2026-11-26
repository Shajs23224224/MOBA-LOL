import argparse
import json
from .vision import read_image, observe


def main():
    p = argparse.ArgumentParser(description='AION local image perception diagnostics')
    p.add_argument('image')
    p.add_argument('--no-ocr', action='store_true')
    p.add_argument('--neural', action='store_true')
    p.add_argument('--rtdetr-dir')
    p.add_argument('--grounding-dir')
    p.add_argument('--labels', default='object. text. icon.')
    p.add_argument('--vl-dir')
    a = p.parse_args()
    if a.rtdetr_dir and a.grounding_dir:
        p.error('choose one detector per run')
    detector = bank = None
    if a.rtdetr_dir:
        from .vision_models import RTDETR
        detector = RTDETR(a.rtdetr_dir)
    if a.grounding_dir:
        from .vision_models import GroundingDINO
        detector = GroundingDINO(a.grounding_dir, a.labels)
    if a.neural:
        from .neural import ExpertBank
        bank = ExpertBank()
    image = read_image(a.image)
    report = observe(image, ocr=not a.no_ocr, detector=detector, bank=bank)
    if a.vl_dir:
        from .vision_models import VisualLanguage
        report['unverified_description'] = VisualLanguage(a.vl_dir).describe(image)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
