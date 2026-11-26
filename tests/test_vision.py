import shutil
import cv2
import numpy as np
import pytest
import torch
from aion.vision import observe, read_image, region, encode


def sample():
    image = np.full((240,640,3),255,np.uint8)
    cv2.putText(image,'GOLD 1250',(20,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),4)
    cv2.rectangle(image,(30,150),(160,210),(0,0,0),3)
    return image


def test_pixels(tmp_path):
    image = sample()
    path = tmp_path/'test.png'
    cv2.imwrite(str(path),image)
    actual = read_image(path)
    assert np.array_equal(image,actual)
    report = observe(actual,ocr=False)
    assert report['regions'] and len(report['features']) == 256
    assert all(r['label']=='unclassified' for r in report['regions'])
    assert not report['game_control']
    assert not np.array_equal(report['features'],observe(255-actual,ocr=False)['features'])


def test_regions():
    image = sample()
    r = region('text','GOLD',.9,(-5,-3,20,40),'fixture',image)
    assert r.box == (0,0,20,40)
    assert region('text','x',.9,(700,20,750,40),'fixture',image) is None
    assert not torch.equal(encode(image,[]),encode(image,[r]))
    with pytest.raises(ValueError): region('text','x',2,(0,0,10,10),'fixture',image)
    with pytest.raises(ValueError): observe(np.zeros((2,2)),ocr=False)


@pytest.mark.skipif(not shutil.which('tesseract'),reason='requires external pretrained OCR')
def test_real_ocr():
    report = observe(sample())
    texts = [r['label'] for r in report['regions'] if r['kind']=='text']
    assert 'GOLD' in texts and '1250' in texts


def test_real_neural_bridge():
    from aion.neural import ExpertBank
    report = observe(sample(),ocr=False,bank=ExpertBank())
    assert report['neural']['finite'] and len(report['neural']['output']) == 256
    assert report['neural']['route']['selected'][0] in ('N001','N002')
    assert report['neural']['trained'] is False
