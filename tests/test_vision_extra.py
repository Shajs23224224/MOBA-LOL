import hashlib
import json
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pytest
from aion.vision import observe, region
from aion.vision_extra import OmniParser, omni_regions, paddle_regions, verify_file


@pytest.fixture
def image():
    return np.zeros((100,200,3),dtype=np.uint8)


def test_weight_hash(tmp_path):
    p = tmp_path/'fixture.pt'
    p.write_bytes(b'test bytes, not a model')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert verify_file(p,digest) == p
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'):
        verify_file(p,digest)


def test_omni_contract(image):
    payload = {'parsed_content_list':[{'type':'icon','bbox':[.1,.2,.5,.6],'content':'example'}]}
    r = omni_regions(image,payload)[0]
    assert r.box == (20,20,100,60) and r.confidence is None
    assert observe(image,ocr=False,detector=lambda _: [r])['regions'][-1]['source'] == 'omniparser_v2'
    payload['parsed_content_list'][0]['bbox'][0] = -1
    with pytest.raises(ValueError):
        omni_regions(image,payload)


def test_paddle_contract(image):
    data = {'rec_texts':['hello'],'rec_scores':[.9],'rec_polys':[[[10,10],[60,10],[60,30],[10,30]]]}
    r = paddle_regions(image,data)[0]
    assert r.box == (10,10,60,30) and r.label == 'hello' and r.confidence == .9
    data['rec_scores'] = []
    with pytest.raises(ValueError):
        paddle_regions(image,data)


def test_omni_http_contract(image):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            data = json.dumps({'parsed_content_list':[{'type':'text','bbox':[0,0,.5,.5],'content':'fixture'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = HTTPServer(('127.0.0.1',0),Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        r = OmniParser(server.server_port)(image)[0]
        assert r.label == 'fixture' and received[0][0] == '/parse/'
        assert base64.b64decode(received[0][1]['base64_image']).startswith(b'\x89PNG')
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_sam_contract(image):
    from aion.vision_extra import SAM2
    class FixturePredictor:
        def set_image(self, frame):
            self.shape = frame.shape[:2]
        def predict(self, **kwargs):
            assert kwargs['box'].tolist() == [10,10,30,30]
            return np.ones((1,*self.shape),bool),np.array([.8]),None
    model = SAM2.__new__(SAM2)
    model.predictor = FixturePredictor()
    out = model.segment(image,[(10,10,30,30)])[0]
    assert out['mask'].shape == (100,200) and out['quality'] == .8


def test_tracker_contract(image):
    from aion.vision_extra import ByteTrack
    from types import SimpleNamespace
    model = ByteTrack.__new__(ByteTrack)
    model.sv = SimpleNamespace(Detections=lambda **kw: SimpleNamespace(**kw))
    def update(d):
        assert len(d.xyxy) == 1 and d.confidence[0] == pytest.approx(.7)
        return SimpleNamespace(xyxy=d.xyxy,confidence=d.confidence,tracker_id=[9])
    model.tracker = SimpleNamespace(update_with_detections=update)
    regions = [region('object','object',.7,(10,10,30,30),'fixture',image),region('icon','unknown',None,(40,40,60,60),'fixture',image)]
    assert model.update(regions)[0]['track_id'] == 9
