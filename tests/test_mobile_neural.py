import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from aion.fusion import Fusion
from aion.catalog import SUBSYSTEMS
from aion.neural import ExpertBank
from aion.mobile_neural import export_fusion, export_expert

@pytest.fixture
def jvm(tmp_path):
    if not shutil.which('javac') or not shutil.which('java'): pytest.skip('JDK unavailable')
    root=Path(__file__).resolve().parents[1]
    sources=[root/'android/app/src/main/java/org/aion/mobile/NeuralPort.java',root/'android/tests/NeuralPortProbe.java']
    subprocess.run(['javac','--release','17','-d',str(tmp_path),*map(str,sources)],check=True)
    return ['java','-Xmx64m','-cp',str(tmp_path),'NeuralPortProbe']

def test_fusion_parity(tmp_path,jvm):
    model=Fusion(); path=tmp_path/'fusion.bin'; export_fusion(model,path)
    v=torch.tensor([(i%17-8)/8 for i in range(256)]).reshape(1,256)
    with torch.no_grad():
        out=model(v,torch.arange(16).reshape(16,1).expand(16,32)/16,{SUBSYSTEMS[0]:v},torch.tensor([[.2,.3,.5,.1]]))
        expected=torch.cat([t.flatten() for g in SUBSYSTEMS for t in (out['representations'][g],out['estimates'][g])]+[out['strategic_logits'].flatten(),out['policy_logits'].flatten()]).numpy()
    actual=np.array([float(x) for x in subprocess.check_output(jvm+[str(path)],text=True).splitlines()])
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-5)
    with path.open('r+b') as f: f.seek(40); b=f.read(1); f.seek(40); f.write(bytes([b[0]^1]))
    assert subprocess.run(jvm+[str(path)],capture_output=True).returncode!=0

def test_full_expert_stream(tmp_path,jvm):
    torch.set_num_threads(2)
    model=ExpertBank()._materialize('N001'); path=tmp_path/'expert.bin'; export_expert(model,1,path)
    v=torch.tensor([(i%17-8)/8 for i in range(256)]).reshape(1,256)
    with torch.no_grad(): expected=model(v).flatten().numpy()
    actual=np.array([float(x) for x in subprocess.check_output(jvm+[str(path),'1'],text=True).splitlines()])
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=5e-5)
    assert subprocess.run(jvm+[str(path),'2'],capture_output=True).returncode!=0
