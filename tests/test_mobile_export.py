import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from aion.learning import ActorCritic
from aion.mobile_export import export


def test_java_pytorch_parity(tmp_path):
    if not shutil.which('javac') or not shutil.which('java'):
        pytest.skip('JDK missing')
    root=Path(__file__).resolve().parents[1]
    sources=[root/'android/app/src/main/java/org/aion/mobile/Policy.java',root/'android/tests/PolicyProbe.java']
    subprocess.run(['javac','--release','17','-d',str(tmp_path),*map(str,sources)],check=True)
    model=ActorCritic(); model.adapter.b.weight.data.fill_(.01)
    path=tmp_path/'policy.aion'; export(model,path)
    run=['java','-cp',str(tmp_path),'PolicyProbe',str(path)]
    actual=np.array([float(v) for v in subprocess.check_output(run,text=True).splitlines()])
    x=torch.tensor([(i%17-8)/8 for i in range(256)]).reshape(1,256)
    with torch.no_grad():
        logits,value,prediction=model(x)
    expected=torch.cat((logits.flatten(),value,prediction.flatten())).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
    data=bytearray(path.read_bytes()); data[20]^=1; path.write_bytes(data)
    assert subprocess.run(run,capture_output=True).returncode!=0
