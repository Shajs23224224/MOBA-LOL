import shutil
import subprocess
from pathlib import Path
import pytest


def test_native_mobile_learning(tmp_path):
    if not shutil.which('javac') or not shutil.which('java'):
        pytest.skip('JDK missing')
    root=Path(__file__).resolve().parents[1]
    sources=[root/'android/app/src/main/java/org/aion/mobile/MobileLab.java',root/'android/tests/MobileLabProbe.java']
    subprocess.run(['javac','--release','17','-d',str(tmp_path),*map(str,sources)],check=True)
    out=subprocess.check_output(['java','-cp',str(tmp_path),'MobileLabProbe',str(tmp_path/'state.bin')],text=True)
    assert 'Aceptado' in out and 'restore=exact; corruption=rejected; parameters=771' in out
