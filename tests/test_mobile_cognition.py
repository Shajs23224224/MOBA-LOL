import shutil
import subprocess
from pathlib import Path
import pytest


def test_native_cognition(tmp_path):
    if not shutil.which('javac') or not shutil.which('java'): pytest.skip('JDK unavailable')
    root=Path(__file__).resolve().parents[1]
    src=root/'android/app/src/main/java/org/aion/mobile'
    subprocess.run(['javac','--release','17','-d',str(tmp_path),str(src/'CognitiveStore.java'),str(src/'MobileCouncil.java'),str(root/'android/tests/CognitionProbe.java')],check=True)
    out=subprocess.check_output(['java','-cp',str(tmp_path),'CognitionProbe',str(tmp_path/'memory.bin')],text=True)
    assert 'memory=8x32; restore=ok; roles=160; contradiction=blocked for review; logic=ok; belief=ok' in out
