"""Local, checksum-verified weight registry. No automatic downloads or pickle loading."""
from pathlib import Path
import hashlib
import json
import os
from tempfile import NamedTemporaryFile
from pydantic import BaseModel, ConfigDict, Field

class Artifact(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    id: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    filename: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    version: str
    source: str
    license: str
    purpose: str

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as f:
            temp=Path(f.name); json.dump(value,f,indent=2,ensure_ascii=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp and temp.exists(): temp.unlink()

class ModelRegistry:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True); self.index=self.root/'registry.json'
    def entries(self):
        if not self.index.exists(): return {}
        raw=json.loads(self.index.read_text(encoding='utf-8'))
        entries={k:Artifact.model_validate(v) for k,v in raw.items()}
        if any(k!=v.id for k,v in entries.items()): raise ValueError('invalid registry key')
        return entries
    def path_for(self,filename):
        p=(self.root/filename).resolve()
        if p.parent!=self.root or p.suffix!='.safetensors': raise ValueError('only local safetensors files are accepted')
        return p
    def verify(self,entry):
        path=self.path_for(entry.filename)
        if digest(path)!=entry.sha256: raise ValueError(f'checksum mismatch: {entry.id}')
        return path
    def register(self,entry):
        self.verify(entry)
        entries=self.entries(); entries[entry.id]=entry
        atomic_json(self.index,{k:v.model_dump() for k,v in entries.items()})
    def load(self,id):
        from safetensors.torch import load_file
        return load_file(str(self.verify(self.entries()[id])),device='cpu')
    def audit(self):
        report=[]
        for id,entry in self.entries().items():
            try: self.verify(entry); status='checksum_verified'
            except (ValueError,OSError) as exc: status=f'invalid: {exc}'
            report.append({'id':id,'status':status,'purpose':entry.purpose})
        return report
