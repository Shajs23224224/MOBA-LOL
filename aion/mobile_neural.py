"""Export Phase 2/9 weights to diagnostic Java streaming format."""
import argparse
import hashlib
import struct
from pathlib import Path
import torch
from safetensors.torch import load_file
from .fusion import Fusion
from .neural import ExpertBank


def write(path, magic, count, ident, tensors):
    h=hashlib.sha256(); total=0
    with Path(path).open('wb') as f:
        def emit(data):
            f.write(data); h.update(data)
        emit(struct.pack('>III',magic,count,ident))
        for p in tensors:
            a=p.detach().cpu().float()
            if not torch.isfinite(a).all() or a.abs().max()>100:
                raise ValueError('invalid weights')
            total+=a.numel(); emit(a.numpy().astype('>f4').tobytes())
        if total!=count: raise ValueError('parameter count mismatch')
        f.write(h.digest())


def export_fusion(model,path):
    write(path,0x41464e31,2163411,0,model.parameters())


def export_expert(model,ident,path):
    if not 1<=ident<=824: raise ValueError('invalid ID')
    tensors=[]
    for layer,norm in zip(model.layers,model.norms): tensors.extend((layer.weight,layer.bias,norm.weight,norm.bias))
    tensors.extend((model.head.weight,model.head.bias))
    write(path,0x41454e31,64061696,ident,tensors)


def main():
    p=argparse.ArgumentParser(); p.add_argument('output'); p.add_argument('--expert',type=int); p.add_argument('--checkpoint'); a=p.parse_args()
    if a.expert is None:
        model=Fusion()
        if a.checkpoint: model.load_state_dict(load_file(a.checkpoint),strict=True)
        export_fusion(model,a.output)
    else:
        if not 1<=a.expert<=824: p.error('expert ID 1..824')
        model=ExpertBank()._materialize(f'N{a.expert:03}')
        if a.checkpoint: model.load_state_dict(load_file(a.checkpoint),strict=True)
        export_expert(model,a.expert,a.output)
    print('Diagnostic export; training and provenance not certified.')

if __name__=='__main__': main()
