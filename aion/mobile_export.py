"""Export an ActorCritic checkpoint for the Android diagnostic runtime, not a game policy."""
import argparse
import hashlib
import struct
from pathlib import Path
import torch
from .learning import ActorCritic, Checkpoints

ORDER = ('adapter.base.weight','adapter.base.bias','adapter.a.weight','adapter.b.weight',
         'actor.weight','actor.bias','critic.weight','critic.bias','predictor.weight','predictor.bias')


def export(model, path):
    state = model.state_dict()
    if set(state) != set(ORDER):
        raise ValueError('unexpected ActorCritic architecture')
    values = torch.cat([state[k].detach().cpu().float().reshape(-1) for k in ORDER])
    if values.numel()!=34628 or not torch.isfinite(values).all() or values.abs().max()>100:
        raise ValueError('invalid parameters')
    body = b'AIN1'+struct.pack('<I',1)+values.numpy().astype('<f4').tobytes()
    Path(path).write_bytes(body+hashlib.sha256(body).digest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint-dir',type=Path)
    source.add_argument('--diagnostic',action='store_true')
    a=p.parse_args(); model=ActorCritic()
    if a.checkpoint_dir and not Checkpoints(a.checkpoint_dir).restore(model):
        p.error('no active checkpoint')
    export(model,a.output)
    print('Exported diagnostic parameters; no trained game policy asserted.')

if __name__=='__main__': main()
