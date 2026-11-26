"""Synthetic research diagnostic: three abstract alternatives; no game simulator."""
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from .learning import ActorCritic, Replay, Experience, Checkpoints, train


def fixture(replay, prefix="fixture", seed=7):
    rng=np.random.default_rng(seed)
    for i in range(180):
        x=np.zeros(256); x[i%3]=1; x[3:]=rng.normal(0,.02,253)
        for a in range(3):
            replay.append(Experience(episode=f'{prefix}-{i}',step=a,state=tuple(x),next_state=tuple(x),action=a,reward=1. if a==i%3 else -1.,done=True,behavior_probability=1/3,authorized=True))


def run(root):
    torch.set_num_threads(2)
    root=Path(root); replay=Replay(root/'replay.sqlite'); fixture(replay)
    model=ActorCritic(); checkpoints=Checkpoints(root/'weights')
    report=train(model,replay,checkpoints,epochs=25)
    restored=ActorCritic(seed=9)
    report['restored']=checkpoints.restore(restored)
    report['equal_after_restore']=report['restored'] and all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values()))
    return report

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as root: print(json.dumps(run(root),indent=2))
