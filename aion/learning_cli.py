"""Ingest authorized logged experiences and run one bounded candidate update."""
import argparse
import json
from pathlib import Path
from .learning import Experience, Replay, ActorCritic, Checkpoints, train


def records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield Experience.model_validate_json(line)


def main():
    p=argparse.ArgumentParser(description='AION offline research learner')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--anchor',type=Path)
    p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args(); replay=Replay(a.root/'replay.sqlite'); checkpoints=Checkpoints(a.root/'weights')
    if a.input:
        for row in records(a.input): replay.append(row)
    model=ActorCritic(); resumed=checkpoints.restore(model)
    result=train(model,replay,checkpoints,epochs=a.epochs,anchor=tuple(records(a.anchor)) if a.anchor else ())
    print(json.dumps({'resumed':resumed,**result},indent=2,allow_nan=False))

if __name__=='__main__': main()
