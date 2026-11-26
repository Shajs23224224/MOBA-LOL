"""Bounded offline continual learning from authorized research logs; no game actions."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import save_file, load_file
from .models import atomic_json, digest

DIM, ACTIONS = 256, 3

class Experience(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    episode: str = Field(min_length=1,max_length=128)
    step: int = Field(ge=0)
    state: tuple[float,...] = Field(min_length=DIM,max_length=DIM)
    next_state: tuple[float,...] = Field(min_length=DIM,max_length=DIM)
    action: int = Field(ge=0,lt=ACTIONS)
    reward: float = Field(ge=-1,le=1)
    done: bool
    behavior_probability: float = Field(ge=.05,le=1)
    authorized: bool


def split(episode):
    return 'validation' if int(hashlib.sha256(episode.encode()).hexdigest(),16)%5==0 else 'train'

class Replay:
    def __init__(self,path,capacity=4096):
        if not 1<=capacity<=100000: raise ValueError('invalid capacity')
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.capacity=capacity
        with closing(sqlite3.connect(self.path)) as c,c:
            c.execute('CREATE TABLE IF NOT EXISTS replay(seq INTEGER PRIMARY KEY AUTOINCREMENT, episode TEXT, step INTEGER, partition TEXT, body TEXT, UNIQUE(episode,step))')
    def append(self,e):
        if not e.authorized: raise ValueError('authorized research data required')
        for v in (e.state,e.next_state):
            if max(abs(x) for x in v)>100: raise ValueError('feature magnitude exceeds budget')
        with closing(sqlite3.connect(self.path)) as c,c:
            c.execute('INSERT INTO replay(episode,step,partition,body) VALUES(?,?,?,?)',(e.episode,e.step,split(e.episode),e.model_dump_json()))
            c.execute('DELETE FROM replay WHERE partition=? AND seq NOT IN (SELECT seq FROM replay WHERE partition=? ORDER BY seq DESC LIMIT ?)',(split(e.episode),split(e.episode),self.capacity))
    def read(self,partition):
        if partition not in ('train','validation'): raise ValueError('invalid partition')
        with closing(sqlite3.connect(self.path)) as c:
            return [Experience.model_validate_json(r[0]) for r in c.execute('SELECT body FROM replay WHERE partition=? ORDER BY seq',(partition,))]

class LoRA(nn.Module):
    def __init__(self,width=64,rank=4):
        super().__init__(); self.base=nn.Linear(DIM,width); self.base.requires_grad_(False)
        self.a=nn.Linear(DIM,rank,bias=False); self.b=nn.Linear(rank,width,bias=False)
        nn.init.zeros_(self.b.weight); self.scale=1/rank
    def forward(self,x): return self.base(x)+self.scale*self.b(self.a(x))

class ActorCritic(nn.Module):
    def __init__(self,seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.adapter=LoRA(); self.actor=nn.Linear(64,ACTIONS); self.critic=nn.Linear(64,1); self.predictor=nn.Linear(64,DIM)
    def forward(self,x):
        h=torch.tanh(self.adapter(x)); return self.actor(h),self.critic(h).squeeze(-1),self.predictor(h)


def batch(rows):
    if not rows: raise ValueError('empty experience set')
    return (torch.tensor([e.state for e in rows],dtype=torch.float32),torch.tensor([e.next_state for e in rows],dtype=torch.float32),torch.tensor([e.action for e in rows]),torch.tensor([e.reward for e in rows]),torch.tensor([e.done for e in rows],dtype=torch.float32),torch.tensor([e.behavior_probability for e in rows]))

@torch.no_grad()
def metrics(model,rows):
    x,y,a,r,d,p=batch(rows); logits,v,pred=model(x)
    # One-step logged reward estimate, not a long-horizon policy value.
    ips=(logits.softmax(-1).gather(1,a[:,None]).squeeze(1)/p*r).mean()
    target=r+.95*(1-d)*model(y)[1]
    return {'logged_reward_ips':float(ips),'td_mse':float(F.mse_loss(v,target)), 'prediction_mse':float(F.mse_loss(pred,y))}

class Checkpoints:
    """Atomic active pointer and immutable versioned full policy snapshots; one writer."""
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.pointer=self.root/'active.json'
    def save(self,model,report):
        name=uuid4().hex+'.safetensors'; path=self.root/name
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path))
        atomic_json(self.pointer,{'filename':name,'sha256':digest(path),'report':report})
    def restore(self,model):
        if not self.pointer.exists(): return False
        meta=json.loads(self.pointer.read_text()); path=(self.root/meta['filename']).resolve()
        if path.parent!=self.root.resolve() or path.suffix!='.safetensors' or digest(path)!=meta['sha256']: raise ValueError('invalid checkpoint')
        model.load_state_dict(load_file(str(path)),strict=True); return True


def train(model,replay,checkpoints,*,epochs=20,lr=.01,seed=42,anchor=()):
    """Offline advantage-weighted actor/TD critic plus next-feature prediction and replay.
    Anchor is an optional independent historical validation set, not a guarantee against forgetting.
    """
    if not 1<=epochs<=100 or not 0<lr<=.1: raise ValueError('invalid training budget')
    import copy
    train_rows=replay.read('train'); validation=replay.read('validation')
    if len(train_rows)<24 or len(validation)<12: raise ValueError('insufficient separated experiences')
    if len({e.action for e in train_rows})!=ACTIONS or len({e.action for e in validation})!=ACTIONS: raise ValueError('action coverage required')
    if anchor and (any(not e.authorized for e in anchor) or {e.episode for e in anchor}&{e.episode for e in train_rows+validation}): raise ValueError('anchor must be authorized and disjoint')
    candidate=copy.deepcopy(model); opt=torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad],lr=lr)
    before=metrics(model,validation); old=metrics(model,anchor) if anchor else None
    x,y,a,r,d,p=batch(train_rows); generator=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ids in torch.randperm(len(x),generator=generator).split(64):
            logits,v,pred=candidate(x[ids])
            with torch.no_grad():
                target=r[ids]+.95*(1-d[ids])*candidate(y[ids])[1]
                weight=(target-v).clamp(-2,2).exp()
                reference=model(x[ids])[0].softmax(-1)
            actor=-(weight*logits.log_softmax(-1).gather(1,a[ids,None]).squeeze(1)).mean()
            loss=actor+F.mse_loss(v,target)+.2*F.mse_loss(pred,y[ids])+.02*F.kl_div(logits.log_softmax(-1),reference,reduction='batchmean')
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.); opt.step()
    after=metrics(candidate,validation); new=metrics(candidate,anchor) if anchor else None
    accepted=after['logged_reward_ips']>before['logged_reward_ips']+1e-4 and after['td_mse']<=before['td_mse']+.05 and (not old or new['logged_reward_ips']>=old['logged_reward_ips']-.05)
    report={'before':before,'after':after,'anchor_before':old,'anchor_after':new,'accepted':accepted,'train_count':len(train_rows),'validation_count':len(validation),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'game_control':False}
    if accepted:
        checkpoints.save(candidate,report); model.load_state_dict(candidate.state_dict())
    return report


def expert_features(bank,vector,expert_id):
    """Optional frozen Phase 2 diagnostic representation; use same representation for all records."""
    out,_=bank.infer(torch.tensor(vector,dtype=torch.float32).reshape(1,DIM),candidates=[expert_id])
    return tuple(out[0].tolist())
