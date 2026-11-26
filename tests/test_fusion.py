import numpy as np
import pytest
import torch
from aion.fusion import Fusion
from aion.catalog import SUBSYSTEMS
from aion.unified import Unified
from aion.cognition import Cognition
from aion.neural import ExpertBank
from aion.moba import Snapshot
from aion.learning import ActorCritic


def test_fusion_audit_and_gradients():
    f=Fusion(); audit=f.audit()
    assert audit['intermediate_count']==32 and audit['final_count']==2
    assert audit['parameters']==sum(p.numel() for p in f.parameters())
    assert audit['resident_bytes']==4*audit['parameters']
    out=f(torch.ones(1,256),torch.ones(16,32),{},torch.zeros(1,4))
    assert len(out['representations'])==len(out['estimates'])==16
    assert out['strategic_logits'].shape==(1,16) and out['policy_logits'].shape==(1,3)
    before=f.heads['F02'][0].weight.detach().clone()
    optimizer=torch.optim.SGD(f.parameters(),lr=.01)
    out['policy_logits'].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters())
    optimizer.step()
    assert not torch.equal(before,f.heads['F02'][0].weight)


def test_missing_expert_and_validation():
    f=Fusion(); v=torch.ones(1,256); s=torch.zeros(16,32); a=torch.zeros(1,4)
    first=f(v,s,{},a); second=f(v,s,{SUBSYSTEMS[0]:v},a)
    assert not torch.equal(first['representations'][SUBSYSTEMS[0]],second['representations'][SUBSYSTEMS[0]])
    assert torch.equal(first['representations'][SUBSYSTEMS[1]],second['representations'][SUBSYSTEMS[1]])
    with pytest.raises(ValueError): f(v,s,{'bad':v},a)
    with pytest.raises(ValueError): f(v,s,{SUBSYSTEMS[0]:torch.ones(1,3)},a)
    with pytest.raises(ValueError): f(v*float('nan'),s,{},a)


def test_actor_input_changes_finals():
    f=Fusion(); v=torch.ones(1,256); s=torch.zeros(16,32)
    a=ActorCritic(); b=ActorCritic(seed=9)
    def features(m):
        logits,value,_=m(v); return torch.cat((logits.softmax(-1),value.reshape(1,1)),dim=1)
    assert not torch.equal(f(v,s,{},features(a))['policy_logits'],f(v,s,{},features(b))['policy_logits'])


def test_unified_research(tmp_path):
    c=Cognition(tmp_path/'test.sqlite',bank=ExpertBank())
    u=Unified(c)
    image=np.full((32,32,3),127,np.uint8)
    snapshot=Snapshot(health=.1,resource=.2,gold=100,threat=.9,ally_support=0,objective_value=.9,farm_value=.4,shop_access=True,free_slots=1,confidence=.95)
    for k in ('semantic','procedural'): c.memory.put('scope',k,0,{'fixture':True},np.ones(256))
    r=u.run('scope',1,image,snapshot,subsystems=('percepcion_visual',),ocr=False)
    assert r['report']['cognition']['council']['roles_evaluated']==160
    assert len(r['report']['moba']['memory_by_kind'])==8
    assert r['integration']['subsystems']['percepcion_visual']['expert_available']
    assert len(r['integration']['subsystems'])==16
    assert r['integration']['research_candidate']=='recover'
    assert r['integration']['intent']['kind']=='noop' and r['integration']['requires_review']
    assert sum(r['integration']['policy_diagnostic'])==pytest.approx(1)
    assert r['audit']['logical_total_parameters']==r['audit']['bank']['logical_total_parameters']+r['audit']['fusion']['parameters']+34628
    assert any('fusion_review' in x['payload'] for x in c.memory.recall('scope',r['report']['cognition']['visual']['features'],kinds=('metacognitive',))['items'])
    with pytest.raises(ValueError): u.run('scope',2,image,snapshot,subsystems=('bad',),ocr=False)
