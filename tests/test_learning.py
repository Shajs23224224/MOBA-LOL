import sqlite3
import pytest
import torch
from aion.learning import ActorCritic, Replay, Experience, Checkpoints, train, split, expert_features
from aion.learning_demo import fixture


def test_replay(tmp_path):
    p=tmp_path/'r.db'; r=Replay(p,capacity=2)
    data=dict(episode='x',step=0,state=(0.,)*256,next_state=(0.,)*256,action=0,reward=1,done=True,behavior_probability=1/3,authorized=True)
    with pytest.raises(ValueError): r.append(Experience(**{**data,'authorized':False}))
    e=Experience(**data); r.append(e)
    with pytest.raises(sqlite3.IntegrityError): r.append(e)
    assert Replay(p).read(split('x'))==[e]
    for i in (1,2): r.append(Experience(**{**data,'step':i}))
    assert [e.step for e in r.read(split('x'))]==[1,2]


def test_training_checkpoint_and_frozen(tmp_path):
    torch.set_num_threads(2)
    r=Replay(tmp_path/'r.db'); fixture(r); model=ActorCritic(); before={k:v.clone() for k,v in model.state_dict().items()}; c=Checkpoints(tmp_path/'weights')
    assert not {e.episode for e in r.read('train')}&{e.episode for e in r.read('validation')}
    report=train(model,r,c,epochs=25)
    assert report['accepted'] and report['after']['logged_reward_ips']>report['before']['logged_reward_ips']
    assert torch.equal(before['adapter.base.weight'],model.adapter.base.weight)
    assert not torch.equal(before['adapter.b.weight'],model.adapter.b.weight)
    restored=ActorCritic(seed=10); assert c.restore(restored)
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values()))
    path=next(c.root.glob('*.safetensors')); path.write_bytes(b'altered')
    with pytest.raises(ValueError): c.restore(restored)


def test_rejection_and_anchor_validation(tmp_path):
    r=Replay(tmp_path/'r.db'); fixture(r); model=ActorCritic(); c=Checkpoints(tmp_path/'weights')
    with pytest.raises(ValueError): train(model,r,c,anchor=r.read('validation'))
    before={k:v.clone() for k,v in model.state_dict().items()}
    report=train(model,r,c,epochs=1,lr=1e-12)
    assert not report['accepted'] and not c.pointer.exists()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_expert_features():
    from aion.neural import ExpertBank
    v=expert_features(ExpertBank(),[1.]*256,'N001')
    assert len(v)==256 and torch.isfinite(torch.tensor(v)).all()
    assert ActorCritic()(torch.tensor(v).reshape(1,256))[0].shape==(1,3)


def test_continual_update_with_anchor(tmp_path):
    r=Replay(tmp_path/'r.db'); fixture(r); c=Checkpoints(tmp_path/'weights'); model=ActorCritic()
    assert train(model,r,c,epochs=15)['accepted']
    historical=Replay(tmp_path/'historical.db'); fixture(historical,prefix='old',seed=17)
    anchor=tuple(historical.read('validation'))
    active=ActorCritic(seed=1); assert c.restore(active)
    report=train(active,r,c,epochs=10,anchor=anchor)
    assert report['anchor_before'] is not None and report['anchor_after'] is not None
    restored=ActorCritic(); assert c.restore(restored)
    assert all(torch.equal(a,b) for a,b in zip(active.state_dict().values(),restored.state_dict().values()))
    if report['accepted']: assert report['anchor_after']['logged_reward_ips']>=report['anchor_before']['logged_reward_ips']-.05
