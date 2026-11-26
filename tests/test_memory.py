import numpy as np
import pytest
from aion.memory import Memory, KINDS
from aion.world import World, Signal
from aion.cognition import Cognition
from aion.agents import Context


def vec(i=0):
    v=np.zeros(256,dtype=np.float32); v[i]=1; return v


def test_all_kinds_persist(tmp_path):
    p=tmp_path/'memory.db'; m=Memory(p)
    for kind in KINDS: m.put('a',kind,0,{'kind':kind},vec())
    m.put('b','semantic',0,{},vec())
    result=Memory(p).recall('a',vec(),limit=8,backend='numpy')
    assert {r['kind'] for r in result['items']}==set(KINDS)
    assert all(r['similarity']==pytest.approx(1) for r in result['items'])


def test_rank_capacity_validation(tmp_path):
    m=Memory(tmp_path/'m.db',capacity=2)
    m.put('a','working',0,{'old':True},vec())
    m.put('a','working',1,{'near':False},vec(1))
    m.put('a','working',2,{'near':True},vec())
    r=m.recall('a',vec(),kinds=('working',),backend='numpy')['items']
    assert len(r)==2 and r[0]['payload']=={'near':True}
    with pytest.raises(ValueError): m.put('a','unknown',0,{},vec())
    with pytest.raises(ValueError): m.put('a','working',0,{},[float('nan')]*256)


def test_faiss_optional(tmp_path):
    import importlib.util
    m=Memory(tmp_path/'m.db'); m.put('a','semantic',0,{},vec())
    r=m.recall('a',vec())
    assert r['backend']==('faiss' if importlib.util.find_spec('faiss') else 'numpy')
    if r['backend']=='numpy':
        with pytest.raises(ImportError): m.recall('a',vec(),backend='faiss')
    else: assert r['items']==m.recall('a',vec(),backend='numpy')['items']


def test_world_bayes_decay_restart(tmp_path):
    p=tmp_path/'w.db'; w=World(Memory(p))
    r=w.update('a',0,[Signal(key='marker',present=True,reliability=.9)])
    assert r['beliefs']['marker']['p']==pytest.approx(.9)
    w=World(Memory(p)); assert w.snapshot('a')==r
    r=w.update('a',1); b=r['beliefs']['marker']
    assert b['p']==pytest.approx(.88) and not b['observed_this_tick']
    r=w.update('a',2,[Signal(key='marker',present=False,reliability=.9)])
    assert r['beliefs']['marker']['p']<.5 and w.snapshot('b')['beliefs']=={}
    with pytest.raises(ValueError): w.update('a',2)


def test_pipeline_restart_and_neural(tmp_path):
    from aion.neural import ExpertBank
    p=tmp_path/'c.db'; c=Cognition(p,bank=ExpertBank())
    image=np.full((32,32,3),127,np.uint8)
    ctx=Context(id='test',options=('review',))
    r=c.ingest('a',0,image,ctx,ocr=False,neural_agent='A121')
    assert r['council']['roles_evaluated']==160 and r['council']['recommended'] is None
    assert len(r['neural']['output'])==256 and not r['neural']['trained']
    assert r['world']['beliefs']=={} and not r['game_control']
    c=Cognition(p); r=c.ingest('a',1,image,ctx,ocr=False)
    assert len(r['recalled']['items'])==5 and r['world']['tick']==1
