import numpy as np
import pytest
from aion.moba import Snapshot, Item, Route, evaluate, review
from aion.agents import Evidence
from aion.cognition import Cognition
from aion.memory import KINDS
from aion.neural import ExpertBank
from aion.planning import parallel_roots

def sample(**changes):
    values=dict(health=.9,resource=.8,gold=100,threat=.2,ally_support=.8,objective_value=.9,
                farm_value=.4,shop_access=True,free_slots=1,confidence=.95,
                items=(Item(id='a',cost=50,offense=.8,defense=.2,sustain=.1),Item(id='b',cost=150,offense=1,defense=1,sustain=1)),
                routes=(Route(id='safe',coverage=.8,risk=.1,cost=.1),Route(id='risky',coverage=.8,risk=.9,cost=.1)))
    values.update(changes)
    return Snapshot(**values)

def test_economy_and_exploration():
    rows,model,_,_=evaluate(sample())
    by={r['option']:r for r in rows}
    assert 'buy:a' in by and 'buy:b' not in by
    assert by['explore:safe']['expected_utility']>by['explore:risky']['expected_utility']
    for changes in ({'shop_access':False},{'free_slots':0},{'gold':0}):
        assert not any(r['option'].startswith('buy:') for r in evaluate(sample(**changes))[0])
    assert parallel_roots(model,'start',depth=1,iterations=1024)[0]['option'] in ('objective','explore:safe','cooperate')

def test_tactical_resources_cooperation():
    low=sample(health=.1,resource=.2,threat=.9,ally_support=0.)
    rows,_,facts,_=evaluate(low)
    by={r['option']:r for r in rows}
    assert 'recovery_priority' in facts and 'caution_priority' in facts
    assert by['recover']['expected_utility']>by['objective']['expected_utility']
    high={r['option']:r for r in evaluate(sample())[0]}
    assert high['cooperate']['expected_utility']>by['cooperate']['expected_utility']

def test_validation_and_uncertainty():
    with pytest.raises(ValueError): sample(health=float('nan'))
    with pytest.raises(ValueError): sample(routes=(Route(id='a',coverage=1,risk=0,cost=0),)*2)
    low={r['option']:r for r in evaluate(sample(confidence=.1))[0]}
    high={r['option']:r for r in evaluate(sample())[0]}
    assert abs(low['hold']['probability']-.5)<abs(high['hold']['probability']-.5)

def test_integrated_review(tmp_path):
    c=Cognition(tmp_path/'m.db',bank=ExpertBank())
    image=np.full((32,32,3),127,np.uint8)
    for kind in ('semantic','procedural'):
        c.memory.put('scope',kind,0,{'fixture':True},np.ones(256))
    ev=Evidence(id='one',source='manual',agent_id='A031',option='objective',value=.8,confidence=.9,risk=.1)
    r=review(c,'scope',1,image,sample(),evidence=(ev,),neural_agent='A031',ocr=False)
    assert r['cognition']['council']['roles_evaluated']==160
    assert r['cognition']['council']['recommended']=='objective'
    assert len(r['cognition']['neural']['output'])==256
    assert set(r['moba']['memory_by_kind'])==set(KINDS)
    assert all(v['items'] for v in r['moba']['memory_by_kind'].values())
    assert r['moba']['forecast_only'] and not r['moba']['game_control']
    r=review(Cognition(tmp_path/'m.db'),'scope',2,image,sample(),ocr=False)
    assert r['cognition']['council']['recommended'] is None
    assert r['planning']['requires_review']
