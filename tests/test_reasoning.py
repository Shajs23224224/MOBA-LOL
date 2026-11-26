import pytest
import numpy as np
from aion.reasoning import closure, abduce, induct, CausalModel
from aion.planning import Model, Outcome, mcts, beam, parallel_roots, hierarchical, expected_reward


def test_logic():
    rules=[(('a',),'b'),(('b','c'),'goal')]
    assert closure(['a','c'],rules)[0]=={'a','b','c','goal'}
    assert abduce(['a'],rules,'goal',['c','z'])==[('c',)]
    assert induct(8,2)==pytest.approx(.75)


def test_causal_counterfactual():
    model=CausalModel({'u':.3}, [('x',('u',),{(False,):False,(True,):True}),('y',('x','u'),{(False,False):False,(False,True):True,(True,False):True,(True,True):True})])
    assert model.probability({'y':True})==pytest.approx(.3)
    assert model.probability({'y':True},intervention={'x':True})==pytest.approx(1)
    assert model.probability({'y':True},evidence={'x':True},intervention={'x':False})==pytest.approx(1)
    assert model.probability({'y':True},intervention={'x':False})==pytest.approx(.3)
    with pytest.raises(ValueError): model.probability({'y':True},evidence={'x':True,'u':False})


def model():
    return Model({'s':{'small':[Outcome('end',1,(1.,0.))],'large':[Outcome('next',1,(0.,0.))]},'next':{'finish':[Outcome('end',1,(5.,1.))]}},(1.,.5))


def test_search():
    m=model()
    assert beam(m,'s')['path']==['large','finish']
    assert mcts(m,'s')['selected']=='large'
    assert parallel_roots(m,'s')[0]['value']==5.5
    assert parallel_roots(m,'s',workers=1)==parallel_roots(m,'s',workers=4)


def test_chance_and_bounds():
    o=[Outcome('end',.25,(4.,)),Outcome('end',.75,(0.,))]
    m=Model({'s':{'chance':o}})
    with pytest.raises(ValueError): beam(m,'s')
    assert mcts(m,'s',iterations=1024)['scores'][0]['value']==pytest.approx(1,abs=.2)
    with pytest.raises(ValueError): mcts(m,'s',depth=17)
    assert expected_reward(o,(1,))==pytest.approx(1)


def test_jax_expectation():
    pytest.importorskip('jax')
    assert expected_reward([Outcome('end',.5,(2.,)),Outcome('end',.5,(4.,))],(1,),backend='jax')==pytest.approx(3)


def test_hierarchy():
    assert hierarchical({'root':['a','b'],'a':['c']},'root')==['c','b']
    with pytest.raises(ValueError): hierarchical({'root':['root']},'root')


def test_integrated(tmp_path):
    from aion.cognition import Cognition
    from aion.agents import Context
    from aion.world import Signal
    from aion.research import analyze
    c=Cognition(tmp_path/'r.db')
    r=analyze(c,'scope',0,np.full((32,32,3),127,np.uint8),Context(id='test',options=('small','large')),model(),'s',signals=[Signal(key='marker',present=True,reliability=.9)],ocr=False)
    assert r['cognition']['council']['roles_evaluated']==160
    assert r['planning']['research_candidate']=='large' and r['planning']['corroborated_label'] is None
    assert r['planning']['belief_snapshot']['beliefs']['marker']['p']==pytest.approx(.9)
    assert any('research_plan' in x['payload'] for x in c.memory.recall('scope',r['cognition']['visual']['features'],kinds=('strategic',))['items'])
