import pytest
from aion.agents import Council, Context, Evidence, build_agents


def e(i, role='A001', option='review', value=.8, confidence=.9, risk=.1, source='source1', observed=True):
    return Evidence(id=str(i),agent_id=role,option=option,value=value,confidence=confidence,risk=risk,source=source,observed=observed)


def test_all_roles_and_candidates():
    agents = build_agents()
    assert len(agents) == len({a.id for a in agents}) == 160
    assert len({a.department for a in agents}) == 16
    assert all(a.candidates for a in agents)
    assert len(set().union(*(set(a.candidates) for a in agents))) == 824
    ctx = Context(id='all',options=('review',),evidence=tuple(e(i,role=a.id,source=f's{i}') for i,a in enumerate(agents)))
    r = Council().deliberate(ctx)
    assert r['roles_evaluated']==r['message_count']==len(r['proposals'])==160
    assert r['abstentions']==0 and r['recommended']=='review'
    assert len({p['agent_id'] for p in r['proposals']})==160


def test_abstention_and_rounds():
    c = Council()
    ctx = Context(id='none',options=('review',),evidence=(e(1,observed=False),))
    r = c.deliberate(ctx)
    assert r['abstentions']==160 and r['recommended'] is None and not r['game_control']
    assert c.deliberate(ctx)['round']==2


def test_contradiction_and_risk():
    ctx = Context(id='conflict',options=('review','hold','defer'),evidence=(e(1),e(2,role='A002',value=-.8,source='source2'),e(3,option='hold',risk=.9),e(4,option='defer')))
    r = Council().deliberate(ctx)
    assert r['recommended']=='defer' and r['conflicts'][0]['option']=='review'
    assert not next(s for s in r['scores'] if s['option']=='hold')['eligible']


def test_dedup_and_no_majority():
    data = tuple(e(i,role=f'A{i+1:03}',value=.2,confidence=.3,source='same') for i in range(100))+(e(101,role='A160',value=-1,confidence=.99,source='independent'),)
    r = Council().deliberate(Context(id='source',options=('review',),evidence=data))
    assert r['scores'][0]['sources']==2 and r['scores'][0]['utility']<0
    assert r['recommended'] is None


def test_ties_invalid_evidence_and_hypotheses():
    c=Council()
    ctx=Context(id='tie',options=('review','hold'),evidence=(e(1),e(2,option='hold'),e(3,value=-1,observed=False)))
    r=c.deliberate(ctx)
    assert r['tied'] and r['recommended'] is None and not r['conflicts']
    with pytest.raises(ValueError): Context(id='x',options=('review',),evidence=(e(1),e(1)))
    with pytest.raises(ValueError): Context(id='x',options=('review',),evidence=(e(1,role='A999'),))


def test_real_expert_bridge():
    import torch
    from aion.neural import ExpertBank, INPUT
    agent = build_agents()[0]
    output, route = agent.neural_diagnostic(ExpertBank(),torch.ones(1, INPUT))
    assert output.shape==(1,256) and torch.isfinite(output).all()
    assert route['selected'][0] in agent.candidates
