"""Offline MOBA research evaluator. No game controls or automatic input."""
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .agents import Context
from .planning import Model, Outcome, expected_reward
from .reasoning import closure
from .research import analyze
from .memory import KINDS

class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

class Item(Input):
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,48}$')
    cost: int = Field(gt=0, le=100000)
    offense: float = Field(ge=0, le=1)
    defense: float = Field(ge=0, le=1)
    sustain: float = Field(ge=0, le=1)

class Route(Input):
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,48}$')
    coverage: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    cost: float = Field(ge=0, le=1)

class Snapshot(Input):
    health: float = Field(ge=0, le=1)
    resource: float = Field(ge=0, le=1)
    gold: int = Field(ge=0, le=100000)
    threat: float = Field(ge=0, le=1)
    ally_support: float = Field(ge=0, le=1)
    objective_value: float = Field(ge=0, le=1)
    farm_value: float = Field(ge=0, le=1)
    shop_access: bool
    free_slots: int = Field(ge=0, le=6)
    items: tuple[Item, ...] = Field(default=(), max_length=4)
    routes: tuple[Route, ...] = Field(default=(), max_length=4)
    confidence: float = Field(gt=0, le=1)

    @model_validator(mode='after')
    def unique(self):
        for entries in (self.items, self.routes):
            if len({x.id for x in entries}) != len(entries):
                raise ValueError('duplicate item or route')
        return self

WEIGHTS = (1., 1.5, 2.)

def clamp(x):
    return max(0., min(1., x))

def evaluate(s):
    """Heuristic estimates; not observed facts or calibrated probabilities."""
    facts = set()
    if min(s.health,s.resource)<.3: facts.add('low_resources')
    if s.threat>.7: facts.add('high_threat')
    if s.ally_support>=.6: facts.add('support_available')
    if s.shop_access and s.free_slots: facts.add('shopping_possible')
    known, trace = closure(facts, [(('low_resources',),'recovery_priority'),
        (('high_threat',),'caution_priority'), (('support_available',),'cooperation_possible')])
    rows = []
    def add(name, short, long, risk, probability, cost=0.):
        p = min(.99,max(.01,.5+(clamp(probability)-.5)*s.confidence))
        rows.append({'option':name,'short':clamp(short),'long':clamp(long),'risk':clamp(risk),
                     'probability':p,'cost':clamp(cost)})
    readiness=min(s.health,s.resource)
    add('hold',.08,.12,.05,.95)
    add('recover',1-readiness,.5*(1-readiness),.1,.9)
    add('farm',s.farm_value,.4*s.farm_value,s.threat*.65,.3+.6*readiness)
    add('objective',.3*s.objective_value,s.objective_value,s.threat*(1-.5*s.ally_support),.2+.5*readiness+.25*s.ally_support)
    add('cooperate',.3*s.ally_support,s.objective_value*s.ally_support,s.threat*(1-s.ally_support),.2+.7*s.ally_support)
    for item in s.items:
        if 'shopping_possible' in known and item.cost<=s.gold:
            benefit=(item.offense*(1-s.threat)+item.defense*s.threat+item.sustain*(1-readiness))/2
            add('buy:'+item.id,.4*benefit,benefit,.02,.98,item.cost/max(s.gold,1)*.25)
    for route in s.routes:
        add('explore:'+route.id,.2*route.coverage,route.coverage,route.risk,1-route.risk,route.cost*.25)
    transitions={}
    for row in rows:
        p=row['probability']; risk=row['risk']; cost=row['cost']
        transitions[row['option']]=[Outcome('end',p,(row['short'],row['long'],-cost)),
                                     Outcome('end',1-p,(0.,0.,-risk-cost))]
        row['expected_utility']=expected_reward(transitions[row['option']],WEIGHTS)
    return rows, Model({'start':transitions},WEIGHTS), sorted(known), trace

def review(cognition, scope, tick, image, snapshot, *, evidence=(), signals=(), neural_agent=None, ocr=True):
    rows, model, facts, trace = evaluate(snapshot)
    context=Context(id=f'moba-{tick}',options=tuple(r['option'] for r in rows),evidence=tuple(evidence))
    result=analyze(cognition,scope,tick,image,context,model,'start',signals=signals,
                   neural_agent=neural_agent,ocr=ocr,depth=1,iterations=256)
    features=result['cognition']['visual']['features']
    result['moba']={'estimates':rows,'derived_labels':facts,'logic_trace':trace,
                    'memory_by_kind':{k:cognition.memory.recall(scope,features,kinds=(k,),limit=2) for k in KINDS},
                    'forecast_only':True,'game_control':False}
    return result
