"""Phase 4: evidence-based deliberation, never an executable game policy."""
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .catalog import agents, networks, SUBSYSTEMS
from .contracts import Event, Proposal
from .bus import EventBus

# Map departments to relevant existing neural subsystem candidates.
GROUPS = (('percepcion_visual','reconocimiento_entidades'), ('cartografia','navegacion'),
          ('logica_causal',), ('planificacion',), ('combate',), ('economia',),
          ('compras',), ('prediccion',), ('estado_juego','navegacion'), ('cooperacion',),
          ('estado_juego','planificacion'), ('aprendizaje',), ('memoria',),
          ('metacognicion',), ('logica_causal','planificacion'), ('coordinacion',))
SPECS = agents()
NETWORKS = networks()

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

class Evidence(Model):
    id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=128)
    agent_id: str
    option: str
    value: float = Field(ge=-1, le=1)
    confidence: float = Field(gt=0, le=1)
    risk: float = Field(ge=0, le=1)
    observed: bool = True

class Context(Model):
    id: str = Field(min_length=1, max_length=128)
    options: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence: tuple[Evidence, ...] = Field(default=(), max_length=4096)

    @model_validator(mode='after')
    def validate_context(self):
        if len(set(self.options)) != len(self.options) or any(not x.strip() or len(x)>128 for x in self.options):
            raise ValueError('invalid options')
        ids = {s.id for s in SPECS}
        if len({e.id for e in self.evidence}) != len(self.evidence):
            raise ValueError('duplicate evidence id')
        if any(e.agent_id not in ids or e.option not in self.options for e in self.evidence):
            raise ValueError('unknown role or option')
        return self

@dataclass(frozen=True)
class Agent:
    id: str
    department: str
    role: str
    candidates: tuple[str, ...]

    def evaluate(self, context, evidence):
        """Independent role-scoped proposals; abstain without evidence."""
        results = []
        for option in context.options:
            items = [e for e in evidence if e.option == option and e.observed]
            # Repeated reports from one source cannot multiply its influence.
            by_source = defaultdict(list)
            for e in items:
                by_source[e.source].append(e)
            groups = list(by_source.values())
            if not groups:
                continue
            estimates = [sum(e.value*e.confidence for e in g)/sum(e.confidence for e in g) for g in groups]
            confidence = sum(max(e.confidence for e in g) for g in groups)/len(groups)
            spread = max(e.value for e in items)-min(e.value for e in items)
            confidence *= 1-spread/2
            results.append(Proposal(agent_id=self.id, observation_id=context.id, policy_version='evidence-v1',
                objective=option, horizon_seconds=1, utility=sum(estimates)/len(estimates),
                confidence=confidence, risk=max(e.risk for e in items), evidence=tuple(e.id for e in items)))
        return results

    def neural_diagnostic(self, bank, features):
        output, route = bank.infer(features, candidates=self.candidates, top_k=1)
        return output, route


def build_agents():
    departments = tuple(dict.fromkeys(s.department for s in SPECS))
    # Assign disjoint experts to roles within each department where possible.
    result = []
    for s in SPECS:
        d = departments.index(s.department)
        pool = tuple(n.id for n in NETWORKS if n.kind=='main' and n.subsystem in GROUPS[d])
        role_index = (int(s.id[1:])-1)%10
        result.append(Agent(s.id,s.department,s.role,pool[role_index::10]))
    return tuple(result)


def resolve(context, proposals):
    """Source-deduplicated evidence aggregation; no majority vote or action execution."""
    lookup = {e.id:e for e in context.evidence}
    scores, conflicts = [], []
    for option in context.options:
        selected = [p for p in proposals if p.objective == option and p.observation_id == context.id]
        groups = defaultdict(list)
        for key in sorted({key for p in selected for key in p.evidence}):
            e = lookup[key]
            if e.observed and e.option == option:
                groups[e.source].append(e)
        if not groups:
            continue
        all_e = [e for group in groups.values() for e in group]
        strong = [e for e in all_e if e.confidence >= .7]
        opposite = any(e.value >= .4 for e in strong) and any(e.value <= -.4 for e in strong)
        if opposite:
            conflicts.append({'option':option,'evidence':sorted(e.id for e in strong),'request':'independent corroboration'})
        values = [sum(e.value*e.confidence for e in group)/sum(e.confidence for e in group) for group in groups.values()]
        weights = [max(e.confidence for e in group) for group in groups.values()]
        utility = sum(v*w for v,w in zip(values,weights))/sum(weights)
        risk = max(e.risk for e in all_e)
        confidence = sum(weights)/len(weights)
        eligible = not opposite and risk < .8 and confidence >= .6 and utility > 0
        scores.append({'option':option,'utility':utility,'confidence':confidence,'risk':risk,
                       'sources':len(groups),'eligible':eligible,'score':utility*confidence-risk})
    ranked = sorted((s for s in scores if s['eligible'] and s['score']>0), key=lambda s:(-s['score'],s['option']))
    tied = len(ranked)>1 and abs(ranked[0]['score']-ranked[1]['score']) < 1e-9
    return {'recommended':ranked[0]['option'] if ranked and not tied else None,
            'scores':scores,'conflicts':conflicts,'tied':tied,'game_control':False}


class Council:
    """A bounded local event-driven council. One independent evaluation per role."""
    def __init__(self, workers=4):
        self.agents = build_agents()
        self.by_id = {a.id:a for a in self.agents}
        self.workers = workers
        self._lock = RLock()
        self.rounds = 0

    def deliberate(self, context):
        with self._lock:
            collected, messages = [], []
            lock = Lock()
            indexed = defaultdict(list)
            for e in context.evidence:
                indexed[e.agent_id].append(e)
            bus = EventBus(self.workers, len(self.agents))
            def receive(event):
                with lock:
                    messages.append(event.model_dump(mode='json'))
                    collected.extend(Proposal.model_validate(p) for p in event.payload['proposals'])
            def evaluate(agent, event):
                proposals = agent.evaluate(context,indexed[agent.id])
                message = Event(topic='council.proposals',source=agent.id,payload={'proposals':[p.model_dump(mode='json') for p in proposals]})
                # Return the event to coordinator, avoiding nested executor deadlock.
                return message
            try:
                bus.subscribe('council.proposals',receive)
                for agent in self.agents:
                    bus.subscribe('council.observe',lambda event,a=agent:evaluate(a,event))
                futures = bus.publish(Event(topic='council.observe',source='coordinator',payload={'context_id':context.id}))
                results = [f.result() for f in futures]
                for event in results:
                    for f in bus.publish(event): f.result()
            finally:
                bus.close()
            self.rounds += 1
            collected.sort(key=lambda p:(p.agent_id,p.objective))
            result = resolve(context,collected)
            result.update(round=self.rounds,roles_evaluated=len(messages),departments=len({a.department for a in self.agents}),
                          proposals=[p.model_dump(mode='json') for p in collected],
                          abstentions=len(self.agents)-len({p.agent_id for p in collected}),
                          message_count=len(messages),trained_experts=False)
            return result
