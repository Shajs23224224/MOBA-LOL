"""Finite research decision models, bounded searches and independent root evaluation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite, log, sqrt
from random import Random
from collections import defaultdict

@dataclass(frozen=True)
class Outcome:
    state: str
    probability: float
    reward: tuple[float,...]


class Model:
    def __init__(self, transitions, weights=(1.,)):
        self.weights = tuple(weights)
        self.transitions = {s:{a:tuple(o) for a,o in actions.items()} for s,actions in transitions.items()}
        if not 1<=len(weights)<=8 or not all(isfinite(w) and 0<=w<=1e6 for w in weights) or not sum(weights):
            raise ValueError('invalid weights')
        if len(transitions)>256 or any(len(a)>16 for a in transitions.values()):
            raise ValueError('model budget exceeded')
        for actions in self.transitions.values():
            for outcomes in actions.values():
                if not 1<=len(outcomes)<=16 or abs(sum(o.probability for o in outcomes)-1)>1e-8:
                    raise ValueError('invalid distribution')
                for o in outcomes:
                    if not 0<o.probability<=1 or len(o.reward)!=len(weights) or any(not isfinite(r) or abs(r)>1e6 for r in o.reward):
                        raise ValueError('invalid outcome')

    def actions(self,state):
        return self.transitions.get(state,{})

    def utility(self,outcome):
        return sum(w*r for w,r in zip(self.weights,outcome.reward))


def _bounds(depth):
    if type(depth) is not int or not 1<=depth<=16:
        raise ValueError('depth outside budget')


def mcts(model,start,*,depth=5,iterations=256,seed=42):
    """UCT with sampled chance outcomes, depth-bounded; statistics keyed by state/depth."""
    _bounds(depth)
    if type(iterations) is not int or not 16<=iterations<=1024:
        raise ValueError('iteration budget must be 16–1024')
    rng=Random(seed)
    visits=defaultdict(int); totals=defaultdict(float)
    for _ in range(iterations):
        state=start; path=[]
        for d in range(depth):
            actions=model.actions(state)
            if not actions: break
            keys=[(state,d,a) for a in sorted(actions)]
            unvisited=[k for k in keys if not visits[k]]
            k= rng.choice(unvisited) if unvisited else max(keys,key=lambda k:totals[k]/visits[k]+sqrt(2*log(sum(visits[q] for q in keys))/visits[k]))
            outcomes=actions[k[2]]
            o=rng.choices(outcomes,weights=[v.probability for v in outcomes],k=1)[0]
            path.append((k,model.utility(o))); state=o.state
        value=0.
        for k,reward in reversed(path):
            value+=reward; visits[k]+=1; totals[k]+=value
    scores=[{'option':a,'visits':visits[(start,0,a)],'value':totals[(start,0,a)]/visits[(start,0,a)] if visits[(start,0,a)] else None} for a in sorted(model.actions(start))]
    ranked=sorted((s for s in scores if s['visits']),key=lambda s:(-s['visits'],-s['value'],s['option']))
    return {'selected':ranked[0]['option'] if ranked else None,'scores':scores,'iterations':iterations}


def beam(model,start,*,depth=5,width=16):
    """Deterministic beam search; reject chance outcomes rather than selecting lucky futures."""
    _bounds(depth)
    if type(width) is not int or not 1<=width<=64:
        raise ValueError('invalid beam width')
    if any(len(o)!=1 for a in model.transitions.values() for o in a.values()):
        raise ValueError('beam requires deterministic model')
    frontier=[(0.,(),start)]
    expanded=0
    for _ in range(depth):
        new=[]
        for value,path,state in frontier:
            actions=model.actions(state)
            if not actions: new.append((value,path,state))
            for a,outcomes in actions.items():
                o=outcomes[0]; new.append((value+model.utility(o),path+(a,),o.state)); expanded+=1
        frontier=sorted(new,key=lambda x:(-x[0],x[1],x[2]))[:width]
    return {'path':list(frontier[0][1]),'value':frontier[0][0],'expanded':expanded}


def parallel_roots(model,start,*,depth=5,iterations=128,workers=4,seed=42):
    """Independent forced-root searches, parallel threads, deterministic per-root seeds."""
    _bounds(depth)
    if not 1<=workers<=8 or not 16<=iterations<=1024:
        raise ValueError('invalid parallel budget')
    actions=sorted(model.actions(start))
    def run(pair):
        i,a=pair
        # Synthetic root avoids restricting later visits to the original state.
        root='__root__'
        while root in model.transitions or any(o.state==root for actions in model.transitions.values() for outcomes in actions.values() for o in outcomes): root+='_' 
        restricted=Model(model.transitions,model.weights)
        restricted.transitions[root]={a:model.actions(start)[a]}
        result=mcts(restricted,root,depth=depth,iterations=iterations,seed=seed+i)
        return result['scores'][0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores=list(pool.map(run,enumerate(actions)))
    return sorted(scores,key=lambda s:(-s['value'],s['option']))


def hierarchical(tasks, goal):
    """Expand a supplied acyclic task decomposition to primitive labels."""
    def expand(name, stack):
        if name in stack or len(stack)>16:
            raise ValueError('cyclic or deep hierarchy')
        if name not in tasks: return [name]
        result=[]
        for child in tasks[name]:
            result.extend(expand(child,stack+(name,)))
            if len(result)>1024: raise ValueError('task budget exceeded')
        return result
    return expand(goal,())


def expected_reward(outcomes,weights,backend='numpy'):
    if backend=='jax':
        import jax.numpy as xp
    elif backend=='numpy':
        import numpy as xp
    else: raise ValueError('unknown backend')
    return float(xp.asarray([o.probability for o in outcomes]) @ (xp.asarray([o.reward for o in outcomes]) @ xp.asarray(weights)))
