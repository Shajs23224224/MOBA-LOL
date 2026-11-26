"""Bounded, explicit research reasoning; no learned causal discovery or game actions."""
from itertools import product, combinations
from math import prod


def closure(facts, rules):
    """Finite positive Horn rules: (premises, conclusion)."""
    known = set(facts)
    rules = [(frozenset(a), b) for a,b in rules]
    trace = []
    changed = True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if premises <= known and conclusion not in known:
                known.add(conclusion)
                trace.append((tuple(sorted(premises)), conclusion))
                changed = True
    return known, trace


def abduce(facts, rules, goal, candidates):
    """Find minimal sufficient hypotheses, not proven facts."""
    candidates = tuple(sorted(set(candidates)-set(facts)))
    if len(candidates)>12:
        raise ValueError('abduction budget exceeded')
    found = []
    for n in range(len(candidates)+1):
        for subset in combinations(candidates,n):
            if any(set(h)<=set(subset) for h in found):
                continue
            if goal in closure(set(facts)|set(subset),rules)[0]:
                found.append(subset)
    return found


def induct(successes, failures):
    """Beta(1,1) posterior mean for supplied independent Bernoulli observations."""
    if any(type(x) is not int or x<0 for x in (successes,failures)):
        raise ValueError('invalid counts')
    return (successes+1)/(successes+failures+2)


class CausalModel:
    """Explicit binary structural equations, independent exogenous roots, exact enumeration.
    Equations are topologically ordered (name, parents, truth table indexed by Boolean tuple).
    Counterfactuals reuse the same exogenous assignment after conditioning on factual evidence.
    """
    def __init__(self, roots, equations):
        self.roots = dict(roots)
        self.equations = tuple((n,tuple(p),dict(t)) for n,p,t in equations)
        if len(roots)>12 or len(self.equations)>64 or any(not 0<=p<=1 for p in roots.values()):
            raise ValueError('invalid causal model budget or probabilities')
        known = set(roots)
        for name, parents, table in self.equations:
            if name in known or not set(parents)<=known or len(parents)>10 or len(set(parents))!=len(parents):
                raise ValueError('invalid topological order or parents')
            if set(table)!=set(product((False,True),repeat=len(parents))) or any(type(v) is not bool for v in table.values()):
                raise ValueError('incomplete Boolean truth table')
            known.add(name)
        self.variables = known

    def _run(self, noise, intervention):
        values = {**noise, **{k:v for k,v in intervention.items() if k in self.roots}}
        for name, parents, table in self.equations:
            values[name] = intervention[name] if name in intervention else table[tuple(values[p] for p in parents)]
        return values

    def probability(self, query, *, evidence=None, intervention=None):
        evidence, intervention = evidence or {}, intervention or {}
        for mapping in (query,evidence,intervention):
            if not set(mapping)<=self.variables or any(type(v) is not bool for v in mapping.values()):
                raise ValueError('unknown or non-Boolean variable')
        numerator = denominator = 0.
        for bits in product((False,True),repeat=len(self.roots)):
            noise = dict(zip(self.roots,bits))
            weight = prod(p if noise[k] else 1-p for k,p in self.roots.items())
            factual = self._run(noise,{})
            if not all(factual[k]==v for k,v in evidence.items()):
                continue
            denominator += weight
            hypothetical = self._run(noise,intervention)
            if all(hypothetical[k]==v for k,v in query.items()):
                numerator += weight
        if denominator==0:
            raise ValueError('impossible evidence')
        return numerator/denominator
