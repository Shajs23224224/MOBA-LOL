"""Generic binary belief filter for supplied, explicitly scoped observations."""
import math
from contextlib import closing
from pydantic import BaseModel, ConfigDict, Field


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    key: str = Field(min_length=1,max_length=128)
    present: bool
    reliability: float = Field(gt=.5,lt=1)


class World:
    def __init__(self, memory, *, retention=.95):
        if not 0<retention<1:
            raise ValueError('invalid retention')
        self.memory, self.retention = memory, retention
        with closing(memory.connect()) as c, c:
            c.execute('CREATE TABLE IF NOT EXISTS world(scope TEXT PRIMARY KEY,tick INTEGER NOT NULL,state TEXT NOT NULL)')

    def snapshot(self, scope):
        import json
        with closing(self.memory.connect()) as c:
            row = c.execute('SELECT tick,state FROM world WHERE scope=?',(scope,)).fetchone()
        return {'tick':row[0],'beliefs':json.loads(row[1])} if row else {'tick':-1,'beliefs':{}}

    def update(self, scope, tick, signals=()):
        import json
        if not scope or len(scope)>128 or type(tick) is not int or tick<0:
            raise ValueError('invalid scope or tick')
        signals = tuple(signals)
        if len(signals)>256 or len({s.key for s in signals})!=len(signals):
            raise ValueError('duplicate or excessive signals')
        with closing(self.memory.connect()) as c, c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT tick,state FROM world WHERE scope=?',(scope,)).fetchone()
            previous, beliefs = (row[0],json.loads(row[1])) if row else (-1,{})
            if tick<=previous:
                raise ValueError('tick must increase')
            for b in beliefs.values():
                b['p'] = .5+(b['p']-.5)*self.retention**(tick-previous)
                b['observed_this_tick'] = False
            for s in signals:
                prior = beliefs.get(s.key,{}).get('p',.5)
                likelihood = s.reliability if s.present else 1-s.reliability
                posterior = prior*likelihood/(prior*likelihood+(1-prior)*(1-likelihood))
                beliefs[s.key] = {'p':min(1-1e-9,max(1e-9,posterior)), 'last_observed_tick':tick,
                                  'observed_this_tick':True,'reported_present':s.present}
            if len(beliefs)>4096:
                raise ValueError('world belief limit exceeded')
            c.execute('INSERT INTO world VALUES(?,?,?) ON CONFLICT(scope) DO UPDATE SET tick=excluded.tick,state=excluded.state',
                      (scope,tick,json.dumps(beliefs,allow_nan=False)))
        return {'tick':tick,'beliefs':beliefs}
