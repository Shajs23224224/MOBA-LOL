from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, JsonValue

class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    id: str = Field(default_factory=lambda: uuid4().hex)
    topic: str = Field(min_length=1)
    source: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, JsonValue] = Field(default_factory=dict)

class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    id: str
    tick: int = Field(ge=0)
    timestamp: datetime
    facts: tuple[str,...] = ()
    hypotheses: tuple[str,...] = ()

class Proposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    agent_id: str
    observation_id: str
    policy_version: str
    objective: str
    horizon_seconds: float = Field(gt=0)
    confidence: float = Field(ge=0,le=1)
    risk: float = Field(ge=0,le=1)
    utility: float
    evidence: tuple[str,...] = ()

class ActionIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    kind: Literal['noop'] = 'noop'  # no live control implemented
    observation_id: str
