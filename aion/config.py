from pathlib import Path
import tomllib
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    data_dir: Path = Path('data')
    workers: int = Field(default=2,ge=1,le=64)
    max_pending: int = Field(default=32,ge=1,le=4096)
    min_free_ram_mb: int = Field(default=128,ge=0)
    max_reserved_mb: int = Field(default=256,ge=1)
    device: Literal['auto','cpu','cuda'] = 'auto'
    environment: Literal['diagnostics'] = 'diagnostics'
    heartbeat_seconds: float = Field(default=0.2,gt=0,le=60)

def load_settings(path=None):
    if path is None: return Settings()
    path = Path(path).resolve()
    with path.open('rb') as file: raw = tomllib.load(file)
    if 'data_dir' in raw:
        raw['data_dir'] = path.parent / raw['data_dir']
    return Settings.model_validate(raw)
