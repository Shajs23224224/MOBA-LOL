import json,time,threading
from collections import Counter
from pathlib import Path
import pytest
from pydantic import ValidationError
from aion.catalog import architecture,networks,agents
from aion.config import Settings,load_settings
from aion.contracts import Event,Proposal,ActionIntent
from aion.bus import EventBus
from aion.resources import ResourceBudget,MB
from aion.storage import Store
from aion.models import ModelRegistry,Artifact,digest
from aion.runtime import Engine

@pytest.fixture
def settings(tmp_path): return Settings(data_dir=tmp_path,device='cpu',heartbeat_seconds=.02,min_free_ram_mb=0)

def test_architecture():
    a=architecture(); assert len(networks())==len({x.id for x in networks()})==858
    assert Counter(x.kind for x in networks())=={'main':824,'intermediate':32,'final':2}
    assert all((x.hidden_layers,x.hidden_width)==(16,2048) for x in networks() if x.kind=='main')
    assert a['main_hidden_units']==27000832 and a['dense_hidden_to_hidden_weights']==51841597440
    assert a['materialized_main_networks']==0
    assert sorted(Counter(x.subsystem for x in networks() if x.kind=='main').values())==[51]*8+[52]*8

def test_agents():
    assert len(agents())==len({x.id for x in agents()})==160
    assert len(Counter(x.department for x in agents()))==16
    assert set(Counter(x.department for x in agents()).values())=={10}
    assert all(x.status=='planned' for x in agents())

def test_config(tmp_path):
    path=tmp_path/'config.toml'; path.write_text('data_dir="state"\nworkers=3\n')
    assert load_settings(path).data_dir==tmp_path/'state'
    assert load_settings(path).workers==3
    with pytest.raises(ValidationError): Settings(workers=0)
    with pytest.raises(ValidationError): Settings(environment='live_riot')
    with pytest.raises(ValidationError): Settings(unknown=True)

def test_contracts():
    with pytest.raises(ValidationError): Proposal(agent_id='A001',observation_id='x',policy_version='1',objective='test',horizon_seconds=1,confidence=1.1,risk=0,utility=1)
    with pytest.raises(ValidationError): ActionIntent(kind='attack',observation_id='x')

def test_bus_copy():
    bus=EventBus(2,3); seen=[]
    def mutate(e): e.payload['x']=2
    bus.subscribe('x',mutate); bus.subscribe('x',lambda e:seen.append(e.payload['x']))
    e=Event(topic='x',source='test',payload={'x':1})
    try:
        for f in bus.publish(e): f.result(2)
        assert seen==[1] and e.payload['x']==1
    finally: bus.close()
    with pytest.raises(RuntimeError): bus.publish(e)

def test_bus_backpressure():
    bus=EventBus(1,1); gate=threading.Event()
    bus.subscribe('x',lambda e:gate.wait(3)); e=Event(topic='x',source='test')
    try:
        futures=bus.publish(e)
        with pytest.raises(RuntimeError,match='saturated'): bus.publish(e)
        gate.set(); futures[0].result(2)
    finally: gate.set(); bus.close()

def test_bus_handler_error():
    bus=EventBus(1,1)
    def fail(e): raise ValueError('handler')
    bus.subscribe('x',fail)
    try:
        for _ in range(2):
            f=bus.publish(Event(topic='x',source='test'))[0]
            with pytest.raises(ValueError): f.result(2)
    finally: bus.close()

def test_resource_budget(settings):
    budget=ResourceBudget(settings,lambda *_:{'ram_available_bytes':1000*MB,'cuda_devices':[]})
    with budget.reserve(100):
        assert budget.reserved_mb==100
        with pytest.raises(RuntimeError):
            with budget.reserve(200): pass
    assert budget.reserved_mb==0 and budget.choose_device()=='cpu'
    with pytest.raises(ValueError):
        with budget.reserve(-1): pass
    with pytest.raises(RuntimeError): ResourceBudget(settings.model_copy(update={'device':'cuda'}),lambda *_:{'cuda_devices':[]}).choose_device()

def test_storage(tmp_path):
    path=tmp_path/'db.sqlite'; s=Store(path); s.start('session'); s.record('session',Event(topic='x',source='test')); s.finish('session','stopped')
    assert Store(path).summary()=={'sessions':[{'id':'session','status':'stopped'}],'events':1}

def test_model_registry_tamper(tmp_path):
    reg=ModelRegistry(tmp_path); p=tmp_path/'x.safetensors'; p.write_bytes(b'checksum only test')
    entry=Artifact(id='x',filename=p.name,sha256=digest(p),version='1',source='test',license='test',purpose='test')
    reg.register(entry); assert reg.audit()[0]['status']=='checksum_verified'
    p.write_bytes(b'tampered')
    assert reg.audit()[0]['status'].startswith('invalid')
    with pytest.raises(ValueError): reg.verify(entry)
    with pytest.raises(ValueError): reg.path_for('../x.safetensors')
    with pytest.raises(ValueError): reg.path_for('x.pkl')

def test_neural_probe():
    pytest.importorskip('torch'); pytest.importorskip('safetensors')
    from aion.probe import neural_probe
    result=neural_probe('cpu'); assert result['passed'] and result['diagnostic_parameters']==212 and result['production_networks']==0 and not result['trained']

def test_engine_lifecycle(settings):
    e=Engine(settings)
    for _ in range(2):
        e.start()
        try:
            assert e.snapshot()['state']=='running'
            with pytest.raises(RuntimeError): e.start()
            time.sleep(.09)
        finally: assert e.stop()=='stopped'
    saved=Store(settings.data_dir/'aion.sqlite3').summary()
    assert len(saved['sessions'])==2 and all(x['status']=='stopped' for x in saved['sessions']) and saved['events']>=6
    assert json.loads((settings.data_dir/'architecture.json').read_text())['network_count']==858

def test_engine_start_failure(settings):
    e=Engine(settings); e.resources.probe=lambda *_:{'cuda_devices':[],'ram_available_bytes':0}
    with pytest.raises(RuntimeError): e.start()
    assert e.snapshot()['state']=='failed' and e.stop()=='failed'
