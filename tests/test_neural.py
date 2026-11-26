import pytest
import torch
from aion.neural import Expert, ExpertBank, EXPERT_IDS, INPUT, WIDTH, DEPTH

@pytest.fixture(scope='module')
def bank():
    return ExpertBank()

def test_all_blueprints(bank):
    assert len(EXPERT_IDS) == len(set(EXPERT_IDS)) == len(bank.blueprints) == 824
    assert all(len(e.layers) == DEPTH and all(l.out_features == WIDTH for l in e.layers) for e in bank.blueprints.values())
    assert all(p.is_meta for p in bank.blueprints.parameters())
    assert len({id(e.layers[0].weight) for e in bank.blueprints.values()}) == 824
    expected = 256*2048 + 15*2048**2 + 16*2048 + 2*16*2048 + 2048*256 + 256
    assert bank.audit()['logical_expert_parameters'] == expected*824
    assert bank.audit()['resident_experts'] == []

def test_inference_full_width_and_eviction(bank):
    x = torch.ones(1, INPUT)
    a, r = bank.infer(x, candidates=['N001'])
    assert a.shape == (1, 256) and torch.isfinite(a).all() and r['selected'] == ['N001']
    assert bank.audit()['resident_experts'] == ['N001']
    b, r = bank.infer(x, top_k=2, candidates=['N001','N824'])
    assert torch.isfinite(b).all() and len(r['selected']) == 2 and sum(r['weights']) == pytest.approx(1)
    assert len(bank.audit()['resident_experts']) == 1
    again, _ = bank.infer(x, candidates=['N001'])
    assert torch.equal(a, again)

def test_invalid_inputs_and_budget(bank):
    with pytest.raises(ValueError): bank.infer(torch.ones(1, 2))
    with pytest.raises(ValueError): bank.infer(torch.ones(1, INPUT), candidates=['N999'])
    old = bank.max_weight_bytes
    try:
        bank.max_weight_bytes = 1
        with pytest.raises(MemoryError): bank.infer(torch.ones(1, INPUT), candidates=['N003'])
    finally: bank.max_weight_bytes = old

def test_real_gradients():
    # One full-sized expert, independent from the inference bank.
    model = Expert()
    model(torch.ones(1, INPUT)).square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert torch.isfinite(model.layers[0].weight.grad).all()
