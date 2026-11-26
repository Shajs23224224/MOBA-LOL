"""Real tensor/checkpoint smoke test, NOT a production expert or trained model."""
from pathlib import Path
from tempfile import TemporaryDirectory
from .models import ModelRegistry, Artifact, digest

def neural_probe(device='cpu'):
    import torch
    from safetensors.torch import save_file
    # A tiny deterministic diagnostic network, excluded from all AION model counts.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,4))
    model=model.to(device).eval()
    x=torch.ones(2,8,device=device)
    with torch.inference_mode(): expected=model(x).cpu()
    with TemporaryDirectory() as root:
        path=Path(root)/'diagnostic.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path))
        registry=ModelRegistry(root)
        registry.register(Artifact(id='diagnostic',filename=path.name,sha256=digest(path),version='1',source='locally generated diagnostic',license='project diagnostic',purpose='smoke test only'))
        loaded=registry.load('diagnostic')
        model.load_state_dict(loaded,strict=True)
        with torch.inference_mode(): actual=model(x).cpu()
        return {'passed':bool(torch.equal(expected,actual) and torch.isfinite(actual).all()),'device':device,'output_shape':list(actual.shape),'diagnostic_parameters':sum(p.numel() for p in model.parameters()),'production_networks':0,'trained':False}
