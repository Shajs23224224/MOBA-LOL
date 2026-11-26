"""Phase 9: independent neural integrators; scores are uncalibrated diagnostics."""
import torch
from torch import nn
from .catalog import SUBSYSTEMS, networks

INPUT, LATENT, GLOBAL = 545, 64, 1092


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class Fusion(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.integrators = nn.ModuleDict()
            for i in range(16):
                self.integrators[f'I{2*i+1:02}'] = mlp(INPUT,128,LATENT)
                self.integrators[f'I{2*i+2:02}'] = mlp(INPUT+LATENT,64,4)
            self.heads = nn.ModuleDict({'F01':mlp(GLOBAL,128,16), 'F02':mlp(GLOBAL+16,128,3)})

    def forward(self, visual, summaries, experts, actor):
        if visual.shape != (1,256) or summaries.shape != (16,32) or actor.shape != (1,4):
            raise ValueError('invalid fusion dimensions')
        if not set(experts)<=set(SUBSYSTEMS) or any(t.shape!=(1,256) for t in experts.values()):
            raise ValueError('invalid expert input')
        if any(not torch.isfinite(t).all() for t in (visual,summaries,actor,*experts.values())):
            raise ValueError('nonfinite fusion input')
        representations, estimates = {}, {}
        for i, group in enumerate(SUBSYSTEMS):
            expert = experts.get(group,torch.zeros_like(visual))
            flag = visual.new_tensor([[float(group in experts)]])
            x = torch.cat((visual,expert,summaries[i:i+1],flag),dim=1)
            latent = self.integrators[f'I{2*i+1:02}'](x)
            raw = self.integrators[f'I{2*i+2:02}'](torch.cat((x,latent),dim=1))
            representations[group] = latent
            estimates[group] = torch.cat((raw[:,:1].tanh(),raw[:,1:].sigmoid()),dim=1)
        state = torch.cat([v for g in SUBSYSTEMS for v in (representations[g],estimates[g])] + [actor],dim=1)
        strategic = self.heads['F01'](state)
        policy = self.heads['F02'](torch.cat((state,strategic),dim=1))
        return {'representations':representations,'estimates':estimates,'strategic_logits':strategic,'policy_logits':policy}

    def audit(self):
        modules = {**dict(self.integrators.items()),**dict(self.heads.items())}
        params = [p for m in modules.values() for p in m.parameters()]
        if len({id(p) for p in params}) != len(params) or len({p.untyped_storage().data_ptr() for p in params}) != len(params):
            raise RuntimeError('shared parameters or storage detected')
        if set(modules) != {n.id for n in networks() if n.kind != 'main'}:
            raise RuntimeError('catalog mismatch')
        return {'intermediate_count':len(self.integrators),'final_count':len(self.heads),
                'parameters_by_network':{k:sum(p.numel() for p in m.parameters()) for k,m in modules.items()},
                'parameters':sum(p.numel() for p in params),'resident_bytes':sum(p.numel()*p.element_size() for p in params),
                'training_status':'not_established'}
