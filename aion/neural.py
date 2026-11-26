"""Phase 2: independent dense experts; meta blueprints are not allocated weights."""
from collections import OrderedDict
from threading import RLock
import torch
from torch import nn
from .catalog import networks
from .resources import hardware

INPUT = OUTPUT = 256
WIDTH, DEPTH = 2048, 16
EXPERT_IDS = tuple(n.id for n in networks() if n.kind == 'main')

class Expert(nn.Module):
    """16 hidden affine layers, GELU, LayerNorm and residuals; 256-d output."""
    def __init__(self, *, device=None):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(INPUT if i == 0 else WIDTH, WIDTH, device=device) for i in range(DEPTH))
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH, device=device) for _ in range(DEPTH))
        self.head = nn.Linear(WIDTH, OUTPUT, device=device)
        self.activation = nn.GELU()

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            y = self.activation(norm(layer(x)))
            x = y if i == 0 else x + y
        return self.head(x)

class ExpertBank:
    """Bounded, synchronous inference service. Untrained until real checkpoints are supplied."""
    def __init__(self, *, capacity=1, max_weight_bytes=512*1024**2, seed=42, registry=None, device='cpu'):
        if not 1 <= capacity <= len(EXPERT_IDS) or max_weight_bytes <= 0:
            raise ValueError('invalid capacity or weight budget')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('CPU or CUDA required')
        self.capacity, self.max_weight_bytes, self.seed = capacity, max_weight_bytes, seed
        self.registry = registry
        self.blueprints = nn.ModuleDict({key: Expert(device='meta') for key in EXPERT_IDS})
        self.cache = OrderedDict()
        self.lock = RLock()
        # Local generator preserves global RNG; router is intentionally untrained.
        self.router = nn.Linear(INPUT, len(EXPERT_IDS), device='meta').to_empty(device='cpu')
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.router.weight.uniform_(-INPUT**-.5, INPUT**-.5, generator=g)
            self.router.bias.zero_()
        self.router.eval()
        self.expert_params = sum(p.numel() for p in self.blueprints[EXPERT_IDS[0]].parameters())
        self.weight_bytes = self.expert_params * 4

    def _materialize(self, key):
        if key not in self.blueprints:
            raise KeyError(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.weight_bytes > self.max_weight_bytes:
            raise MemoryError('one full expert exceeds weight budget')
        while self.cache and (len(self.cache) >= self.capacity or (len(self.cache)+1)*self.weight_bytes > self.max_weight_bytes):
            self.cache.popitem(last=False)
        # CPU initialization and optional checkpoint load require extra temporary RAM.
        if hardware(False)['ram_available_bytes'] < 2*self.weight_bytes:
            raise MemoryError('insufficient available RAM for expert allocation')
        if self.device.type == 'cuda' and torch.cuda.mem_get_info(self.device)[0] < 2*self.weight_bytes:
            raise MemoryError('insufficient available VRAM for expert allocation')
        model = Expert(device='meta').to_empty(device='cpu')
        g = torch.Generator().manual_seed(self.seed + int(key[1:]))
        with torch.no_grad():
            for layer in (*model.layers, model.head):
                bound = layer.in_features**-.5
                layer.weight.uniform_(-bound, bound, generator=g)
                layer.bias.uniform_(-bound, bound, generator=g)
            for norm in model.norms:
                norm.weight.fill_(1)
                norm.bias.zero_()
        if self.registry is not None and key in self.registry.entries():
            model.load_state_dict(self.registry.load(key), strict=True)
        model = model.to(self.device).eval()
        self.cache[key] = model
        return model

    @torch.inference_mode()
    def infer(self, x, *, top_k=1, candidates=None):
        """One observation at a time; select only permitted expert IDs, aggregate outputs."""
        if x.shape != (1, INPUT) or not torch.isfinite(x).all():
            raise ValueError('expected finite observation of shape (1, 256)')
        ids = EXPERT_IDS if candidates is None else tuple(candidates)
        if len(set(ids)) != len(ids) or any(key not in self.blueprints for key in ids) or not 1 <= top_k <= len(ids):
            raise ValueError('invalid candidate set or top_k')
        with self.lock:
            indices = [int(key[1:])-1 for key in ids]
            logits = self.router(x.to(device='cpu', dtype=torch.float32))[0, indices]
            values, order = logits.topk(top_k)
            weights = values.softmax(0)
            selected = [ids[i] for i in order.tolist()]
            result = torch.zeros((1, OUTPUT), device=self.device)
            for key, weight in zip(selected, weights):
                model = self._materialize(key)
                result.add_(model(x.to(device=self.device, dtype=torch.float32)) * weight.to(self.device))
                del model
            return result.cpu(), {'selected': selected, 'weights': weights.tolist(), 'active_expert_parameters': top_k*self.expert_params, 'router_parameters': sum(p.numel() for p in self.router.parameters())}

    def audit(self):
        with self.lock:
            total = sum(p.numel() for p in self.blueprints.parameters())
            resident = sum(p.numel() for m in self.cache.values() for p in m.parameters())
            router = sum(p.numel() for p in self.router.parameters())
            return {'expert_blueprints': len(self.blueprints), 'hidden_layers_per_expert': DEPTH, 'hidden_width': WIDTH, 'expert_parameters_each': self.expert_params, 'logical_expert_parameters': total, 'router_parameters': router, 'logical_total_parameters': total+router, 'resident_experts': list(self.cache), 'resident_parameters': resident+router, 'meta_blueprint_parameters': total, 'trained_status': 'not_established', 'resident_weight_bytes': (resident+router)*4}
