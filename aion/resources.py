from contextlib import contextmanager
from threading import Lock
import os
import psutil

MB=1024**2

def hardware(include_torch=True):
    vm=psutil.virtual_memory()
    result={'cpu_logical':os.cpu_count() or 1,'ram_total_bytes':vm.total,'ram_available_bytes':vm.available,'torch_available':False,'cuda_devices':[]}
    try:
        # Account for cgroup v2 limits when visible in a container.
        from pathlib import Path
        cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used=int(Path('/sys/fs/cgroup/memory.current').read_text())
        if cap != 'max':
            result['ram_total_bytes']=min(vm.total,int(cap))
            result['ram_available_bytes']=min(vm.available,max(0,int(cap)-used))
    except (OSError,ValueError): pass
    if include_torch:
        try:
            import torch
            result['torch_available']=True
            result['torch_version']=torch.__version__
            for i in range(torch.cuda.device_count()):
                free,total=torch.cuda.mem_get_info(i)
                result['cuda_devices'].append({'index':i,'name':torch.cuda.get_device_name(i),'free_bytes':free,'total_bytes':total})
        except (ImportError, OSError, RuntimeError) as e: result['torch_error']=str(e)
    return result

class ResourceBudget:
    """Cooperative admission budget, not OS-enforced memory isolation."""
    def __init__(self, settings, probe=hardware):
        self.settings=settings; self.probe=probe; self._reserved=0; self._lock=Lock()
    @property
    def reserved_mb(self):
        with self._lock: return self._reserved
    def choose_device(self):
        info=self.probe()
        if self.settings.device=='cuda' and not info['cuda_devices']:
            raise RuntimeError('CUDA requested but not available')
        return 'cuda:0' if self.settings.device!='cpu' and info['cuda_devices'] else 'cpu'
    @contextmanager
    def reserve(self, mb):
        if mb <= 0: raise ValueError('reservation must be positive')
        with self._lock:
            available=self.probe(False)['ram_available_bytes']//MB
            if self._reserved+mb>self.settings.max_reserved_mb or available-self._reserved-mb<self.settings.min_free_ram_mb:
                raise RuntimeError('resource budget exceeded')
            self._reserved+=mb
        try: yield
        finally:
            with self._lock: self._reserved-=mb
