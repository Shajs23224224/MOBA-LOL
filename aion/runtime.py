from threading import Event as StopEvent, Thread, RLock
from uuid import uuid4
import logging
from .bus import EventBus
from .contracts import Event
from .storage import Store
from .resources import ResourceBudget
from .models import ModelRegistry, atomic_json
from .catalog import architecture

log=logging.getLogger(__name__)
class Engine:
    """Phase 1 lifecycle. No screen capture, game adapter or game actions."""
    def __init__(self,settings):
        self.settings=settings; self.resources=ResourceBudget(settings); self.store=Store(settings.data_dir/'aion.sqlite3'); self.models=ModelRegistry(settings.data_dir/'models'); self.state='idle'; self.session_id=None; self._lock=RLock(); self._stop=StopEvent(); self._thread=None; self._bus=None; self.error=None; self.device=None
    def _emit(self,topic,payload=None):
        event=Event(topic=topic,source='engine',payload=payload or {})
        for future in self._bus.publish(event): future.result()
    def start(self):
        with self._lock:
            if self.state not in ('idle','stopped','failed') or (self._thread and self._thread.is_alive()): raise RuntimeError('engine already active')
            self.state='starting'; self.error=None; self.session_id=None; self.device=None; self._bus=None; self._stop.clear()
            try:
                self.device=self.resources.choose_device()
                with self.resources.reserve(1):
                    atomic_json(self.settings.data_dir/'architecture.json',architecture())
                self.session_id=uuid4().hex; self.store.start(self.session_id)
                self._bus=EventBus(self.settings.workers,self.settings.max_pending)
                sid=self.session_id
                self._bus.subscribe('*',lambda event:self.store.record(sid,event))
                self._emit('engine.started',{'device':self.device,'mode':'diagnostics','game_control':False})
                self.state='running'; self._thread=Thread(target=self._run,name='aion-engine',daemon=False); self._thread.start()
            except BaseException as exc:
                self.error=str(exc); self.state='failed'
                if self._bus: self._bus.close()
                if self.session_id: self.store.finish(self.session_id,'failed')
                raise
        return self.session_id
    def _run(self):
        status='stopped'
        try:
            while not self._stop.wait(self.settings.heartbeat_seconds):
                self._emit('engine.heartbeat',{'reserved_mb':self.resources.reserved_mb})
            self._emit('engine.stopped')
        except Exception as exc:
            status='failed'; self.error=str(exc); log.exception('engine worker failed')
        finally:
            try:
                self._bus.close(); self.store.finish(self.session_id,status)
            except Exception as exc:
                status='failed'; self.error=str(exc); log.exception('engine cleanup failed')
            with self._lock: self.state=status
    def stop(self,timeout=15):
        with self._lock:
            if self.state=='running': self.state='stopping'
            self._stop.set(); thread=self._thread
        if thread and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive(): raise TimeoutError('engine did not stop before timeout')
        return self.state
    def snapshot(self):
        with self._lock:
            return {'state':self.state,'session_id':self.session_id,'device':self.device,'error':self.error,'mode':'diagnostics','game_control':False,'materialized_main_networks':0,'functional_expert_agents':0}
