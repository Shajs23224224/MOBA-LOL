from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, RLock

class EventBus:
    """Bounded local pub/sub. Each handler receives its own validated copy."""
    def __init__(self, workers, max_pending):
        self._pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='aion')
        self._slots=BoundedSemaphore(max_pending); self._lock=RLock(); self._handlers={}; self._closed=False
    def subscribe(self, topic, handler):
        with self._lock:
            if self._closed: raise RuntimeError('bus closed')
            self._handlers.setdefault(topic,[]).append(handler)
    def publish(self,event):
        with self._lock:
            if self._closed: raise RuntimeError('bus closed')
            handlers=tuple(self._handlers.get(event.topic,()))+tuple(self._handlers.get('*',())) if event.topic!='*' else tuple(self._handlers.get('*',()))
            acquired=0
            for _ in handlers:
                if not self._slots.acquire(blocking=False):
                    for _ in range(acquired): self._slots.release()
                    raise RuntimeError('event bus saturated')
                acquired+=1
            futures=[]
            try:
                for handler in handlers:
                    future=self._pool.submit(handler,event.model_copy(deep=True))
                    future.add_done_callback(lambda _:self._slots.release())
                    futures.append(future)
            except BaseException:
                for _ in range(acquired-len(futures)): self._slots.release()
                raise
            return futures
    def close(self):
        with self._lock: self._closed=True
        self._pool.shutdown(wait=True)
