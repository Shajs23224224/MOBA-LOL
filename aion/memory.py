"""Scoped persistent research memory; retrieval is similarity, not truth."""
import json
import sqlite3
from pathlib import Path
from contextlib import closing
from uuid import uuid4
import numpy as np

KINDS = ('sensory','working','episodic','semantic','procedural','strategic','predictive','metacognitive')
DIM = 256


def unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    if v.shape != (DIM,) or not np.isfinite(v).all():
        raise ValueError('expected 256 finite features')
    norm = float(np.linalg.norm(v.astype(np.float64)))
    return (v / norm).astype('<f4') if norm else v.astype('<f4')


class Memory:
    def __init__(self, path, *, capacity=256):
        if not 1 <= capacity <= 4096:
            raise ValueError('invalid capacity per scope and kind')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        with closing(self.connect()) as c, c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('''CREATE TABLE IF NOT EXISTS memory(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL, scope TEXT NOT NULL, kind TEXT NOT NULL, tick INTEGER NOT NULL,
                payload TEXT NOT NULL, vector BLOB NOT NULL)''')
            c.execute('CREATE INDEX IF NOT EXISTS memory_scope ON memory(scope,kind,seq)')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def put(self, scope, kind, tick, payload, vector):
        if not scope or len(scope)>128 or kind not in KINDS or type(tick) is not int or tick<0:
            raise ValueError('invalid memory metadata')
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(data.encode()) > 2_000_000:
            raise ValueError('payload too large')
        record_id = uuid4().hex
        blob = unit(vector).tobytes()
        with closing(self.connect()) as c, c:
            c.execute('INSERT INTO memory(id,scope,kind,tick,payload,vector) VALUES(?,?,?,?,?,?)',
                      (record_id,scope,kind,tick,data,blob))
            c.execute('''DELETE FROM memory WHERE scope=? AND kind=? AND seq NOT IN
                (SELECT seq FROM memory WHERE scope=? AND kind=? ORDER BY seq DESC LIMIT ?)''',
                (scope,kind,scope,kind,self.capacity))
        return record_id

    def recall(self, scope, vector, *, kinds=KINDS, limit=5, backend='auto'):
        if not kinds or any(k not in KINDS for k in kinds) or not 1<=limit<=100 or backend not in ('auto','numpy','faiss'):
            raise ValueError('invalid retrieval settings')
        q = unit(vector)
        with closing(self.connect()) as c:
            rows = c.execute('SELECT id,kind,tick,payload,vector FROM memory WHERE scope=? AND kind IN (' + ','.join('?' for _ in kinds)+') ORDER BY seq DESC', (scope,*kinds)).fetchall()
        if not rows:
            return {'backend':'none','items':[]}
        matrix = np.stack([np.frombuffer(r[4],dtype='<f4') for r in rows]).astype(np.float32)
        selected = 'numpy'
        if backend != 'numpy':
            try:
                import faiss
            except ImportError:
                if backend == 'faiss':
                    raise
            else:
                index = faiss.IndexFlatIP(DIM)
                index.add(matrix)
                scores, indices = index.search(q.reshape(1,DIM),len(rows))
                similarities = np.empty(len(rows),dtype=np.float32)
                similarities[indices[0]] = scores[0]
                selected = 'faiss'
        if selected == 'numpy':
            similarities = matrix @ q
        order = np.argsort(-similarities,kind='stable')[:limit]
        return {'backend':selected,'items':[{'id':rows[i][0],'kind':rows[i][1],'tick':rows[i][2],
            'payload':json.loads(rows[i][3]),'similarity':float(similarities[i])} for i in order]}
