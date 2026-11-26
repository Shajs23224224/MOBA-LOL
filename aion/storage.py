import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime,timezone

class Store:
    def __init__(self, path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.executescript('''CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,started TEXT NOT NULL,ended TEXT,status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(id), topic TEXT NOT NULL,body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_session ON events(session_id);''')
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10)
        try:
            c.execute('PRAGMA foreign_keys=ON')
            with c: yield c
        finally: c.close()
    def start(self,id):
        with self.connect() as c: c.execute('INSERT INTO sessions VALUES(?,?,NULL,?)',(id,datetime.now(timezone.utc).isoformat(),'running'))
    def finish(self,id,status):
        with self.connect() as c: c.execute('UPDATE sessions SET ended=?,status=? WHERE id=? AND status=?',(datetime.now(timezone.utc).isoformat(),status,id,'running'))
    def record(self,session,event):
        with self.connect() as c: c.execute('INSERT INTO events VALUES(?,?,?,?)',(event.id,session,event.topic,event.model_dump_json()))
    def summary(self):
        with self.connect() as c:
            return {'sessions': [dict(zip(('id','status'),row)) for row in c.execute('SELECT id,status FROM sessions ORDER BY started')], 'events':c.execute('SELECT count(*) FROM events').fetchone()[0]}
