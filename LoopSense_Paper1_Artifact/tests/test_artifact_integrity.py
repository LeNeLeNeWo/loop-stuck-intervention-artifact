from __future__ import annotations
import csv,hashlib,re,subprocess,sys,py_compile,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; POOLS=['Natural500','Final500','StressFresh100','OpenHandsExternal330']
LOCAL_PATTERNS=[re.compile(r'(?<![A-Za-z0-9_])D:[\\/][^\s,;]*',re.I),re.compile(r'\\\\'+'wsl'+r'\.localhost',re.I),re.compile(r'/home/[A-Za-z0-9_.-]+')]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def main():
    for x in ['README.md','LICENSE','CITATION.cff','manifest_sha256.csv','data/normalized/normalized_steps.csv','data/normalized/normalized_spans.csv']:
        assert (ROOT/x).exists(),f'missing {x}'
    for pool in POOLS:
        b=ROOT/'data/final_adjudicated'/pool
        assert (b/'step_labels.csv').exists() and (b/'span_labels.csv').exists(),pool
        assert len(pd.read_csv(b/'step_labels.csv'))>0
    for p in ROOT.rglob('*'):
        assert '__pycache__' not in p.parts and '.git' not in p.parts and '.venv' not in p.parts, f'forbidden cache/env {p}'
        assert p.suffix.lower()!='.zip', f'nested zip {p}'
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.md','.csv','.json','.py','.yml','.yaml','.cff','.txt'} and p.stat().st_size<80_000_000:
            t=p.read_text(encoding='utf-8',errors='ignore')
            for pat in LOCAL_PATTERNS: assert not pat.search(t), f'local path in {p}'
            assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',t),f'email-like string in {p}'
            assert not re.search(r'sk-[A-Za-z0-9_-]{20,}',t),f'secret-like token in {p}'
    with (ROOT/'manifest_sha256.csv').open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            p=ROOT/r['path']; assert p.exists(),f'manifest missing {r["path"]}'; assert sha(p)==r['sha256'],f'hash mismatch {r["path"]}'
    for s in (ROOT/'scripts').rglob('*.py'):
            with tempfile.TemporaryDirectory() as td:
                py_compile.compile(str(s), cfile=str(Path(td)/'out.pyc'), doraise=True)
    print('artifact integrity test passed')
if __name__=='__main__': main()
