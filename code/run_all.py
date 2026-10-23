#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

trans = load('transformation_suite', HERE / 'transformation_and_quotient_suite.py')
recon = load('reconstructibility_suite', HERE / 'final_reconstructibility_suite.py')

print('Running transformation/quotient suite...')
trans.main(RESULTS)
print('Running reconstructibility suite...')
recon.main(RESULTS)

# Reproducibility manifest
manifest = []
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name != 'SHA256SUMS.txt':
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest.append(f'{h}  {p.relative_to(ROOT)}')
(ROOT / 'SHA256SUMS.txt').write_text('\n'.join(manifest), encoding='utf-8')
print('Done. Results:', RESULTS)
