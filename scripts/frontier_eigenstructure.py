#!/usr/bin/env python3
"""Compute frontier coupling matrix eigenstructure + per-lab coupling slopes."""
import json, numpy as np, os
DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, '..', 'data')
f = json.load(open(os.path.join(DATA, 'frontier_34models.json')))
models = f['models']
swe = np.array([m['swe'] for m in models])
gpqa = np.array([m['gpqa'] for m in models])
labs = [m['lab'] for m in models]
C = np.array([[1.000, 0.848, -0.251], [0.848, 1.000, 0.715], [-0.251, 0.715, 1.000]])
eigenvalues, eigenvectors = np.linalg.eigh(C)
print("Eigenvalues:", sorted(eigenvalues, reverse=True))
for lab in set(labs):
    i = [j for j, l in enumerate(labs) if l == lab]
    if len(i) >= 4:
        s = np.polyfit(swe[i], gpqa[i], 1)[0]
        r = np.corrcoef(swe[i], gpqa[i])[0,1]
        print(f"{lab}: slope={s:.3f}, r={r:+.3f}")
