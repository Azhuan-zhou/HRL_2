"""Reproduce the training plot from the adjacent metrics.jsonl."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
records = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
steps = np.array([r['step'] for r in records])
loss = np.array([r['train_loss'] for r in records])
val = [r for r in records if 'val_loss' in r]
window = 100
smooth = np.convolve(loss, np.ones(window) / window, mode='valid')
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
ax.plot(steps, loss, color='#9cbfdd', alpha=.45, linewidth=.6, label='Training loss (per step)')
ax.plot(steps[window-1:], smooth, color='#1767a5', linewidth=2.2, label='Training loss (100-step moving average)')
ax.plot([r['step'] for r in val], [r['val_loss'] for r in val], 'o--', color='#d65a31', linewidth=2, markersize=7, label='Validation loss')
for r in val:
    ax.annotate(f"{r['val_loss']:.4f}", (r['step'], r['val_loss']), xytext=(-12, -24), textcoords='offset points', color='#a73b1b', fontweight='bold')
ax.set(yscale='log', xlabel='Training step', ylabel='Diffusion loss (log scale)', xlim=(0, 2070), title='Cube placement — Diffusion policy training')
ax.grid(True, which='major', alpha=.18)
ax.legend(loc='upper right', frameon=False)
fig.supxlabel('2,000 steps · batch size 8 · 31 training / 3 validation episodes\nValidation uses 64 sampled frames; loss does not measure robot success rate.', fontsize=9, color='#555555')
fig.savefig(root / 'training_curve.png', dpi=180)
fig.savefig(root / 'training_curve.pdf')
plt.close(fig)
