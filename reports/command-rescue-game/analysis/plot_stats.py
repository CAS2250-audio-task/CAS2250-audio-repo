"""Export report figures from run-stats.mjs outputs; no model/data access."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    folder = Path(folder)
    data = json.loads((folder / 'summary.json').read_text())
    rows = data['summary']
    config = data['manifest']['base_config']
    base = [r for r in rows if r['scenario'] == 'base']
    names = {'original': 'Original', 'car': 'Car', 'restaurant': 'Restaurant', 'compressor': 'Compressor'}
    colors = {'original': '#53646e', 'car': '#247b92', 'restaurant': '#c56c35', 'compressor': '#7660a4'}
    label = lambda r: names[r['noise']] + ('' if r['snr'] is None else f" {r['snr']} dB")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'figure.facecolor': 'white'})

    def save(fig, name):
        fig.savefig(folder / (name + '.png'), dpi=220, bbox_inches='tight')
        fig.savefig(folder / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.7), layout='constrained')
    for i, r in enumerate(base):
        x = r['survival_rate'] * 100
        ax.errorbar(x, i, xerr=[[max(0, x - r['survival_mc95_low'] * 100)],
                               [max(0, r['survival_mc95_high'] * 100 - x)]],
                    fmt='o', color=colors[r['noise']], capsize=3)
        ax.plot(r['analytic_survival'] * 100, i, '|', color='black', markersize=11)
    ax.set(yticks=range(len(base)), yticklabels=[label(r) for r in base], xlim=(-1, 101),
           xlabel='Mission survival (%)', title='COMMAND RESCUE / safe return, by acoustic condition')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    ax.plot([], [], 'o', color='#53646e', label='Simulation + 95% conditional Monte Carlo interval')
    ax.plot([], [], '|', color='black', label='Analytic cross-check')
    ax.legend(loc='lower right', fontsize=8)
    save(fig, 'survival')

    sweep = [r for r in rows if r['scale'] == config['scale']]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True, layout='constrained')
    for ax, noise in zip(axes, ['car', 'restaurant', 'compressor']):
        for snr, color in zip([10, 0, -5], ['#247b92', '#c56c35', '#7660a4']):
            group = sorted([r for r in sweep if r['noise'] == noise and r['snr'] == snr], key=lambda r: r['deadline'])
            ax.plot([r['deadline']/1000 for r in group], [r['survival_rate']*100 for r in group],
                    '-o', color=color, markersize=3, label=f'{snr} dB')
        original = sorted([r for r in sweep if r['noise'] == 'original'], key=lambda r: r['deadline'])
        ax.plot([r['deadline']/1000 for r in original], [r['survival_rate']*100 for r in original],
                '--', color='#53646e', label='Original')
        ax.set(title=names[noise], xlabel='Deadline (s)', ylim=(-2, 102)); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mission survival (%)'); axes[-1].legend(fontsize=8)
    fig.suptitle(f'Deadline sensitivity / compute latency x{config["scale"]}')
    save(fig, 'deadline-sweep')

    fig, ax = plt.subplots(figsize=(10.5, 4.6), layout='constrained')
    x = np.arange(len(base))
    for offset, key, title, color in [(-.24, 'recognition_errors', 'Recognition error', '#c56c35'),
                                     (0, 'deadline_misses', 'Deadline missed', '#247b92'),
                                     (.24, 'recognition_and_time_errors', 'Both errors', '#7660a4')]:
        ax.bar(x+offset, [100*r[key]/r['transmissions'] for r in base], .23, label=title, color=color)
    ax.set(xticks=x, xticklabels=[label(r) for r in base], ylabel='Observed transmitted rounds (%)',
           title='Failure flags / recognition and deadline flags can overlap')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    save(fig, 'failure-flags')

    fig, ax = plt.subplots(figsize=(8.5, 5.7), layout='constrained')
    for i, r in enumerate(base):
        native = next((s for s in rows if s['noise'] == r['noise'] and s['snr'] == r['snr'] and s['scenario'] == 'unscaled_compute'), r)
        ax.plot([r['survival_rate']*100, native['survival_rate']*100], [i, i], '-', color='#b4c2c7')
        ax.plot(r['survival_rate']*100, i, 'o', color='#c56c35')
        ax.plot(native['survival_rate']*100, i, 's', color='#247b92')
    ax.plot([], [], 'o', color='#c56c35', label=f'Demo compute latency x{config["scale"]}')
    ax.plot([], [], 's', color='#247b92', label='Measured compute latency x1')
    ax.set(yticks=range(len(base)), yticklabels=[label(r) for r in base], xlim=(-1, 101),
           xlabel='Mission survival (%)', title='Same recordings, different compute latency scale')
    ax.invert_yaxis(); ax.legend(loc='lower right', fontsize=8); ax.grid(axis='x', alpha=.2)
    save(fig, 'compute-scale')
    print('Saved four figures as PNG (220dpi) and vector PDF.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python analysis/plot_stats.py OUTPUT_DIRECTORY')
    main(sys.argv[1])
