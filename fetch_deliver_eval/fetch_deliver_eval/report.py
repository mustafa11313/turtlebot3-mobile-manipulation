"""Aggregate trial JSON files into summary.csv / summary.md (and plots if matplotlib exists)."""

import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

COLUMNS = ['trial', 'layout', 'difficulty', 'seed', 'task_success', 'found', 'picked_target',
           'delivered_target', 'picked_wrong_object', 'time_to_complete', 'time_to_detect',
           'pick_attempts', 'attach_attempts', 'attach_passed', 'grasp_error_m', 'placement_error_m',
           'detection_precision', 'detection_recall', 'collisions', 'recoveries', 'failure_reason']


def _load(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, 'trials', '*.json'))):
        if path.endswith('.scenario.json'):
            continue
        with open(path) as f:
            r = json.load(f)
        sc = r.get('scenario') or {}
        r['difficulty'] = sc.get('difficulty')
        r['seed'] = sc.get('seed')
        rows.append(r)
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _fmt(v, pct=False, unit=''):
    if v is None:
        return '–'
    if pct:
        return f'{100 * v:.0f}%'
    if isinstance(v, float):
        return f'{v:.3f}{unit}' if abs(v) < 1 else f'{v:.1f}{unit}'
    return f'{v}{unit}'


def aggregate(rows):
    def agg(group):
        n = len(group)
        att = sum(r.get('attach_attempts') or 0 for r in group)
        ok = sum(r.get('attach_passed') or 0 for r in group)
        tp = sum((r.get('detection_counts') or {}).get('tp', 0) for r in group)
        fp = sum((r.get('detection_counts') or {}).get('fp', 0) for r in group)
        fn = sum((r.get('detection_counts') or {}).get('fn', 0) for r in group)
        succ = [r for r in group if r.get('task_success')]
        return {
            'trials': n,
            'success_rate': sum(bool(r.get('task_success')) for r in group) / n if n else None,
            'found_rate': sum(bool(r.get('found')) for r in group) / n if n else None,
            'picked_rate': sum(bool(r.get('picked_target')) for r in group) / n if n else None,
            'delivered_rate': sum(bool(r.get('delivered_target')) for r in group) / n if n else None,
            'mean_time_success_s': _mean([r.get('time_to_complete') for r in succ]),
            'attach_pass_rate': ok / att if att else None,
            'mean_grasp_error_m': _mean([r.get('grasp_error_m') for r in group]),
            'mean_placement_error_m': _mean([r.get('placement_error_m') for r in group]),
            'detection_precision': tp / (tp + fp) if tp + fp else None,
            'detection_recall': tp / (tp + fn) if tp + fn else None,
            'collisions_per_trial': _mean([r.get('collisions') for r in group]),
            'wrong_object_picks': sum(bool(r.get('picked_wrong_object')) for r in group),
        }
    by = defaultdict(list)
    for r in rows:
        by[(r.get('layout'), r.get('difficulty'))].append(r)
    groups = {k: agg(v) for k, v in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))}
    return agg(rows), groups


def summarize(out_dir):
    rows = _load(out_dir)
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    overall, groups = aggregate(rows)
    lines = ['| Layout | Difficulty | Trials | Success | Found | Picked | Delivered | Mean time (s) | '
             'Attach pass | Grasp err (mm) | Place err (cm) | Det. P / R | Collisions/trial |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']

    def row(name_a, name_b, g):
        ge = g['mean_grasp_error_m']
        pe = g['mean_placement_error_m']
        return (f"| {name_a} | {name_b} | {g['trials']} | {_fmt(g['success_rate'], True)} | "
                f"{_fmt(g['found_rate'], True)} | {_fmt(g['picked_rate'], True)} | "
                f"{_fmt(g['delivered_rate'], True)} | {_fmt(g['mean_time_success_s'])} | "
                f"{_fmt(g['attach_pass_rate'], True)} | {'–' if ge is None else f'{1000 * ge:.1f}'} | "
                f"{'–' if pe is None else f'{100 * pe:.1f}'} | "
                f"{_fmt(g['detection_precision'], True)} / {_fmt(g['detection_recall'], True)} | "
                f"{_fmt(g['collisions_per_trial'])} |")
    for (layout, diff), g in groups.items():
        lines.append(row(layout, diff, g))
    lines.append(row('**all**', '', overall))
    md = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(md)
    _plots(out_dir, rows, groups)
    return md


def _plots(out_dir, rows, groups):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not groups:
        return
    labels = [f'{a}\n{b}' for a, b in groups]
    rates = [100 * (g['success_rate'] or 0) for g in groups.values()]
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels)), 3.2))
    ax.bar(labels, rates, color='#3b6ea5')
    ax.set_ylabel('Task success [%]')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'success_rate.png'), dpi=150)
    plt.close(fig)
    ge = [1000 * r['grasp_error_m'] for r in rows if r.get('grasp_error_m') is not None]
    if ge:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.hist(ge, bins=12, color='#3b6ea5')
        ax.set_xlabel('Gripper-to-object error at attach [mm]')
        ax.set_ylabel('Trials')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'grasp_error.png'), dpi=150)
        plt.close(fig)


def main():
    print(summarize(sys.argv[1] if len(sys.argv) > 1 else 'results'))


if __name__ == '__main__':
    main()
