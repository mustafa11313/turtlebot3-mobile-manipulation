"""Run a batch of randomized fetch & deliver trials and aggregate the metrics.

Each trial starts a fresh simulation (mission.launch.py, headless) together
with the metrics recorder, waits for the mission to finish or time out, then
tears everything down. Results go to <out>/trials/*.json, <out>/summary.csv
and <out>/summary.md.

  ros2 run fetch_deliver_eval run_benchmark --layouts apartment office \
      --difficulties easy distractors occluded dynamic hard --seeds 0-4 --out results
"""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time

from .report import summarize


def parse_seeds(spec: str):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def kill_tree(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    # stray simulator processes would break the next trial
    subprocess.run(['pkill', '-9', '-x', 'gzserver'], check=False)
    subprocess.run(['pkill', '-9', '-x', 'gzclient'], check=False)


def run_trial(layout, difficulty, seed, target, out_dir, timeout, extra_args, gui):
    name = f'{layout}_{difficulty}_s{seed}'
    trial_file = os.path.join(out_dir, 'trials', f'{name}.json')
    log_file = open(os.path.join(out_dir, 'logs', f'{name}.log'), 'w')
    scenario_file = os.path.join(out_dir, 'trials', f'{name}.scenario.json')
    launch = subprocess.Popen(
        ['ros2', 'launch', 'fetch_deliver_bringup', 'mission.launch.py', f'layout:={layout}',
         f'difficulty:={difficulty}', f'seed:={seed}', f'target:={target}', 'rviz:=false',
         f'gui:={str(gui).lower()}', f'scenario_output:={scenario_file}'] + extra_args,
        stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    recorder = subprocess.Popen(
        ['ros2', 'run', 'fetch_deliver_eval', 'metrics_recorder', '--ros-args',
         '-p', 'use_sim_time:=true', '-p', f'layout:={layout}', '-p', f'output_file:={trial_file}',
         '-p', f'trial_name:={name}'],
        stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    t0 = time.time()
    status = 'timeout'
    while time.time() - t0 < timeout:
        if os.path.exists(trial_file):
            status = 'done'
            break
        if launch.poll() is not None:
            status = 'crashed'
            break
        time.sleep(2.0)
    kill_tree(recorder)
    kill_tree(launch)
    log_file.close()
    if status != 'done':
        result = {'trial': name, 'layout': layout, 'mission_success': False, 'task_success': False,
                  'found': False, 'picked_target': False, 'delivered_target': False,
                  'failure_reason': status, 'scenario': {'difficulty': difficulty, 'seed': seed}}
        if os.path.exists(scenario_file):
            with open(scenario_file) as f:
                result['scenario'] = json.load(f)
        with open(trial_file, 'w') as f:
            json.dump(result, f, indent=2)
    with open(trial_file) as f:
        res = json.load(f)
    print(f'[{name}] {status}: task_success={res.get("task_success")} '
          f'time={res.get("time_to_complete")} reason={res.get("failure_reason")}', flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layouts', nargs='+', default=['arena'])
    ap.add_argument('--difficulties', nargs='+', default=['easy'])
    ap.add_argument('--seeds', default='0-2')
    ap.add_argument('--target', default='red_cylinder')
    ap.add_argument('--out', default='results')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-trial wall-clock timeout [s]')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--launch-arg', action='append', default=[],
                    help='extra mission.launch.py argument, e.g. --launch-arg slam:=true')
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.out, 'trials'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    for layout, diff, seed in itertools.product(args.layouts, args.difficulties, parse_seeds(args.seeds)):
        name = f'{layout}_{diff}_s{seed}'
        if args.skip_existing and os.path.exists(os.path.join(args.out, 'trials', f'{name}.json')):
            continue
        run_trial(layout, diff, seed, args.target, args.out, args.timeout,
                  args.launch_arg + ['shutdown_when_done:=false'], args.gui)
    md = summarize(args.out)
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
