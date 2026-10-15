#!/usr/bin/env python3
"""Score a saved SLAM map against the layout's ground truth, or export the ground-truth map.

  map_quality.py --layout apartment --map maps/apartment.yaml      # score
  map_quality.py --layout apartment --export /tmp/apartment_gt      # write pgm/yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_deliver_gazebo.gt_map import occupied_iou, rasterize, save_map  # noqa: E402
from fetch_deliver_gazebo.layouts import load_layout  # noqa: E402


def load_pgm_map(yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    pgm = os.path.join(os.path.dirname(yaml_path), meta['image'])
    with open(pgm, 'rb') as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        img = np.frombuffer(f.read(), np.uint8 if maxval < 256 else np.uint16).reshape(h, w)
    img = img[::-1]
    occ = img.astype(float) / maxval
    occ = (1.0 - occ) if not meta.get('negate', 0) else occ
    grid = np.full(img.shape, -1, np.int8)
    grid[occ > meta['occupied_thresh']] = 100
    grid[occ < meta['free_thresh']] = 0
    if meta.get('mode', 'trinary') == 'trinary':
        grid[img == 205] = -1      # map_saver writes unknown cells as 205
    return grid, meta['origin'][:2], meta['resolution']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', required=True)
    ap.add_argument('--map')
    ap.add_argument('--export')
    ap.add_argument('--layouts-file')
    args = ap.parse_args()
    layout = load_layout(args.layout, args.layouts_file)
    gt, gt_origin = rasterize(layout)
    if args.export:
        save_map(gt, gt_origin, 0.05, args.export)
        print(f'wrote {args.export}.yaml')
    if args.map:
        grid, origin, res = load_pgm_map(args.map)
        score = occupied_iou(grid, origin, res, gt, gt_origin, 0.05)
        known = float(np.mean(grid >= 0))
        print(json.dumps({'layout': args.layout, **score, 'known_fraction': known}, indent=2))


if __name__ == '__main__':
    main()
