#!/usr/bin/env python3
"""Generate Gazebo worlds and ArUco marker models from config/layouts.yaml.

Run from anywhere:  python3 fetch_deliver_gazebo/scripts/generate_worlds.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from fetch_deliver_gazebo.layouts import load_layouts  # noqa: E402
from fetch_deliver_gazebo.sdf_gen import iter_marker_ids, marker_model_files, world_sdf  # noqa: E402


def write_marker_texture(path: str, marker_id: int, pixels: int = 400) -> None:
    import cv2
    import numpy as np
    if hasattr(cv2.aruco, 'generateImageMarker'):        # OpenCV >= 4.7
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    else:                                                # OpenCV 4.5 (Ubuntu 22.04)
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.drawMarker(dictionary, marker_id, pixels)
    border = pixels // 6  # white quiet zone around the marker
    img = cv2.copyMakeBorder(img, border, border, border, border,
                             cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, np.ascontiguousarray(img))


def main() -> None:
    layouts = load_layouts(os.path.join(PKG, 'config', 'layouts.yaml'))
    os.makedirs(os.path.join(PKG, 'worlds'), exist_ok=True)
    for name, layout in layouts.items():
        out = os.path.join(PKG, 'worlds', f'{name}.world')
        with open(out, 'w') as f:
            f.write(world_sdf(layout))
        print('wrote', out)
    for marker_id in iter_marker_ids(layouts.values()):
        mdir = os.path.join(PKG, 'models', f'aruco_marker_{marker_id}')
        os.makedirs(os.path.join(mdir, 'materials', 'scripts'), exist_ok=True)
        os.makedirs(os.path.join(mdir, 'materials', 'textures'), exist_ok=True)
        config, sdf, material = marker_model_files(marker_id)
        for rel, content in (('model.config', config), ('model.sdf', sdf),
                             (f'materials/scripts/aruco_{marker_id}.material', material)):
            with open(os.path.join(mdir, rel), 'w') as f:
                f.write(content)
        write_marker_texture(os.path.join(mdir, 'materials', 'textures', f'aruco_{marker_id}.png'),
                             marker_id)
        print('wrote', mdir)


if __name__ == '__main__':
    main()
