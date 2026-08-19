import json
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='path to test_results.json')
parser.add_argument('--align_matrices', default='/workspaces/univlg/univlg/data_video/scans_axis_alignment_matrices.json')
parser.add_argument('--align', action='store_true', help='apply alignment matrices to predictions')
args = parser.parse_args()

input_path = Path(args.input)
output_path = input_path.with_stem(input_path.stem + '_converted')
if args.align:
    output_path = input_path.with_stem(input_path.stem + '_converted_aligned')
    align_matrices = json.load(open(args.align_matrices))
data = json.load(open(input_path))

preds = {}
for r in data:
    scene_id = r['scene_id']
    key = f"{scene_id}_{r['ann_id']}"
    corners = np.array(r['bbox'])  # (8, 3)
    if args.align and scene_id in align_matrices:
        M = np.array(align_matrices[scene_id]).reshape(4, 4)
        corners_h = np.hstack([corners, np.ones((8, 1))])
        corners = (M @ corners_h.T).T[:, :3]

    preds[key] = [[[
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 2].min())
    ], [
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
        float(corners[:, 2].max())
    ]]]

json.dump(preds, open(output_path, 'w'), indent=2)
print(f"converted {len(preds)} predictions → {output_path}")