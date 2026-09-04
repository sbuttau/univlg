import json
import numpy as np
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchmetrics_ext.metrics.visual_grounding import ViGiL3DMetric
from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument('--test_results', default='tests/vigil3d/vigil3d_ref_scannet_all_batched_test_results_scannet_converted_aligned.json', help='path to converted test results JSON')
parser.add_argument('--scannet_csv', default='data/vigil3d/vigil3d_scannet.csv')
parser.add_argument('--scannetpp_csv', default=None, help='path to scannetpp csv (optional)')
args = parser.parse_args()

print("Loading data...")

# # Load from HuggingFace
# ds = load_dataset("3dlg-hcvc/vigil3d", split="validation")
# flags = {f"{r['scene_id']}_{r['ann_id']}": r for r in ds}

# Load from CSV
ds_scannet = pd.read_csv(args.scannet_csv)
# dopo aver caricato il CSV
rename_map = {
    'num_attribute_type_number': 'attribute_type_number',
    'num_attribute_type_color': 'attribute_type_color',
    'num_attribute_type_state': 'attribute_type_state',
    'num_attribute_type_text_label': 'attribute_type_text_label',
    'num_relationship_type_far': 'relationship_type_far',
    'num_relationship_type_arrangement': 'relationship_type_arrangement',
    'num_relationship_type_ordinal': 'relationship_type_ordinal',
    'num_relationship_type_comparison': 'relationship_type_comparison',
    'num_anchor_type_single': 'anchor_type_single',
    'num_anchor_type_multiple': 'anchor_type_multiple',
    'num_anchor_type_non_object': 'anchor_type_non_object',
    'num_anchor_type_viewpoint': 'anchor_type_viewpoint',
    'num_negation': 'negation',
    'num_target_not_first_np': 'target_not_first_np',
    'num_coreferences': 'coreferences',
}
if args.scannetpp_csv is not None:
    ds_scannetpp = pd.read_csv(args.scannetpp_csv)
    ds = pd.concat([ds_scannet, ds_scannetpp], ignore_index=True)
else:
    ds = ds_scannet.rename(columns=rename_map)
    # ds = ds_scannet
flags = {f"{r['scene_id']}_{r['prompt_id']}": r.to_dict() for _, r in ds.iterrows()}

path = hf_hub_download(repo_id="torchmetrics-ext/metadata",
                       filename="scannetv2/obj_aabbs_validation.npz",
                       repo_type="dataset")
meta = np.load(path)

preds_aligned = json.load(open(args.test_results))
preds_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in preds_aligned.items()}
# ── conteggio predizioni ──────────────────────────────────────────────────────
n_preds   = len(preds_aligned)
n_in_csv  = sum(1 for k in preds_aligned if k in flags)
n_out_csv = n_preds - n_in_csv
print(f"Predictions in test_results: {n_preds}  "
      f"(matched in CSV: {n_in_csv}, total in csv: {len(flags)}, missing from eval: {len(flags) - n_in_csv})")
base_metric = ViGiL3DMetric(split="validation", strict=False)
gt_data = base_metric.gt_data

# ------------------------------------------------------------------ helpers

def box_iou_3d(box1, box2):
    b1_min, b1_max = np.array(box1[0]), np.array(box1[1])
    b2_min, b2_max = np.array(box2[0]), np.array(box2[1])
    inter_vol = np.maximum(0, np.minimum(b1_max, b2_max) - np.maximum(b1_min, b2_min)).prod()
    vol1 = (b1_max - b1_min).prod()
    vol2 = (b2_max - b2_min).prod()
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

# precomputa iou con GT box
iou_results = {}
for key, pred_box in preds_aligned.items():
    if key not in flags:
        continue
    row = flags[key]
    object_id = row.get('object_id')
    if pd.isna(object_id):
        continue
    # prendi il primo id in caso di multi-target
    object_id = str(object_id).split(',')[0].strip()
    gt_key = f"{row['scene_id']}_{object_id}"
    if gt_key not in meta:
        continue
    iou_results[key] = {"iou": box_iou_3d(pred_box[0], meta[gt_key].tolist()), "flags": row}

def acc_gt(condition=None, threshold=0.25):
    keys = [k for k in iou_results if condition is None or condition(iou_results[k]["flags"])]
    if not keys:
        return None
    return sum(1 for k in keys if iou_results[k]["iou"] >= threshold) / len(keys) * 100

def f1_pred(condition=None, threshold=0.25):
    subset = {k: v for k, v in preds_torch.items()
              if k in flags and (condition is None or condition(flags[k]))}
    if not subset:
        return None
    m = ViGiL3DMetric.__new__(ViGiL3DMetric)
    ViGiL3DMetric.__init__(m, split="validation", strict=False)
    m.gt_data = gt_data
    m.update(subset)
    r = m.compute()
    key = f"st_{threshold}"
    return r.get(key, torch.tensor(float("nan"))).item() * 100

# baselines dal paper (Table 6)
baselines = {
    "OpenScene":    (2.1,  1.7,  1.3,  2.1,  1.7,  1.2),
    "LERF":         (2.5,  2.1,  2.1,  2.5,  2.1,  2.1),
    "ZSVG3D":       (18.9, 8.5,  5.6,  12.2, 6.7,  5.8),
    "LLM-Grounder": (2.5,  7.1,  5.0,  2.5,  5.3,  3.1),
    "3D-VisTA":     (14.2, 15.8, 13.3, 14.1, 15.7, 13.2),
    "3D-GRAND":     (17.9, 15.8, 12.5, 17.9, 15.3, 11.8),
    "PQ3D":         (26.2, 10.8, 10.8, 26.8, 5.6,  5.1),
}

# ------------------------------------------------------------------ Table 6

t6 = PrettyTable()
t6.field_names = ["Model", "Acc/GT", "Acc@25", "Acc@50", "F1/GT", "F1@25", "F1@50"]
t6.align = "r"
t6.align["Model"] = "l"

for model, vals in baselines.items():
    t6.add_row([model] + [f"{v:.1f}" for v in vals])

t6.add_divider()

univlg_acc_gt_25 = acc_gt(threshold=0.25)
univlg_acc_gt_50 = acc_gt(threshold=0.50)
univlg_f1_25    = f1_pred(threshold=0.25)
univlg_f1_50    = f1_pred(threshold=0.50)

t6.add_row([
    "UniVLG",
    f"{univlg_acc_gt_25:.1f}",
    f"{univlg_acc_gt_25:.1f}",
    f"{univlg_acc_gt_50:.1f}",
    f"{univlg_acc_gt_25:.1f}",
    f"{univlg_f1_25:.1f}" if univlg_f1_25 is not None else "-",
    f"{univlg_f1_50:.1f}" if univlg_f1_50 is not None else "-",
])

print("\n=== Table 6: Accuracy and F1 score (%) on ViGiL3D — ScanNet ===")
print(t6)

# ------------------------------------------------------------------ Table 7

# subgroups = {
#     "Overall":  None,
#     "Num":      lambda r: r["attribute_type_number"],
#     "Lab":      lambda r: r["attribute_type_text_label"],
#     "State":    lambda r: r["attribute_type_state"],
#     "Far":      lambda r: r["relationship_type_far"],
#     "Arr":      lambda r: r["relationship_type_arrangement"],
#     "Ord":      lambda r: r["relationship_type_ordinal"],
#     "Comp":     lambda r: r["relationship_type_comparison"],
#     "Gen":      lambda r: r["granularity"] == "generic",
#     "CG":       lambda r: r["granularity"] == "coarse-grained",
#     "FG":       lambda r: r["granularity"] == "fine-grained",
#     "NFN":      lambda r: r["target_not_first_np"],
#     "Sing":     lambda r: r["anchor_type_single"],
#     "Mul":      lambda r: r["anchor_type_multiple"],
#     "Non":      lambda r: r["anchor_type_non_object"],
#     "Agt":      lambda r: r["anchor_type_viewpoint"],
#     "Neg":      lambda r: r["negation"],
# }
subgroups = {
    "Overall":  None,
    "Num":      lambda r: r["attribute_type_number"] > 0,
    "Lab":      lambda r: r["attribute_type_text_label"] > 0,
    "State":    lambda r: r["attribute_type_state"] > 0,
    "Far":      lambda r: r["relationship_type_far"] > 0,
    "Arr":      lambda r: r["relationship_type_arrangement"] > 0,
    "Ord":      lambda r: r["relationship_type_ordinal"] > 0,
    "Comp":     lambda r: r["relationship_type_comparison"] > 0,
    "Gen":      lambda r: r["granularity"] == "generic",
    "CG":       lambda r: r["granularity"] == "categorical",
    "FG":       lambda r: r["granularity"] == "fine-grained",
    "NFN":      lambda r: r["target_not_first_np"] > 0,
    "Sing":     lambda r: r["anchor_type_single"] > 0,
    "Mul":      lambda r: r["anchor_type_multiple"] > 0,
    "Non":      lambda r: r["anchor_type_non_object"] > 0,
    "Agt":      lambda r: r["anchor_type_viewpoint"] > 0,
    "Neg":      lambda r: r["negation"] > 0,
}
cols = list(subgroups.keys())

# baselines Table 7
baselines_t7 = {
    "OpenScene":    [2.1,  4.4,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.5,  1.9,  2.0,  0.0,  3.8,  1.1,  1.6,  0.0,  8.1],
    "LERF":         [2.5,  0.0,  4.0,  4.0,  3.3,  2.9,  3.7,  6.1,  2.5,  1.9,  2.7,  2.9,  0.8,  4.4,  6.6,  3.7,  0.0],
    "ZSVG3D":       [18.9, 20.5, 12.0, 28.0, 13.3, 8.8,  19.2, 25.0, 15.8, 13.2, 21.8, 19.4, 19.4, 15.7, 14.8, 23.1, 10.8],
    "LLM-Grounder": [2.5,  2.2,  0.0,  0.0,  3.3,  5.7,  11.1, 6.1,  0.0,  0.0,  4.1,  7.2,  1.5,  5.5,  4.9,  11.1, 2.7],
    "3D-VisTA":     [14.2, 6.7,  0.0,  8.0,  10.0, 5.7,  7.4,  8.2,  0.0,  13.2, 18.4, 8.8,  13.7, 12.1, 15.0, 19.2, 8.1],
    "3D-GRAND":     [17.9, 13.3, 4.0,  12.0, 13.3, 8.6,  14.8, 18.4, 7.5,  13.2, 22.4, 17.4, 18.3, 15.4, 19.7, 18.5, 21.6],
    "PQ3D":         [26.2, 28.9, 8.0,  28.0, 26.7, 22.9, 7.4,  24.5, 20.0, 24.5, 28.6, 26.1, 23.7, 22.0, 24.6, 18.5, 13.5],
}

t7 = PrettyTable()
t7.field_names = ["Model"] + cols
t7.align = "r"
t7.align["Model"] = "l"

for model, vals in baselines_t7.items():
    t7.add_row([model] + [f"{v:.1f}" for v in vals])

t7.add_divider()

univlg_row = ["UniVLG"]
for name, condition in subgroups.items():
    val = acc_gt(condition=condition, threshold=0.25)
    univlg_row.append(f"{val:.1f}" if val is not None else "-")
t7.add_row(univlg_row)

print("\n=== Table 7: Subgroup Analysis — Acc/GT (%) ===")
print(t7)