"""Plot in the style of Sun et al. (Massive Activations in LLMs, Fig. 4 /
Table 1) for the language encoder: Top1/Top2/Top3 activation magnitudes
plus the median, per layer.

Data sources (same conventions as plot_massive_3d.py):
- topk_abs/<module_name>: entries = dict with top_vals (..., seq, k),
  top_dims (..., seq, k), median_abs (..., seq). Covers ALL scenes.
  The GLOBAL Top1/2/3 for the layer is reconstructed by pooling the
  top-k of all real tokens and re-sorting (exact as long as the global
  top-3 lies within the top-k of *some* token -- virtually guaranteed
  with k=10 if the sink is concentrated in 1-2 tokens). The median is
  approximated as the median of per-token medians (median_abs), to be
  validated against the --raw_data check.
- raw/<module_name> (max 3 scenes): raw pre-norm tensor, used ONLY as
  an exact sanity check on Top1/2/3 and the median.

Padding tokens are excluded using the same mask convention as
plot_massive_3d.py (masks[scene], True = padding).

Usage:
    python plot_massive_per_layer.py --topk_data topk_abs.pt --masks attention_masks.pt \
        --raw_data raw.pt --which norm2 --out_dir ./plots
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Same helpers as plot_massive_3d.py, for consistency across scripts
# ---------------------------------------------------------------------
def layer_sort_key(name):
    m = re.search(r"layers\.(\d+)\.(norm\d)", name)
    return (int(m.group(1)), m.group(2)) if m else (999, name)


def parse_layer_norm(name):
    m = re.search(r"layers\.(\d+)\.(norm\d)", name)
    if m is None:
        return None, None
    return int(m.group(1)), m.group(2)


def short_name(name):
    layer, norm = parse_layer_norm(name)
    return f"L{layer} \u00b7 {norm}" if layer is not None else name


def split_scenes(t, non_batch_ndim):
    """
    Split a tensor into a list of per-scene slices along dim 0.

    Handles two cases transparently:
    - one hook call per scene: t has exactly `non_batch_ndim` dims
      (e.g. (seq, k)) -> returned as a single-element list [t].
    - one hook call for a whole batch of scenes ("batched" runs): t has
      `non_batch_ndim + 1` dims (e.g. (B, seq, k)) -> unbind dim 0 into
      B per-scene slices.

    This matters because some saved files store ALL scenes in a single
    batched forward call rather than one call per scene; naively taking
    index 0 (as the old get_seq_hidden/get_seq_only helpers did) silently
    drops every scene but the first, which shows up as std=0 across all
    "scenes" in the aggregated stats.
    """
    if t.dim() == non_batch_ndim + 1:
        return list(t.unbind(dim=0))
    return [t]


def real_length(masks, scene_idx):
    mask = masks[scene_idx]
    L = (~mask).sum(dim=-1)
    return int(L[0] if L.dim() > 0 else L)


# ---------------------------------------------------------------------
# Top1/2/3 + median from topk_abs (approximate, all scenes)
# ---------------------------------------------------------------------
def global_topk_from_topk_abs(entries, masks, k=3):
    """
    entries: list of dicts (one per hook call -- may cover one scene or a
    whole batch of scenes) with top_vals/median_abs.
    Returns: array (n_valid_scenes, k) with the global Top1..Topk per
    scene, and array (n_valid_scenes,) with the approximate median.
    """
    top_per_scene, median_per_scene = [], []
    scene_counter = 0

    for e in entries:
        top_vals_scenes = split_scenes(e["top_vals"], non_batch_ndim=2)   # each (seq, k_saved)
        median_scenes = split_scenes(e["median_abs"], non_batch_ndim=1)  # each (seq,)
        assert len(top_vals_scenes) == len(median_scenes), \
            "top_vals and median_abs disagree on the number of scenes in this entry"

        for top_vals, median_abs in zip(top_vals_scenes, median_scenes):
            L = real_length(masks, scene_counter) if masks is not None else top_vals.shape[0]
            scene_counter += 1

            top_vals_np = top_vals[:L].float().numpy()
            median_abs_np = median_abs[:L].float().numpy()

            pooled_sorted = np.sort(top_vals_np.flatten())[::-1]
            if len(pooled_sorted) < k:
                continue
            top_per_scene.append(pooled_sorted[:k])
            median_per_scene.append(np.median(median_abs_np))

    if len(top_per_scene) == 0:
        return None, None
    return np.stack(top_per_scene, axis=0), np.array(median_per_scene)


# ---------------------------------------------------------------------
# Exact Top1/2/3 + median from raw (sanity check, few scenes)
# ---------------------------------------------------------------------
def exact_topk_and_median_from_raw(entries, masks, k=3):
    top_per_scene, median_per_scene = [], []
    scene_counter = 0

    for e in entries:
        h_scenes = split_scenes(e.abs(), non_batch_ndim=2)  # each (seq, hidden)

        for h in h_scenes:
            L = real_length(masks, scene_counter) if masks is not None else h.shape[0]
            scene_counter += 1

            h_np = h[:L].float().numpy()
            flat_sorted = np.sort(h_np.flatten())[::-1]
            if len(flat_sorted) < k:
                continue
            top_per_scene.append(flat_sorted[:k])
            median_per_scene.append(np.median(h_np))

    if len(top_per_scene) == 0:
        return None, None
    return np.stack(top_per_scene, axis=0), np.array(median_per_scene)


# ---------------------------------------------------------------------
# Aggregate per layer
# ---------------------------------------------------------------------
def collect_stream(data, masks, norm_filter, n_layers, k=3, use_raw=False):
    top_means = np.full((n_layers, k), np.nan)
    top_stds = np.full((n_layers, k), np.nan)
    med_means = np.full(n_layers, np.nan)
    med_stds = np.full(n_layers, np.nan)

    keys = [key for key in data if "lang_encoder" in key]
    for key in keys:
        layer_idx, norm = parse_layer_norm(key)
        if layer_idx is None or norm != norm_filter or layer_idx >= n_layers:
            continue

        entries = data[key]
        if use_raw:
            tops, meds = exact_topk_and_median_from_raw(entries, masks, k=k)
        else:
            tops, meds = global_topk_from_topk_abs(entries, masks, k=k)

        if tops is None:
            continue

        top_means[layer_idx] = tops.mean(axis=0)
        top_stds[layer_idx] = tops.std(axis=0)
        med_means[layer_idx] = meds.mean()
        med_stds[layer_idx] = meds.std()

    return top_means, top_stds, med_means, med_stds


# ---------------------------------------------------------------------
# Plot in the style of Sun et al. Fig. 4
# ---------------------------------------------------------------------
def plot_stream(ax, top_means, top_stds, med_means, med_stds, title, k=3, log_scale=False):
    layers = np.arange(len(med_means))
    colors = ["tab:red", "tab:orange", "tab:green"]

    for i in range(k):
        ax.errorbar(
            layers, top_means[:, i], yerr=top_stds[:, i],
            marker="o", markersize=3, capsize=2, label=f"Top {i+1}",
            color=colors[i % len(colors)],
        )

    ax.errorbar(
        layers, med_means, yerr=med_stds,
        marker="s", markersize=3, capsize=2, label="Median",
        color="tab:blue", linestyle="--",
    )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Magnitude" + (" (log)" if log_scale else ""))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_data", type=str, required=True,
                         help="Path to the .pt file with topk_abs/<module> keys")
    parser.add_argument("--masks", type=str, default=None,
                         help="Path to the .pt file with attention masks (True=padding), same as plot_massive_3d.py")
    parser.add_argument("--raw_data", type=str, default=None,
                         help="Optional: .pt file with raw/<module> keys, for the sanity check")
    parser.add_argument("--out_dir", type=str, default="./plots")
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--which", type=str, default="both", choices=["norm1", "norm2", "both"])
    parser.add_argument("--log_scale", action="store_true",
                         help="Use a log y-axis (default: linear)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_scale = args.log_scale

    data = torch.load(args.topk_data, map_location="cpu")
    data = {name: entries for name, entries in data.items() if name.startswith("topk_abs/")}
    print(f"Loaded {len(data)} topk_abs keys from {args.topk_data}")
    import pdb; pdb.set_trace()
    masks = torch.load(args.masks, map_location="cpu") if args.masks else None
    if masks is None:
        print("WARNING: no mask provided, using the full tensor length "
              "(padding tokens may contaminate the median and top-k).")

    norms_to_plot = ["norm1", "norm2"] if args.which == "both" else [args.which]
    fig, axes = plt.subplots(len(norms_to_plot), 1, figsize=(14, 4.5 * len(norms_to_plot)),
                              squeeze=False, sharey=True)
    axes = axes[:, 0]

    for ax, norm_name in zip(axes, norms_to_plot):
        top_means, top_stds, med_means, med_stds = collect_stream(
            data, masks, norm_name, args.n_layers, k=args.k, use_raw=False
        )
        plot_stream(ax, top_means, top_stds, med_means, med_stds,
                    title=f"{norm_name}", k=args.k, log_scale=log_scale)

        print(f"\n{norm_name} -- per-layer values (mean \u00b1 std over scenes):")
        header = "  layer | " + " | ".join(f"Top{i+1}" for i in range(args.k)) + " | median | " + \
                 " | ".join(f"Top{i+1}/med" for i in range(args.k))
        print(header)
        for layer_idx in range(args.n_layers):
            if np.isnan(med_means[layer_idx]):
                print(f"  {layer_idx:5d} | (no data)")
                continue
            top_str = " | ".join(
                f"{top_means[layer_idx, i]:6.2f}\u00b1{top_stds[layer_idx, i]:5.2f}" for i in range(args.k)
            )
            med = med_means[layer_idx]
            med_std = med_stds[layer_idx]
            ratio_str = " | ".join(f"{top_means[layer_idx, i] / med:8.2f}x" for i in range(args.k))
            print(f"  {layer_idx:5d} | {top_str} | {med:6.3f}\u00b1{med_std:5.3f} | {ratio_str}")

    fig.suptitle(
        "Three largest activation magnitudes and the median magnitude at each layer in JINA-CLIPv2",
        fontsize=13,
    )
    fig.tight_layout()

    stem = Path(args.topk_data).stem
    fname = out_dir / f"{stem}_massive_per_layer_{args.which}.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")

    # ------------------------------------------------------------------
    # Optional sanity check on the exact median/top-k (few scenes)
    # ------------------------------------------------------------------
    if args.raw_data is not None:
        raw_data = torch.load(args.raw_data, map_location="cpu")
        raw_data = {name: entries for name, entries in raw_data.items() if name.startswith("raw/")}
        print(f"\nSanity check (raw, {len(raw_data)} keys):")

        for norm_name in norms_to_plot:
            top_approx, _, med_approx, _ = collect_stream(
                data, masks, norm_name, args.n_layers, k=args.k, use_raw=False
            )
            top_exact, _, med_exact, _ = collect_stream(
                raw_data, masks, norm_name, args.n_layers, k=args.k, use_raw=True
            )

            valid = ~np.isnan(med_approx) & ~np.isnan(med_exact)
            if valid.sum() == 0:
                print(f"  {norm_name}: no overlapping layers between topk_abs and raw.")
                continue

            med_rel_err = np.abs(med_approx[valid] - med_exact[valid]) / (med_exact[valid] + 1e-8)
            top1_rel_err = np.abs(top_approx[valid, 0] - top_exact[valid, 0]) / (top_exact[valid, 0] + 1e-8)
            print(f"  {norm_name}: median rel. err mean={med_rel_err.mean():.4f} max={med_rel_err.max():.4f} | "
                  f"Top1 rel. err mean={top1_rel_err.mean():.4f} max={top1_rel_err.max():.4f}")


if __name__ == "__main__":
    main()