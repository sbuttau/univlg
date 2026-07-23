"""
Summary violin plots for the per-token norm distributions.

Unlike plot_backbone_and_decoder.py (which only plots mean_max +- std,
i.e. two numbers per point), here we plot the ENTIRE distribution of norm
values, so the SHAPE is directly visible:

  Scenario A (homogeneous growth, NOT a sink):
      the violin is compact and shifts/widens together, no isolated
      tails. max/median stays close to 1.

  Scenario B (true sink / isolated outlier):
      the bulk of the violin stays compact, but a thin tail sticks out
      towards high values -- a few tokens (e.g. prefix/registers)
      detached from the rest. max/median is high.

Five figures:

  1) DINO: a violin plot with ALL resolutions (res0..res3) on the x-axis,
     and for each one two side-by-side violins (with prefix / without
     prefix), to see both the prefix effect and the growth with depth in
     one shot.

  2) Pixel decoder: 3 blocks, each with cross-view-attn / ffn / post-ffn
     (gate), as a single sequential violin flow (no prefix or query/text
     concept here).

  3) Mask decoder (query + text): two subplots (query tokens on top,
     text tokens at the bottom), full sequential flow across ALL layers
     AND ALL sub-layers (cross-attn, self-attn, ffn), to see how the
     distribution behaves with depth for each sub-layer.

  4) Mask decoder (vis output tokens): a separate figure, since these use
     a different pair of sub-layers (vis cross-attn / vis ffn), full
     sequential flow across all layers.

  5) Text encoder (Jina-BERT-v2): emb_ln + 12 layers (norm1/norm2), full
     sequential flow, loaded from a SEPARATE .pt file (text-only hooks).

Usage:
    python plot_norm_distributions.py /path/to/filtered_norms.pt --out_dir ./plots
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from plot_backbone_and_decoder import get_tensor, entries_matching
from norm_hooks import split_post_dino_by_resolution

_T0 = time.time()
_RNG = np.random.default_rng(0)


def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def subsample(arr, max_n=15000):
    """
    Subsample an array before feeding it to the violin plot (violinplot's
    KDE scales badly with size, easily minutes on arrays with hundreds of
    thousands / millions of points). The printed statistics (median/max)
    are computed SEPARATELY on the full array, so this subsampling only
    affects the visual shape of the violin, not the reported numbers.
    """
    if len(arr) <= max_n:
        return arr
    idx = _RNG.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


# ---------------------------------------------------------------------
# Raw value extraction (no aggregation into mean/std/ratio)
# ---------------------------------------------------------------------
def flatten_all_values(entries, exclude_first_n=0, max_total=20000, rng=None):
    """
    Concatenate per-token norm values from a list of entries, but
    subsampling PER-ENTRY BEFORE concatenating, so the full array is
    never built (which, with thousands of tokens per scene and hundreds
    of scenes, can reach tens of millions of elements and become the
    real bottleneck otherwise).

    max_total: approximate total point budget for the final result.
    It's split evenly across entries (max(1, max_total // n_entries) per
    entry); if an entry has fewer tokens than the budget, all are kept.

    Note: the statistics (median/max) computed on this array are
    therefore APPROXIMATE (sample-based estimate), not exact. For exact
    numbers use the statistics already computed by
    plot_backbone_and_decoder.py.
    """
    if rng is None:
        rng = _RNG
    out = []
    n_entries = max(1, len(entries))
    per_entry_budget = max(1, max_total // n_entries)
    for e in entries:
        t = get_tensor(e)
        if t is None:
            continue
        t = t.float()
        if exclude_first_n > 0 and t.dim() == 2:
            d0, d1 = t.shape
            if d1 >= d0:
                t = t[:, exclude_first_n:]
            else:
                t = t[exclude_first_n:, :]
        flat = t.flatten().numpy()
        if flat.size > per_entry_budget:
            idx = rng.choice(flat.size, size=per_entry_budget, replace=False)
            flat = flat[idx]
        out.append(flat)
    if len(out) == 0:
        return np.array([])
    return np.concatenate(out)


def flatten_token_subset(entries, start, end=None, max_total=20000, rng=None):
    """
    Same as flatten_all_values, but restricted to a sub-range of tokens
    along the sequence (query or text inside a combined sequence), with
    the same per-entry subsampling before concatenating.
    end=None -> go to the end of the tensor for THAT specific scene
    (needed for text, whose length varies per sentence).
    """
    if rng is None:
        rng = _RNG
    out = []
    n_entries = max(1, len(entries))
    per_entry_budget = max(1, max_total // n_entries)
    for e in entries:
        t = get_tensor(e)
        if t is None:
            continue
        flat = t.float().flatten().numpy()
        subset = flat[start:end] if end is not None else flat[start:]
        if subset.size == 0:
            continue
        if subset.size > per_entry_budget:
            idx = rng.choice(subset.size, size=per_entry_budget, replace=False)
            subset = subset[idx]
        out.append(subset)
    if len(out) == 0:
        return np.array([])
    return np.concatenate(out)


# ---------------------------------------------------------------------
# Helper for grouped violin plots (several groups side-by-side per x)
# ---------------------------------------------------------------------
def _style_violin(parts, color):
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_edgecolor(color)
            parts[key].set_linewidth(1.2)


def grouped_violin_plot(ax, groups, x_labels, colors, width=0.35, log_y=True):
    """
    groups: dict {group_label: [array_per_x_position]} -- each list must
        have the same length as x_labels (one array, possibly empty, per
        x-axis position).
    Draws, for each x position, one violin per group, side by side with
    a horizontal offset.
    """
    x_positions = np.arange(len(x_labels))
    n_groups = len(groups)

    for i, (label, arrays) in enumerate(groups.items()):
        offset = (i - (n_groups - 1) / 2) * width
        positions, data = [], []
        for x, arr in zip(x_positions, arrays):
            if arr is not None and len(arr) > 1:
                positions.append(x + offset)
                data.append(arr)
        if len(data) == 0:
            continue
        parts = ax.violinplot(data, positions=positions, widths=width * 0.9,
                               showmedians=True, showextrema=True)
        _style_violin(parts, colors[label])
        # dummy entry for the legend (violins don't support a direct label)
        ax.plot([], [], color=colors[label], alpha=0.55, linewidth=8, label=label)

    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")


# ---------------------------------------------------------------------
# Plot 1: DINO, all resolutions, with/without prefix
# ---------------------------------------------------------------------
def plot_dino_summary(data, out_dir, n_resolutions=4, n_prefix=5):
    dino_key = "visual_backbone.backbone.dinov2.inner.norm"
    if dino_key not in data:
        print(f"Warning: key '{dino_key}' not found, skipping the DINO plot.")
        return

    log(f"DINO: splitting by resolution ({len(data[dino_key])} total entries)...")
    by_res = split_post_dino_by_resolution(data[dino_key], n_resolutions=n_resolutions)

    x_labels = [f"res{i}" for i in range(n_resolutions)]
    with_prefix, without_prefix = [], []
    for i in range(n_resolutions):
        entries = by_res.get(i, [])
        log(f"DINO res{i}: {len(entries)} entries, flatten+subsampling in progress...")
        with_prefix.append(flatten_all_values(entries, exclude_first_n=0))
        without_prefix.append(flatten_all_values(entries, exclude_first_n=n_prefix))
        log(f"DINO res{i}: done ({len(with_prefix[-1])} values with prefix, "
            f"{len(without_prefix[-1])} without -- NOTE: these are samples, not the full dataset)")

    log("DINO: building violin plot (subsampling large arrays)...")
    with_prefix_sub = [subsample(a) for a in with_prefix]
    without_prefix_sub = [subsample(a) for a in without_prefix]
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_violin_plot(
        ax,
        groups={
            f"with prefix (CLS+{n_prefix - 1} registers)": with_prefix_sub,
            "without prefix (patches only)": without_prefix_sub,
        },
        x_labels=x_labels,
        colors={
            f"with prefix (CLS+{n_prefix - 1} registers)": "tab:green",
            "without prefix (patches only)": "tab:blue",
        },
    )
    ax.set_ylabel("L2 norm per token (log scale)")
    ax.set_title("DINO: norm distribution per resolution, with vs without prefix tokens")
    fig.tight_layout()
    fname = out_dir / "dino_summary_violin.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log(f"Saved: {fname}")

    print("\nDINO summary (median / max / ratio):")
    for lbl, arr in zip(x_labels, without_prefix):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"  {lbl} without prefix: median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")
    for lbl, arr in zip(x_labels, with_prefix):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"  {lbl} with prefix:    median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")


# ---------------------------------------------------------------------
# Plot 2: pixel decoder, sequential flow across blocks/sub-layers
# ---------------------------------------------------------------------
def plot_pixel_decoder_summary(data, out_dir, n_blocks=3, max_total=20000):
    """
    Pixel decoder: 3 blocks, each with cross-view-attn / ffn / post-ffn
    (gate). No prefix or query/text concept here -- this is the same
    sub-layer sequence already present in plot_backbone_and_decoder.py
    (plot_visual_backbone), but as a full distribution instead of just
    mean_max+-std.
    """
    sublayer_patterns = [
        ("cross", "cross_view_attention_layers"),
        ("ffn", "ffn_layers"),
        ("post-ffn\n(gate)", "layer_norms"),
    ]

    values, x_labels = [], []
    for block_idx in range(n_blocks):
        for label, subpattern in sublayer_patterns:
            full_pattern = f"pixel_decoder.cross_view_attn.{block_idx}.{subpattern}"
            log(f"Pixel decoder blk{block_idx} {label.strip()}: looking for entries matching '{full_pattern}'...")
            entries = entries_matching(data, full_pattern)
            if len(entries) == 0:
                log(f"Pixel decoder blk{block_idx} {label.strip()}: no entries found, skipping.")
                continue
            vals = flatten_all_values(entries, exclude_first_n=0, max_total=max_total)
            values.append(vals)
            x_labels.append(f"blk{block_idx}\n{label}")
            log(f"Pixel decoder blk{block_idx} {label.strip()}: {len(entries)} entries, "
                f"{len(vals)} sampled values")

    if len(values) == 0:
        print("No data for the pixel decoder plot.")
        return

    log("Pixel decoder: building violin plot...")
    fig, ax = plt.subplots(figsize=(11, 6))
    grouped_violin_plot(
        ax, groups={"pixel decoder": values}, x_labels=x_labels,
        colors={"pixel decoder": "tab:purple"}, width=0.6,
    )
    ax.set_ylabel("L2 norm per token (log scale)")
    ax.set_xlabel("Pixel decoder block / sub-layer")
    ax.set_title("Pixel decoder: norm distribution per block and sub-layer")
    fig.tight_layout()
    fname = out_dir / "pixel_decoder_summary_violin.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log(f"Saved: {fname}")

    print("\nPixel decoder summary (median / max / ratio, on sample):")
    for lbl, arr in zip(x_labels, values):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"  {lbl.replace(chr(10), ' ')}: median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")


# ---------------------------------------------------------------------
# Plot 3: mask decoder, all layers, query and text separated
# ---------------------------------------------------------------------
def _build_layer_sublayer_labels(n_layers, sublayer_labels):
    """
    Builds x-axis labels for a sequential flow of (layer, sub-layer)
    combinations, e.g. ["L0\ncross-attn", "L0\nself-attn", "L0\nffn",
    "L1\ncross-attn", ...], plus the list of positions where a layer
    boundary falls (for drawing vertical separators).
    """
    x_labels = []
    boundaries = []
    pos = 0
    for layer_idx in range(n_layers):
        for label in sublayer_labels:
            x_labels.append(f"L{layer_idx}\n{label}")
            pos += 1
        boundaries.append(pos - 0.5)
    if boundaries:
        boundaries = boundaries[:-1]  # the last boundary is the end of the plot, not needed
    return x_labels, boundaries


def plot_mask_decoder_summary(data, out_dir, n_layers=8, n_queries=100, max_total=20000):
    """
    Mask decoder, full sequential flow across ALL sub-layers (cross-attn,
    self-attn, ffn) and ALL layers, for query tokens and text tokens
    separately, plus a third figure for the vis output tokens (which use
    a different pair of sub-layers: vis cross-attn / vis ffn).
    """
    query_text_patterns = {
        "cross-attn": "transformer_cross_attention_layers",
        "self-attn": "transformer_self_attention_layers",
        "ffn": "transformer_ffn_layers",
    }
    vis_output_patterns = {
        "vis cross-attn": "vis_output_cross_attn",
        "vis ffn": "vis_output_ffn",
    }

    # --- query + text tokens, across cross-attn/self-attn/ffn ---
    sublayer_labels = list(query_text_patterns.keys())
    x_labels, boundaries = _build_layer_sublayer_labels(n_layers, sublayer_labels)

    query_values, text_values = [], []
    for layer_idx in range(n_layers):
        for label in sublayer_labels:
            pattern = query_text_patterns[label]
            log(f"Mask decoder L{layer_idx} {label}: looking for entries matching '{pattern}.{layer_idx}.'...")
            entries = entries_matching(data, f"{pattern}.{layer_idx}.")
            log(f"Mask decoder L{layer_idx} {label}: {len(entries)} entries found, extracting query/text...")
            query_values.append(flatten_token_subset(entries, 0, n_queries, max_total=max_total))
            text_values.append(flatten_token_subset(entries, n_queries, end=None, max_total=max_total))

    # --- vis output tokens, across vis cross-attn/vis ffn ---
    vis_sublayer_labels = list(vis_output_patterns.keys())
    vis_x_labels, vis_boundaries = _build_layer_sublayer_labels(n_layers, vis_sublayer_labels)

    vis_values = []
    for layer_idx in range(n_layers):
        for label in vis_sublayer_labels:
            pattern = vis_output_patterns[label]
            log(f"Mask decoder L{layer_idx} {label}: looking for entries matching '{pattern}.{layer_idx}.'...")
            entries = entries_matching(data, f"{pattern}.{layer_idx}.")
            log(f"Mask decoder L{layer_idx} {label}: {len(entries)} entries found, extracting vis output...")
            vis_values.append(flatten_all_values(entries, exclude_first_n=0, max_total=max_total))

    log("Mask decoder: building violin plots (subsampling large arrays)...")
    query_sub = [subsample(a) for a in query_values]
    text_sub = [subsample(a) for a in text_values]
    vis_sub = [subsample(a) for a in vis_values]

    # --- Figure 1: query vs text tokens, full sequential flow ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    grouped_violin_plot(
        ax, groups={"query tokens": query_sub}, x_labels=x_labels,
        colors={"query tokens": "tab:blue"}, width=0.6,
    )
    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel("L2 norm (log scale)")
    ax.set_title("Mask decoder: query tokens -- full sequential flow (cross-attn / self-attn / ffn)")

    ax = axes[1]
    grouped_violin_plot(
        ax, groups={"text tokens": text_sub}, x_labels=x_labels,
        colors={"text tokens": "tab:orange"}, width=0.6,
    )
    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel("L2 norm (log scale)")
    ax.set_xlabel("Layer / sub-layer of the mask decoder")
    ax.set_title("Mask decoder: text tokens -- full sequential flow (cross-attn / self-attn / ffn)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    plt.setp(axes[0].get_xticklabels(), rotation=60, ha="right", fontsize=7)

    fig.tight_layout()
    fname = out_dir / "mask_decoder_summary_violin.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log(f"Saved: {fname}")

    # --- Figure 2: vis output tokens, full sequential flow ---
    fig, ax = plt.subplots(figsize=(14, 6))
    grouped_violin_plot(
        ax, groups={"vis output tokens": vis_sub}, x_labels=vis_x_labels,
        colors={"vis output tokens": "tab:red"}, width=0.6,
    )
    for b in vis_boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel("L2 norm (log scale)")
    ax.set_xlabel("Layer / sub-layer of the mask decoder")
    ax.set_title("Mask decoder: vis output tokens -- full sequential flow (vis cross-attn / vis ffn)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    fname_vis = out_dir / "mask_decoder_vis_output_violin.png"
    fig.savefig(fname_vis, dpi=150)
    plt.close(fig)
    log(f"Saved: {fname_vis}")

    print("\nMask decoder summary (median / max / ratio):")
    print("  Query tokens:")
    for lbl, arr in zip(x_labels, query_values):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"    {lbl.replace(chr(10), ' ')}: median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")
    print("  Text tokens:")
    for lbl, arr in zip(x_labels, text_values):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"    {lbl.replace(chr(10), ' ')}: median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")
    print("  Vis output tokens:")
    for lbl, arr in zip(vis_x_labels, vis_values):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"    {lbl.replace(chr(10), ' ')}: median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")


# ---------------------------------------------------------------------
# Plot 5: text encoder (Jina-BERT-v2), sequential flow
# ---------------------------------------------------------------------
def plot_text_encoder_summary(data, out_dir, n_layers=12, max_total=20000,
                               base_prefix="lang_encoder.text_encoder.text_model.transformer"):
    """
    Same sequential flow as plot_text_encoder in plot_backbone_and_decoder.py
    (emb_ln -> layer0.norm1 -> layer0.norm2 -> ... -> layer{n-1}.norm2), but
    as full distributions (violins) instead of just mean_max+-std.
    """
    values, x_labels = [], []

    emb_ln_pattern = f"{base_prefix}.emb_ln"
    log(f"Text encoder emb_ln: looking for entries matching '{emb_ln_pattern}'...")
    entries = entries_matching(data, emb_ln_pattern)
    if len(entries) > 0:
        values.append(flatten_all_values(entries, exclude_first_n=0, max_total=max_total))
        x_labels.append("emb_ln")
        log(f"Text encoder emb_ln: {len(entries)} entries, {len(values[-1])} sampled values")
    else:
        log("Text encoder emb_ln: no entries found, skipping.")

    for layer_idx in range(n_layers):
        for sub_label in ("norm1", "norm2"):
            full_pattern = f"{base_prefix}.encoder.layers.{layer_idx}.{sub_label}"
            log(f"Text encoder L{layer_idx} {sub_label}: looking for entries matching '{full_pattern}'...")
            entries = entries_matching(data, full_pattern)
            if len(entries) == 0:
                log(f"Text encoder L{layer_idx} {sub_label}: no entries found, skipping.")
                continue
            values.append(flatten_all_values(entries, exclude_first_n=0, max_total=max_total))
            x_labels.append(f"L{layer_idx}\n{sub_label}")
            log(f"Text encoder L{layer_idx} {sub_label}: {len(entries)} entries, "
                f"{len(values[-1])} sampled values")

    if len(values) == 0:
        print("No data for the text encoder plot.")
        return

    log("Text encoder: building violin plot...")
    values_sub = [subsample(a) for a in values]
    fig, ax = plt.subplots(figsize=(15, 6))
    grouped_violin_plot(
        ax, groups={"text encoder": values_sub}, x_labels=x_labels,
        colors={"text encoder": "tab:green"}, width=0.6,
    )
    ax.set_ylabel("L2 norm per token (log scale)")
    ax.set_xlabel("Text encoder layer / sub-layer")
    ax.set_title("Text encoder (Jina-BERT-v2): norm distribution, emb_ln + 12 layers")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=6)
    fig.tight_layout()
    fname = out_dir / "text_encoder_summary_violin.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log(f"Saved: {fname}")

    print("\nText encoder summary (median / max / ratio, on sample):")
    for lbl, arr in zip(x_labels, values):
        if len(arr) == 0:
            continue
        med, mx = np.median(arr), arr.max()
        print(f"  {lbl.replace(chr(10), ' ')}: median={med:.3f}  max={mx:.3f}  max/median={mx / med:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norms_file", type=str,
                         default="tests/scanrefer_scannet_anchor_val_single_batched_hook_norms.pt")
    parser.add_argument("--out_dir", type=str, default="./plots")
    parser.add_argument("--n_mask_decoder_layers", type=int, default=8)
    parser.add_argument("--n_queries", type=int, default=100,
                         help="Number of queries in the combined query+text sequence of the mask decoder.")
    parser.add_argument("--n_prefix", type=int, default=5,
                         help="Number of prefix tokens (CLS + registers) to exclude in DINO.")
    parser.add_argument("--n_pixel_decoder_blocks", type=int, default=3)
    parser.add_argument("--text_norms_file", type=str, default=None,
                         help="Separate .pt file with text encoder norms (if saved separately, "
                              "e.g. from a dedicated text-only hook batch). If omitted, this plot is skipped.")
    parser.add_argument("--n_text_encoder_layers", type=int, default=12)
    parser.add_argument("--skip_dino", action="store_true")
    parser.add_argument("--skip_pixel_decoder", action="store_true")
    parser.add_argument("--skip_mask_decoder", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needs_main_file = not (args.skip_dino and args.skip_pixel_decoder and args.skip_mask_decoder)
    if needs_main_file:
        log(f"Loading {args.norms_file} (may take a while if the file is large)...")
        data = torch.load(args.norms_file, map_location="cpu")
        log(f"Loaded: {len(data)} modules found.")

        if not args.skip_dino:
            log("--- Starting DINO plot ---")
            plot_dino_summary(data, out_dir, n_prefix=args.n_prefix)
            log("--- Finished DINO plot ---\n")
        else:
            log("Skipping DINO plot (--skip_dino).")

        if not args.skip_pixel_decoder:
            log("--- Starting pixel decoder plot ---")
            plot_pixel_decoder_summary(data, out_dir, n_blocks=args.n_pixel_decoder_blocks)
            log("--- Finished pixel decoder plot ---\n")
        else:
            log("Skipping pixel decoder plot (--skip_pixel_decoder).")

        if not args.skip_mask_decoder:
            log("--- Starting mask decoder plot ---")
            plot_mask_decoder_summary(data, out_dir, n_layers=args.n_mask_decoder_layers,
                                       n_queries=args.n_queries)
            log("--- Finished mask decoder plot ---")
        else:
            log("Skipping mask decoder plot (--skip_mask_decoder).")
    else:
        log("Skipping main file load (DINO, pixel decoder, and mask decoder plots are all disabled).")

    if args.text_norms_file is not None:
        log("--- Starting text encoder plot ---")
        text_data = torch.load(args.text_norms_file, map_location="cpu")
        log(f"Loaded text encoder norms data from {args.text_norms_file}, {len(text_data)} modules found.")
        plot_text_encoder_summary(text_data, out_dir, n_layers=args.n_text_encoder_layers)
        log("--- Finished text encoder plot ---")


if __name__ == "__main__":
    main()