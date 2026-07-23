import argparse
from collections import Counter
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- utils

def get_norm_tensor(entry):
    return entry["norms"] if isinstance(entry, dict) else entry


def seq_length_from_mask(mask):
    # mask: True = padding, False = real token -> count the False
    return (~mask).sum(dim=-1)


def scalar_len(L):
    return L[0].item() if L.dim() > 0 else L.item()


def get_seq(entry):
    """Return the 1D norm sequence for a scene (drop batch dim if present)."""
    norms = get_norm_tensor(entry)
    return norms[0] if norms.dim() > 1 else norms


def short_layer_name(module_name):
    """'...encoder.layers.7.norm2' -> 'L7 · norm2'."""
    parts = module_name.split(".")
    try:
        i = parts.index("layers")
        return f"L{parts[i + 1]} · {parts[-1]}"
    except ValueError:
        return module_name


# ---------------------------------------------------------------- checks

def argmax_per_scene(data, module_name, seq_lengths):
    """Argmax restricted to real tokens only."""
    indices = []
    for i, entry in enumerate(data[module_name]):
        seq = get_seq(entry)
        L = scalar_len(seq_lengths[i])
        indices.append(seq[:L].argmax().item())
    return indices


def check_second_peak_is_sep(data, module_name, seq_lengths, exclude_idx=0):
    match_count, total = 0, 0
    for i, entry in enumerate(data[module_name]):
        seq = get_seq(entry).clone()
        L = scalar_len(seq_lengths[i])

        # exclude CLS *and* padding from the argmax
        seq[exclude_idx] = -float("inf")
        seq[L:] = -float("inf")

        second_peak_idx = seq.argmax().item()
        expected_sep_idx = L - 1

        match_count += (second_peak_idx == expected_sep_idx)
        total += 1
    pct = 100 * match_count / total
    print(f"  second peak == SEP position: {match_count}/{total} ({pct:.1f}%)")
    return match_count / total


# ---------------------------------------------------------------- plots

def plot_median_profile(data, module_names, seq_lengths, save_path,
                        min_coverage=0.5):
    """One panel per layer: median norm per token position across scenes,
    with 25-75 percentile band. Padding is excluded position-wise; positions
    covered by fewer than `min_coverage` of the scenes are dropped (they
    would be dominated by a handful of long sentences)."""
    n = len(module_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]

    lengths = [scalar_len(L) for L in seq_lengths]
    n_scenes = len(lengths)

    for ax, name in zip(axes, module_names):
        entries = data[name]
        max_len = max(len(get_seq(e)) for e in entries)

        # stack into (n_scenes, max_len) with NaN on padding positions
        stacked = np.full((len(entries), max_len), np.nan)
        for i, entry in enumerate(entries):
            seq = get_seq(entry).float().numpy()
            L = lengths[i]
            stacked[i, :L] = seq[:L]

        support = (~np.isnan(stacked)).sum(axis=0)
        valid = support >= min_coverage * n_scenes
        x = np.arange(max_len)[valid]

        med = np.nanmedian(stacked, axis=0)[valid]
        q25 = np.nanpercentile(stacked, 25, axis=0)[valid]
        q75 = np.nanpercentile(stacked, 75, axis=0)[valid]

        ax.plot(x, med, color="tab:blue", linewidth=1.2)
        ax.fill_between(x, q25, q75, color="tab:blue", alpha=0.25, linewidth=0)

        ax.set_title(short_layer_name(name))
        ax.set_xlabel("token position")

    axes[0].set_ylabel("norm")
    fig.suptitle(f"Median norm per position (IQR band) — {n_scenes} scenes",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


def plot_class_boxplot(data, module_names, seq_lengths, save_path):
    """One panel per layer: norm distribution for CLS, SEP and content
    tokens, pooled across all scenes. Alignment-free view of the sinks."""
    n = len(module_names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]

    lengths = [scalar_len(L) for L in seq_lengths]

    for ax, name in zip(axes, module_names):
        cls_vals, sep_vals, content_vals = [], [], []
        for i, entry in enumerate(data[name]):
            seq = get_seq(entry).float().numpy()
            L = lengths[i]
            cls_vals.append(seq[0])
            sep_vals.append(seq[L - 1])
            content_vals.extend(seq[1:L - 1])

        ax.boxplot([cls_vals, sep_vals, content_vals],
                   tick_labels=["CLS", "SEP", "content"],
                   showfliers=False)
        ax.set_title(short_layer_name(name))

    axes[0].set_ylabel("norm")
    fig.suptitle(f"Norm by token class — {len(lengths)} scenes", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


def plot_overlay_with_sep(data, module_name, seq_lengths, n_scenes, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    n_scenes = min(n_scenes, len(data[module_name]))
    for i in range(n_scenes):
        seq = get_seq(data[module_name][i]).numpy()
        L = scalar_len(seq_lengths[i])
        # plot real tokens only
        ax.plot(range(L), seq[:L], linewidth=0.7, alpha=0.3, color="tab:blue")
        ax.axvline(L - 1, color="tab:red", alpha=0.15, linewidth=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("token position")
    ax.set_ylabel("norm (log)")
    ax.set_title(f"{short_layer_name(module_name)} — {n_scenes} scenes (red = SEP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


def plot_length_distribution(seq_lengths, save_path):
    """Histogram of real sentence lengths (in tokens, CLS/SEP included)."""
    lengths = [scalar_len(L) for L in seq_lengths]
    lo, hi = min(lengths), max(lengths)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.arange(lo - 0.5, hi + 1.5, 1)  # one bin per integer length
    ax.hist(lengths, bins=bins, color="tab:blue", edgecolor="white")
    ax.set_xlabel("sentence length (tokens, incl. CLS/SEP)")
    ax.set_ylabel("n scenes")
    ax.set_title(f"Length distribution — {len(lengths)} scenes "
                 f"(min {lo}, max {hi}, median {int(np.median(lengths))})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


def plot_profile_by_length(data, module_names, seq_lengths, save_path,
                           min_count=5, max_groups=4):
    """Mean +/- std profiles stratified by *exact* sentence length.
    Within a group all scenes share the same length, so every position
    (including SEP at L-1) is aligned and mean/std are meaningful again.
    Rows = length groups (most populated first), cols = layers."""
    lengths = np.array([scalar_len(L) for L in seq_lengths])

    counts = Counter(lengths.tolist())
    groups = [L for L, c in counts.most_common() if c >= min_count][:max_groups]
    groups = sorted(groups)
    if not groups:
        print(f"no length with >= {min_count} scenes, skipping stratified plot "
              f"(most common: {counts.most_common(3)})")
        return

    n_rows, n_cols = len(groups), len(module_names)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 2.6 * n_rows),
                             sharex="row", sharey=True, squeeze=False)

    for r, L_group in enumerate(groups):
        scene_ids = np.where(lengths == L_group)[0]
        for c, name in enumerate(module_names):
            ax = axes[r][c]
            stacked = np.stack([
                get_seq(data[name][i]).float().numpy()[:L_group]
                for i in scene_ids
            ])  # (n_group, L_group) — fully aligned
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)

            x = np.arange(L_group)
            ax.plot(x, mean, color="tab:blue", linewidth=1.2)
            ax.fill_between(x, mean - std, mean + std,
                            color="tab:blue", alpha=0.25, linewidth=0)
            ax.axvline(L_group - 1, color="tab:red", alpha=0.4,
                       linewidth=0.8, linestyle="--")

            if r == 0:
                ax.set_title(short_layer_name(name))
            if c == 0:
                ax.set_ylabel(f"L={L_group}\n(n={len(scene_ids)})")
            if r == n_rows - 1:
                ax.set_xlabel("token position")

    fig.suptitle("Mean norm ± std, stratified by exact sentence length "
                 "(dashed red = SEP)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path} (groups: "
          + ", ".join(f"L={g} n={counts[g]}" for g in groups) + ")")


def plot_longest_scene(data, module_names, seq_lengths, save_path):
    """Norm profile of the single scene with the longest sentence,
    one panel per layer, real tokens only, SEP marked."""
    lengths = np.array([scalar_len(L) for L in seq_lengths])
    i_max = int(lengths.argmax())
    L = int(lengths[i_max])

    n = len(module_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, module_names):
        seq = get_seq(data[name][i_max]).float().numpy()[:L]
        ax.plot(np.arange(L), seq, color="tab:blue", linewidth=1.2)
        ax.axvline(L - 1, color="tab:red", alpha=0.5,
                   linewidth=0.9, linestyle="--")
        ax.set_title(short_layer_name(name))
        ax.set_xlabel("token position")

    axes[0].set_ylabel("norm")
    fig.suptitle(f"Longest sentence — scene {i_max}, L={L} (dashed red = SEP)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path} (scene {i_max}, L={L})")


# ---------------------------------------------------------------- main

def main(data_path, mask_path, suspect_layers, n_overlay=30):
    data = torch.load(data_path, map_location="cpu", mmap=True)
    keep = {k: v for k, v in data.items() if "text_model" in k}
    torch.save(keep, f"{Path(data_path).stem}_norms_text_only.pt")
    print(f"kept {len(keep)}/{len(data)} modules")
    masks = torch.load(mask_path)

    for name in suspect_layers:
        assert name in data, f"'{name}' not found."
    for name in suspect_layers:
        assert len(masks) == len(data[name]), f"mismatch tra mask e norme ({name})!"

    seq_lengths = [seq_length_from_mask(m) for m in masks]

    # argmax coherence across suspect layers (real tokens only)
    print("argmax per layer (real tokens only):")
    all_argmax = {}
    for name in suspect_layers:
        idxs = argmax_per_scene(data, name, seq_lengths)
        all_argmax[name] = idxs
        frac0 = 100 * sum(i == 0 for i in idxs) / len(idxs)
        print(f"  {short_layer_name(name)}: argmax==0 in {frac0:.1f}% of scenes")

    print("\nSEP check per layer:")
    for name in suspect_layers:
        print(short_layer_name(name))
        check_second_peak_is_sep(data, name, seq_lengths)

    # output names derived from the input data file
    stem = Path(data_path).stem
    out_dir = Path(data_path).parent
    plot_median_profile(data, suspect_layers, seq_lengths,
                        save_path=out_dir / f"{stem}_median_profile.png")
    plot_class_boxplot(data, suspect_layers, seq_lengths,
                       save_path=out_dir / f"{stem}_class_boxplot.png")
    plot_overlay_with_sep(data, suspect_layers[0], seq_lengths, n_scenes=n_overlay,
                          save_path=out_dir / f"{stem}_overlay_sep.png")
    plot_length_distribution(seq_lengths,
                             save_path=out_dir / f"{stem}_length_hist.png")
    plot_profile_by_length(data, suspect_layers, seq_lengths,
                           save_path=out_dir / f"{stem}_profile_by_length.png")
    plot_longest_scene(data, suspect_layers, seq_lengths,
                       save_path=out_dir / f"{stem}_longest_scene.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="norms.pt")
    parser.add_argument("--masks", type=str, default="attention_masks.pt")
    parser.add_argument("--n-overlay", type=int, default=30)
    args = parser.parse_args()

    suspect_layers = [
        "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.0.norm2",
        # "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.7.norm2",
        "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.8.norm2",
        # "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.9.norm2",
    ]
    main(args.data, args.masks, suspect_layers, n_overlay=args.n_overlay)