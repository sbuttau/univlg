"""3D plot in the style of Sun et al. (Massive Activations in LLMs, Fig. 1)
for the language encoder: |h| on axes (token position, dimension index),
one panel per layer, for a chosen scene.

Input: .pt file saved by NormHookManager with attach_raw_hooks
(keys "raw/<module_name>", entries = raw pre-norm tensors).
"""

import argparse
import re
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def layer_sort_key(name):
    m = re.search(r"layers\.(\d+)\.(norm\d)", name)
    return (int(m.group(1)), m.group(2)) if m else (999, name)


def short_name(name):
    m = re.search(r"layers\.(\d+)\.(norm\d)", name)
    return f"L{m.group(1)} \u00b7 {m.group(2)}" if m else name


def get_seq_hidden(t):
    """(B, seq, hidden) or (seq, hidden) -> (seq, hidden)."""
    return t[0] if t.dim() > 2 else t


def plot_3d_grid(data, keys, scene_idx, seq_len, save_path,
                  stride_dim=4, top_annotate=1):
    n = len(keys)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(4.2 * n_cols, 3.6 * n_rows))

    for p, key in enumerate(keys):
        h = get_seq_hidden(data[key][scene_idx]).abs().numpy()[:seq_len]
        seq, dim = h.shape
        local_max = float(h.max())

        ax = fig.add_subplot(n_rows, n_cols, p + 1, projection="3d")

        # subsampled surface across dimensions for lightness. Color and
        # z-scale are normalized PER PANEL (local max), which keeps each
        # panel readable -- but that means panels are NOT directly
        # comparable by eye across layers. The real max value is printed
        # in the title so you can still tell which layers are actually
        # bigger without everything else looking flat/dark.
        dsub = np.arange(0, dim, stride_dim)
        Z = h[:, dsub]
        X, Y = np.meshgrid(dsub, np.arange(seq))
        ax.plot_surface(X, Y, Z, cmap="viridis",
                         rstride=1, cstride=1, linewidth=0, antialiased=False)

        # only the single largest value is annotated, to keep the panel
        # readable -- format is "dim=<dimension index>, tok=<token index>".
        # NOTE: mplot3d's 3D->2D projection does not guarantee the text
        # label visually sits right on top of the spike tip (a known
        # matplotlib limitation, independent of the viewing angle). A
        # marker dot at the exact tip removes the ambiguity even if the
        # text ends up offset on screen.
        flat_idx = np.argsort(h, axis=None)[-top_annotate:]
        for fi in flat_idx:
            ti, di = np.unravel_index(fi, h.shape)
            ax.plot([di, di], [ti, ti], [0, h[ti, di]],
                    color="tab:red", linewidth=1.5)
            ax.scatter([di], [ti], [h[ti, di]], color="tab:red", s=25,
                       depthshade=False, zorder=10)
            ax.text(di, ti, h[ti, di], f"  dim={di}, tok={ti}",
                    fontsize=6, color="tab:red")

        ax.set_title(f"{short_name(key)}  (max={local_max:.1f})", fontsize=9)
        ax.set_xlabel("dim", fontsize=7, labelpad=-4)
        ax.set_ylabel("token", fontsize=7, labelpad=-4)
        ax.tick_params(labelsize=6, pad=-2)

    fig.suptitle(f"|h| pre-norm \u2014 scene {scene_idx}, real tokens only\n"
                 f"(each panel normalized to its own max -- see title for the true value; "
                 f"red = single largest activation, dim=channel index, tok=token position)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="norms_raw.pt")
    parser.add_argument("--masks", type=str, default="attention_masks.pt")
    parser.add_argument("--scene", type=str, default="all",
                         help="Scene index (int), or 'all' to plot every scene present in --data")
    parser.add_argument("--which", type=str, default="norm2",
                         choices=["norm1", "norm2", "both"])
    args = parser.parse_args()

    data = torch.load(args.data)
    masks = torch.load(args.masks)

    keys = sorted([k for k in data if k.startswith("raw/") and "lang_encoder" in k],
                  key=layer_sort_key)
    if args.which != "both":
        keys = [k for k in keys if k.endswith(args.which)]
    assert keys, "no 'raw/...lang_encoder...' keys found in the file."

    n_scenes_available = len(data[keys[0]])
    if args.scene == "all":
        scene_indices = list(range(n_scenes_available))
    else:
        scene_indices = [int(args.scene)]

    stem = Path(args.data).stem
    for scene_idx in scene_indices:
        mask = masks[scene_idx]
        L = (~mask).sum(dim=-1)
        L = int(L[0] if L.dim() > 0 else L)

        out = Path(args.data).parent / f"{stem}_3d_scene{scene_idx}_{args.which}.png"
        plot_3d_grid(data, keys, scene_idx, L, out)


if __name__ == "__main__":
    main()