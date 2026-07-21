"""Plot 3D stile Sun et al. (Massive Activations in LLMs, Fig. 1) per il
language encoder: |h| su assi (posizione token, indice dimensione), un
pannello per layer, per una scena scelta.

Input: file .pt salvato dal NormHookManager con attach_raw_hooks
(chiavi "raw/<module_name>", entries = tensori grezzi pre-norm).
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
    return f"L{m.group(1)} · {m.group(2)}" if m else name


def get_seq_hidden(t):
    """(B, seq, hidden) o (seq, hidden) -> (seq, hidden)."""
    return t[0] if t.dim() > 2 else t


def plot_3d_grid(data, keys, scene_idx, seq_len, save_path,
                 stride_dim=4, top_annotate=3):
    n = len(keys)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(4.2 * n_cols, 3.6 * n_rows))

    for p, key in enumerate(keys):
        h = get_seq_hidden(data[key][scene_idx]).abs().numpy()[:seq_len]
        seq, dim = h.shape

        ax = fig.add_subplot(n_rows, n_cols, p + 1, projection="3d")

        # superficie sottocampionata sulle dimensioni per leggerezza...
        dsub = np.arange(0, dim, stride_dim)
        X, Y = np.meshgrid(dsub, np.arange(seq))
        ax.plot_surface(X, Y, h[:, dsub], cmap="viridis",
                        rstride=1, cstride=1, linewidth=0, antialiased=False)

        # ...ma le guglie vere vanno disegnate senza sottocampionamento,
        # altrimenti rischi di perdere proprio le massive activations
        flat_idx = np.argsort(h, axis=None)[-top_annotate:]
        for fi in flat_idx:
            ti, di = np.unravel_index(fi, h.shape)
            ax.plot([di, di], [ti, ti], [0, h[ti, di]],
                    color="tab:red", linewidth=1.5)
            ax.text(di, ti, h[ti, di], f"  d{di},t{ti}",
                    fontsize=6, color="tab:red")

        ax.set_title(short_name(key), fontsize=9)
        ax.set_xlabel("dim", fontsize=7, labelpad=-4)
        ax.set_ylabel("token", fontsize=7, labelpad=-4)
        ax.tick_params(labelsize=6, pad=-2)

    fig.suptitle(f"|h| pre-norm — scene {scene_idx}, real tokens only",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="norms_raw.pt")
    parser.add_argument("--masks", type=str, default="attention_masks.pt")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--which", type=str, default="norm2",
                        choices=["norm1", "norm2", "both"])
    args = parser.parse_args()

    data = torch.load(args.data)
    masks = torch.load(args.masks)
    import pdb; pdb.set_trace()
    keys = sorted([k for k in data if k.startswith("raw/") and "lang_encoder" in k],
                  key=layer_sort_key)
    if args.which != "both":
        keys = [k for k in keys if k.endswith(args.which)]
    assert keys, "nessuna chiave 'raw/...lang_encoder...' trovata nel file."

    mask = masks[args.scene]
    L = (~mask).sum(dim=-1)
    L = int(L[0] if L.dim() > 0 else L)

    stem = Path(args.data).stem
    out = Path(args.data).parent / f"{stem}_3d_scene{args.scene}_{args.which}.png"
    plot_3d_grid(data, keys, args.scene, L, out)


if __name__ == "__main__":
    main()