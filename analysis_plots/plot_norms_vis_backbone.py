"""
Script di debug per una singola scena: plotta la MAX norm (per checkpoint/layer)
insieme alla std dei token di quella stessa scena, per post_dino, pre_FFN, post_FFN.

Aggiunge statistiche utili per distinguere un vero "sink" isolato da un semplice
innalzamento diffuso dell'energia:
    - max / mean
    - max / median   (piu' robusto della media, meno sensibile a code della distribuzione)
    - coefficiente di variazione (std / mean)
    - numero (e percentuale) di token oltre 3 std sopra la media

Struttura attesa nel file .pt:
    {
        "post_dino": [scene0_tensor_layer0, scene0_tensor_layer1, ...]  # o [[scene..],[scene..],...] se multi-scena
        "pre_FFN":  {block_idx: [layer_tensor, layer_tensor, ...]},     # per singola scena
        "post_FFN": {block_idx: [layer_tensor, layer_tensor, ...]},
    }

Uso:
    python plot_norms_debug.py /path/to/dataset_visual_backbone_norms.pt --out_dir ./plots
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def collect_layers(obj, prefix=""):
    """
    Naviga ricorsivamente dict/list e restituisce (label, tensor) per ogni
    foglia trovata (un tensore di norme per-token, per un dato checkpoint/layer).
    """
    entries = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}/{key}" if prefix else str(key)
            entries.extend(collect_layers(value, new_prefix))

    elif isinstance(obj, list):
        if len(obj) == 0:
            return entries
        if torch.is_tensor(obj[0]):
            for idx, t in enumerate(obj):
                entries.append((f"{prefix}/L{idx}" if prefix else f"L{idx}", t))
        else:
            for idx, item in enumerate(obj):
                new_prefix = f"{prefix}/L{idx}" if prefix else f"L{idx}"
                entries.extend(collect_layers(item, new_prefix))

    elif torch.is_tensor(obj):
        entries.append((prefix, obj))

    return entries


def compute_stats(tensor):
    """
    Calcola statistiche estese su un tensore di norme per-token (flattenato),
    utili a distinguere un vero outlier isolato da un innalzamento diffuso.
    """
    flat = tensor.flatten().float()
    n = flat.numel()

    mx = flat.max().item()
    mn = flat.mean().item()
    md = flat.median().item()
    sd = flat.std().item()

    max_over_mean = mx / mn if mn != 0 else float("nan")
    max_over_median = mx / md if md != 0 else float("nan")
    coeff_var = sd / mn if mn != 0 else float("nan")

    threshold = mn + 3 * sd
    n_outliers = (flat > threshold).sum().item()
    pct_outliers = 100.0 * n_outliers / n

    return {
        "n_tokens": n,
        "max": mx,
        "mean": mn,
        "median": md,
        "std": sd,
        "max_over_mean": max_over_mean,
        "max_over_median": max_over_median,
        "coeff_var": coeff_var,
        "n_outliers_3std": n_outliers,
        "pct_outliers_3std": pct_outliers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("norms_file", type=str)
    parser.add_argument("--out_dir", type=str, default="./plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.norms_file, map_location="cpu")

    all_entries = []
    for top_key in ["post_dino", "pre_FFN", "post_FFN"]:
        if top_key not in data:
            print(f"Attenzione: '{top_key}' non trovato nel file, salto.")
            continue
        entries = collect_layers(data[top_key], prefix=top_key)
        all_entries.extend(entries)

    if len(all_entries) == 0:
        print("Nessuna entry trovata. Struttura di primo livello:", list(data.keys()))
        return

    labels = []
    all_stats = []

    header = (
        f"{'label':32s} {'max':>9s} {'mean':>9s} {'median':>9s} {'std':>9s} "
        f"{'max/mean':>9s} {'max/med':>9s} {'CoV':>7s} {'n_out(3std)':>12s} {'%out':>7s}"
    )
    print("\n" + header)
    print("-" * len(header))

    for label, tensor in all_entries:
        stats = compute_stats(tensor)
        labels.append(label)
        all_stats.append(stats)
        print(
            f"{label:32s} "
            f"{stats['max']:9.3f} {stats['mean']:9.3f} {stats['median']:9.3f} {stats['std']:9.3f} "
            f"{stats['max_over_mean']:9.3f} {stats['max_over_median']:9.3f} {stats['coeff_var']:7.3f} "
            f"{stats['n_outliers_3std']:12d} {stats['pct_outliers_3std']:6.2f}%"
        )

    # ----- Plot 1: max norm con errorbar (std tra i token) -----
    maxs = [s["max"] for s in all_stats]
    stds = [s["std"] for s in all_stats]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
    x = np.arange(len(labels))
    ax.bar(x, maxs, yerr=stds, capsize=4, alpha=0.8, label="max norm (± std tra i token)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Norm")
    ax.set_title("Max norm per checkpoint/layer (singola scena)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "debug_max_norm_single_scene.png", dpi=150)
    print(f"\nSalvato: {out_dir / 'debug_max_norm_single_scene.png'}")

    # ----- Plot 2: max/median, la metrica piu' indicativa di un vero outlier isolato -----
    ratios = [s["max_over_median"] for s in all_stats]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(x, ratios, alpha=0.8, color="tab:orange")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("max / median")
    ax.set_title("Rapporto max/median per checkpoint/layer\n(valori alti = possibile outlier isolato; valori vicini a 1 = distribuzione uniforme)")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_max_over_median.png", dpi=150)
    print(f"Salvato: {out_dir / 'debug_max_over_median.png'}")

    # ----- Plot 3: percentuale di token outlier (oltre 3 std sopra la media) -----
    pct_out = [s["pct_outliers_3std"] for s in all_stats]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(x, pct_out, alpha=0.8, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("% token oltre 3 std sopra la media")
    ax.set_title("Percentuale di token outlier per checkpoint/layer\n(pochi punti isolati = sink; molti punti = fenomeno diffuso)")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_pct_outliers.png", dpi=150)
    print(f"Salvato: {out_dir / 'debug_pct_outliers.png'}")


if __name__ == "__main__":
    main()