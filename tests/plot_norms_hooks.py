"""
Due plot, entrambi come flusso sequenziale unico (una sola curva, sub-layer
sull'asse x, linee tratteggiate a marcare i confini tra layer/blocchi):

1) Visual backbone: DINOv2 (4 risoluzioni) -> pixel decoder (3 blocchi,
   ciascuno espanso in cross / ffn / post-ffn(gate)).

2) Mask decoder: due sottografici separati (query+text vs vis_output,
   perche' sono due tensori/flussi concettualmente diversi), ciascuno come
   flusso sequenziale con i sub-layer (cross-attn/self-attn/ffn, oppure
   vis cross-attn/vis ffn) sull'asse x.

Uso:
    python plot_backbone_and_decoder.py /path/to/filtered_norms.pt --out_dir ./plots
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from norm_hooks import split_post_dino_by_resolution


def get_tensor(entry):
    """entry puo' essere un dict con metadati (formato hook) o un tensore diretto."""
    if isinstance(entry, dict):
        return entry["norms"]
    return entry


def max_per_call(tensor):
    """
    Max sui token/voxel per ciascun elemento di batch/vista, gestendo
    correttamente sia tensori (B, N) che (N, B) -- alcuni checkpoint hanno
    l'ordine invertito per via di permute() usati per nn.MultiheadAttention
    (formato sequence-first).

    Euristica: la dimensione "batch/vista" e' quella piu' piccola tra le
    prime due; quella "token/voxel" (su cui va fatto il max) e' l'altra,
    piu' grande. Per tensori con piu' di 2 dimensioni, la dimensione batch
    resta la prima e si fa il max su tutto il resto (comportamento
    invariato per shape tipo (B*V, H, W)).
    """
    t = tensor.float()

    if t.dim() == 2:
        d0, d1 = t.shape
        if d1 >= d0:
            # shape (B, N): B piccolo, N grande -> max sulla dimensione 1 (corretto, comportamento originale)
            return t.max(dim=1).values.numpy()
        else:
            # shape (N, B) con N > B: i token/voxel sono sulla dimensione 0,
            # il "batch" (spesso 1) sulla dimensione 1 -> max sulla dimensione 0
            return t.max(dim=0).values.numpy()
    else:
        # 3+ dimensioni: assumiamo la prima sia il batch/vista, max su tutto il resto
        flat = t.reshape(t.shape[0], -1)
        return flat.max(dim=1).values.numpy()


def mean_std(arr):
    if len(arr) == 0:
        return None, None
    arr = np.asarray(arr)
    return arr.mean(), (arr.std() if len(arr) > 1 else 0.0)


def stats_from_entries(entries):
    """Da una lista di entry (una per scena/chiamata), calcola mean+std del
    max, e in aggiunta il rapporto max/mediana (calcolato sull'insieme di
    tutti i valori aggregati) -- utile per distinguere un vero outlier
    isolato (rapporto alto) da un innalzamento diffuso (rapporto vicino a 1)."""
    all_maxes = []
    all_values = []
    for e in entries:
        t = get_tensor(e)
        if t is None:
            continue
        all_maxes.append(max_per_call(t))
        all_values.append(t.float().flatten().numpy())
    if len(all_maxes) == 0:
        return None, None, None
    mean_max, std_max = mean_std(np.concatenate(all_maxes))
    all_values_flat = np.concatenate(all_values)
    median_val = np.median(all_values_flat)
    ratio = mean_max / median_val if median_val != 0 else float("nan")
    return mean_max, std_max, ratio


def entries_matching(data, pattern):
    """Tutte le entry di tutti i moduli il cui nome contiene 'pattern'."""
    out = []
    for name, entries in data.items():
        if pattern in name:
            out.extend(entries)
    return out


# ---------------------------------------------------------------------
# Plot 1: visual backbone come flusso sequenziale unico
# ---------------------------------------------------------------------
def plot_visual_backbone(data, out_dir, n_pixel_decoder_blocks=3):
    dino_key = "visual_backbone.backbone.dinov2.inner.norm"

    positions, y, yerr, ratios, labels = [], [], [], [], []
    boundaries = []
    pos = 0

    # --- DINOv2: 4 risoluzioni, un punto ciascuna (nessun sub-layer) ---
    if dino_key in data:
        by_res = split_post_dino_by_resolution(data[dino_key], n_resolutions=4)
        for res_idx in sorted(by_res.keys()):
            m, s, r = stats_from_entries(by_res[res_idx])
            if m is None:
                continue
            positions.append(pos)
            y.append(m)
            yerr.append(s)
            ratios.append(r)
            labels.append(f"DINO\nres{res_idx}")
            pos += 1
    else:
        print(f"Attenzione: chiave '{dino_key}' non trovata, salto la parte DINO.")

    if pos > 0:
        boundaries.append(pos - 0.5)

    # --- Pixel decoder: n blocchi, ciascuno espanso in cross / ffn / post-ffn(gate) ---
    sublayer_patterns = [
        ("cross", "cross_view_attention_layers"),
        ("ffn", "ffn_layers"),
        ("post-ffn\n(gate)", "layer_norms"),
    ]

    for block_idx in range(n_pixel_decoder_blocks):
        block_has_data = False
        for label, subpattern in sublayer_patterns:
            full_pattern = f"pixel_decoder.cross_view_attn.{block_idx}.{subpattern}"
            entries = entries_matching(data, full_pattern)
            m, s, r = stats_from_entries(entries)
            if m is None:
                continue
            block_has_data = True
            positions.append(pos)
            y.append(m)
            yerr.append(s)
            ratios.append(r)
            labels.append(f"PD blk{block_idx}\n{label}")
            pos += 1
        if block_has_data:
            boundaries.append(pos - 0.5)

    if boundaries:
        boundaries = boundaries[:-1]  # l'ultimo confine e' la fine del grafico, non serve

    if len(positions) == 0:
        print("Nessun dato per il plot della visual backbone.")
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.errorbar(positions, y, yerr=yerr, marker="o", capsize=4, color="tab:blue")
    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Max norm (media ± std tra le scene)")
    ax.set_title("Visual backbone: flusso sequenziale (DINOv2 -> pixel decoder)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = out_dir / "visual_backbone_flow.png"
    fig.savefig(fname, dpi=150)
    print(f"Salvato: {fname}")

    print("\nRiepilogo visual backbone:")
    for lbl, m, s, r in zip(labels, y, yerr, ratios):
        print(f"  {lbl.replace(chr(10), ' ')}: mean_max={m:.3f}  std={s:.3f}  max/median={r:.2f}")


# ---------------------------------------------------------------------
# Plot 2: mask decoder, due flussi sequenziali separati
# ---------------------------------------------------------------------
def build_sequential_flow(data, sublayer_patterns, n_layers=8):
    """
    Costruisce un'unica curva sequenziale: per ciascun layer (0..n_layers-1),
    per ciascun sublayer (nell'ordine di sublayer_patterns), calcola
    mean_max/std. Ritorna x_positions, y, yerr, xtick_labels, layer_boundaries.
    """
    x_positions, y, yerr, ratios, xtick_labels = [], [], [], [], []
    layer_boundaries = []
    pos = 0

    for layer_idx in range(n_layers):
        layer_has_data = False
        for label, pattern in sublayer_patterns.items():
            full_pattern = f"{pattern}.{layer_idx}."
            entries = entries_matching(data, full_pattern)
            m, s, r = stats_from_entries(entries)
            if m is None:
                continue
            layer_has_data = True
            x_positions.append(pos)
            y.append(m)
            yerr.append(s)
            ratios.append(r)
            xtick_labels.append(label)
            pos += 1
        if layer_has_data:
            layer_boundaries.append(pos - 0.5)

    if layer_boundaries:
        layer_boundaries = layer_boundaries[:-1]

    return x_positions, y, yerr, ratios, xtick_labels, layer_boundaries


def plot_mask_decoder(data, out_dir, n_layers=8):
    query_text_patterns = {
        "cross-attn": "transformer_cross_attention_layers",
        "self-attn": "transformer_self_attention_layers",
        "ffn": "transformer_ffn_layers",
    }
    vis_output_patterns = {
        "vis cross-attn": "vis_output_cross_attn",
        "vis ffn": "vis_output_ffn",
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    x, y, yerr, ratios, labels, boundaries = build_sequential_flow(data, query_text_patterns, n_layers=n_layers)
    if len(x) > 0:
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, color="tab:blue")
        for b in boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    else:
        print("Attenzione: nessun dato per il flusso query+text.")
    ax.set_ylabel("Max norm (media ± std)")
    ax.set_title("Query + Text tokens (seq. len ~124) -- flusso sequenziale, linee tratteggiate = confine tra layer")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x, y, yerr, ratios, labels, boundaries = build_sequential_flow(data, vis_output_patterns, n_layers=n_layers)
    if len(x) > 0:
        ax.errorbar(x, y, yerr=yerr, marker="s", capsize=3, color="tab:red")
        for b in boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    else:
        print("Attenzione: nessun dato per il flusso vis_output.")
    ax.set_ylabel("Max norm (media ± std)")
    ax.set_title("Vis output tokens (seq. len ~2422) -- flusso sequenziale, linee tratteggiate = confine tra layer")
    ax.grid(alpha=0.3)

    fig.suptitle("Mask decoder: due flussi distinti (query+text vs. vis tokens)", fontsize=13)
    fig.tight_layout()

    fname = out_dir / "mask_decoder_flow.png"
    fig.savefig(fname, dpi=150)
    print(f"Salvato: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("norms_file", type=str)
    parser.add_argument("--out_dir", type=str, default="./plots")
    parser.add_argument("--n_mask_decoder_layers", type=int, default=8)
    parser.add_argument("--n_pixel_decoder_blocks", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.norms_file, map_location="cpu")

    plot_visual_backbone(data, out_dir, n_pixel_decoder_blocks=args.n_pixel_decoder_blocks)
    print()
    plot_mask_decoder(data, out_dir, n_layers=args.n_mask_decoder_layers)


if __name__ == "__main__":
    main()