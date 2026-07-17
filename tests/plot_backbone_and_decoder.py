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


def stats_from_entries_dino(entries):
    """Da una lista di entry (una per scena/chiamata), calcola mean+std del
    max, e in aggiunta il rapporto max/mediana (calcolato sull'insieme di
    tutti i valori aggregati) -- utile per distinguere un vero outlier
    isolato (rapporto alto) da un innalzamento diffuso (rapporto vicino a 1)."""
    all_maxes = []
    all_values = []
    for e in entries:
        t = e['norms'][:,5:] # removes CLS + register tokens
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
            # m, s, r = stats_from_entries_dino(by_res[res_idx])
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

    n_dino_points = pos  # indice di fine del segmento DINO, per spezzare la linea qui sotto

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

    # --- Linea spezzata in due segmenti: DINO e pixel decoder NON sono
    # collegati da un flusso diretto (il pixel decoder fonde TUTTE le
    # risoluzioni insieme, non solo l'ultima), quindi una singola
    # polilinea continua tra res3 e PD-blk0 sarebbe visivamente
    # fuorviante. I 3 blocchi del pixel decoder, invece, SONO sequenziali
    # tra loro e restano collegati.
    dino_positions, dino_y, dino_yerr = positions[:n_dino_points], y[:n_dino_points], yerr[:n_dino_points]
    pd_positions, pd_y, pd_yerr = positions[n_dino_points:], y[n_dino_points:], yerr[n_dino_points:]

    if len(dino_positions) > 0:
        ax.errorbar(dino_positions, dino_y, yerr=dino_yerr, marker="o", capsize=4, color="tab:blue")
    if len(pd_positions) > 0:
        ax.errorbar(pd_positions, pd_y, yerr=pd_yerr, marker="o", capsize=4, color="tab:blue")

    # Draw boundaries
    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # --- NEW LOGIC FOR DYNAMIC BLOCK LABELS ---
    # [Certain] We define the block edges by padding positions with the start and end
    all_bounds = [positions[0] - 0.5] + list(boundaries) + [positions[-1] + 0.5]

    # Generate the labels based on your specifications
    block_labels = []
    layer_counter = 0

    for i in range(len(all_bounds) - 1):
        if i == 0:
            block_labels.append("DINOv2")
        else:
            block_labels.append(f"layer{layer_counter}")
            layer_counter += 1

    # Get the current y-axis limits to place text at the very top safely
    ymin, ymax = ax.get_ylim()
    # [Likely] Placing text at 102% of the max y-limit keeps it neatly above the plot area
    text_y_position = ymax + (ymax - ymin) * 0.02

    # Plot each label centered in its respective block
    for i in range(len(block_labels)):
        # Calculate the midpoint of the block
        block_center = (all_bounds[i] + all_bounds[i + 1]) / 2
        
        ax.text(
            x=block_center,
            y=text_y_position,
            s=block_labels[i],
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
            fontweight="bold",
            color="dimgray"
        )
    # ------------------------------------------

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Avg Max L2 Norm")
    ax.set_title("UniVLG visual backbone: DINOv2 + pixel decoder", pad=20) # Added padding to avoid overlap with block labels
    ax.grid(alpha=0.3)

    # --- Annotazione del rapporto max/mediana sopra ogni punto ---
    # Questo e' il numero che distingue un vero outlier isolato (rapporto
    # alto) da un innalzamento diffuso della norma (rapporto vicino a 1) --
    # non era visibile nel grafico prima, solo nel riepilogo testuale.
    # Colore condizionale: sotto SINK_RATIO_THRESHOLD un colore "freddo"
    # (nessun segnale di sink), sopra soglia rosso (possibile outlier
    # isolato). La soglia e' indicativa, non un test statistico formale --
    # serve solo a guidare l'occhio verso i punti che meritano attenzione.
    SINK_RATIO_THRESHOLD = 3.0
    for xpos, yval, r in zip(positions, y, ratios):
        if r is None:
            continue
        color = "tab:red" if r >= SINK_RATIO_THRESHOLD else "teal"
        ax.annotate(
            f"×{r:.2f}",
            xy=(xpos, yval),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

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
def stats_for_token_subset(entries, start, end=None):
    """
    Come stats_from_entries, ma ristretto a un sotto-intervallo di token
    lungo la sequenza (es. solo le query, o solo il testo, dentro una
    sequenza combinata query+text). Assume che l'ordine nella sequenza sia
    fisso (es. [query_0..query_99, text_0..text_k]) e che una delle due
    dimensioni del tensore di norme sia 1 (batch), cosi' un flatten()
    preserva l'ordine lungo la sequenza indipendentemente dall'orientamento
    (N,1) o (1,N).

    end=None -> prende tutto fino alla fine del tensore per QUELLA scena
    specifica, invece di un indice fisso. Necessario per il testo, la cui
    lunghezza puo' variare frase per frase: un end fisso rischierebbe di
    tagliare token di testo validi nelle scene con frasi piu' lunghe, o di
    includere token di scene diverse in modo scorretto in quelle piu' corte.
    """
    all_maxes = []
    all_values = []
    for e in entries:
        t = get_tensor(e)
        if t is None:
            continue
        flat = t.float().flatten().numpy()
        subset = flat[start:end] if end is not None else flat[start:]
        if subset.size == 0:
            continue
        all_maxes.append(subset.max())
        all_values.append(subset)
    if len(all_maxes) == 0:
        return None, None, None
    mean_max, std_max = mean_std(np.array(all_maxes))
    all_values_flat = np.concatenate(all_values)
    median_val = np.median(all_values_flat)
    ratio = mean_max / median_val if median_val != 0 else float("nan")
    return mean_max, std_max, ratio


def build_sequential_flow_split(data, sublayer_patterns, n_layers, n_queries):
    """
    Come build_sequential_flow, ma calcola le statistiche separatamente per
    la sotto-sequenza "query" (primi n_queries token) e "text" (ultimi
    n_text token), invece di trattare l'intera sequenza combinata come
    un unico blocco.

    Ritorna due tuple (query_x, query_y, query_yerr, query_ratio, labels,
    boundaries) e (text_x, text_y, text_yerr, text_ratio, labels, boundaries)
    -- le xtick labels e i confini di layer sono identici per costruzione,
    dato che entrambe le curve condividono la stessa sequenza di sub-layer.
    """
    q_x, q_y, q_yerr, q_ratio = [], [], [], []
    t_x, t_y, t_yerr, t_ratio = [], [], [], []
    xtick_labels = []
    layer_boundaries = []
    pos = 0

    for layer_idx in range(n_layers):
        layer_has_data = False
        for label, pattern in sublayer_patterns.items():
            full_pattern = f"{pattern}.{layer_idx}."
            entries = entries_matching(data, full_pattern)

            qm, qs, qr = stats_for_token_subset(entries, 0, n_queries)
            tm, ts, tr = stats_for_token_subset(entries, n_queries, end=None)

            if qm is None and tm is None:
                continue
            layer_has_data = True

            q_x.append(pos); q_y.append(qm); q_yerr.append(qs); q_ratio.append(qr)
            t_x.append(pos); t_y.append(tm); t_yerr.append(ts); t_ratio.append(tr)
            xtick_labels.append(label)
            pos += 1
        if layer_has_data:
            layer_boundaries.append(pos - 0.5)

    if layer_boundaries:
        layer_boundaries = layer_boundaries[:-1]

    return (q_x, q_y, q_yerr, q_ratio), (t_x, t_y, t_yerr, t_ratio), xtick_labels, layer_boundaries


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


def plot_mask_decoder(data, out_dir, n_layers=8, n_queries=100):
    """
    n_queries: numero di query assunte come prime posizioni della sequenza
    combinata query+text. Il testo occupa "tutto il resto" della sequenza,
    dato che la sua lunghezza varia frase per frase.
    ASSUNZIONE: nella sequenza, le query occupano le prime n_queries
    posizioni -- verificare nel codice se l'ordine reale e' diverso.
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

    # --- HELPER FUNCTION FOR DYNAMIC LABELS ---
    def add_block_labels(ax, positions, boundaries, first_block_name="DINOv2"):
        """
        [Certain] Calculates block midpoints and adds labels at the top of the given axis.
        """
        if len(positions) == 0:
            return
            
        # Define block edges
        all_bounds = [positions[0] - 0.5] + list(boundaries) + [positions[-1] + 0.5]
        
        # Generate labels
        block_labels = []
        layer_counter = 0
        for i in range(len(all_bounds) - 1):
            if i == 0:
                block_labels.append(first_block_name)
            else:
                block_labels.append(f"layer{layer_counter}")
                layer_counter += 1
                
        # Calculate y position dynamically based on axis limits
        ymin, ymax = ax.get_ylim()
        text_y_position = ymax + (ymax - ymin) * 0.02
        
        # Draw labels
        for i in range(len(block_labels)):
            block_center = (all_bounds[i] + all_bounds[i + 1]) / 2
            ax.text(
                x=block_center,
                y=text_y_position,
                s=block_labels[i],
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=9,
                fontweight="bold",
                color="dimgray"
            )


    # --- MAIN PLOT CODE ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 11)) # [Likely] Height increased slightly to 11 to give breathing room for labels

    # --- Sottografico 1: query vs text, separati ---
    ax = axes[0]
    query_stats, text_stats, labels, boundaries = build_sequential_flow_split(
        data, query_text_patterns, n_layers=n_layers, n_queries=n_queries
    )
    q_x, q_y, q_yerr, q_ratio = query_stats
    t_x, t_y, t_yerr, t_ratio = text_stats

    if len(q_x) > 0:
        ax.errorbar(q_x, q_y, yerr=q_yerr, marker="o", capsize=3, color="tab:blue", label="query tokens")
        ax.errorbar(t_x, t_y, yerr=t_yerr, marker="^", capsize=3, color="tab:orange", label="text tokens")
        for b in boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        
        # [Certain] Apply labels dynamically to the first plot using q_x
        # Automatically triggers autoscale limits before calculating text positions
        ax.relim()
        ax.autoscale_view()
        add_block_labels(ax, q_x, boundaries, first_block_name="DINOv2")
        
        ax.set_xticks(q_x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.legend()

        # --- Annotazione del rapporto max/mediana sopra ogni punto ---
        # Stessa logica del plot della visual backbone: colore condizionale,
        # freddo (teal/dark orange) sotto soglia, rosso sopra -- per
        # segnalare a colpo d'occhio dove il rapporto e' anomalo, senza
        # dover leggere il riepilogo testuale in console.
        SINK_RATIO_THRESHOLD = 3.0
        for xpos, yval, r in zip(q_x, q_y, q_ratio):
            if r is None:
                continue
            color = "tab:red" if r >= SINK_RATIO_THRESHOLD else "teal"
            ax.annotate(
                f"×{r:.2f}", xy=(xpos, yval), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=6, color=color, fontweight="bold",
            )
        for xpos, yval, r in zip(t_x, t_y, t_ratio):
            if r is None:
                continue
            color = "tab:red" if r >= SINK_RATIO_THRESHOLD else "darkorange"
            ax.annotate(
                f"×{r:.2f}", xy=(xpos, yval), xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=6, color=color, fontweight="bold",
            )
    else:
        print("Attenzione: nessun dato per il flusso query+text.")
    ax.set_ylabel("Max norm (media ± std)")
    ax.set_title(f"Query (n={n_queries}) vs Text (lunghezza variabile) tokens -- separati, flusso sequenziale", pad=22)
    ax.grid(alpha=0.3)

    # --- Sottografico 2: vis_output ---
    ax = axes[1]
    x, y, yerr, ratios, vlabels, vboundaries = build_sequential_flow(data, vis_output_patterns, n_layers=n_layers)
    if len(x) > 0:
        ax.errorbar(x, y, yerr=yerr, marker="s", capsize=3, color="tab:red")
        for b in vboundaries:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        
        # [Certain] Apply labels dynamically to the second plot using x and vboundaries
        ax.relim()
        ax.autoscale_view()
        add_block_labels(ax, x, vboundaries, first_block_name="DINOv2")
        
        ax.set_xticks(x)
        ax.set_xticklabels(vlabels, rotation=60, ha="right", fontsize=7)

        # --- Annotazione del rapporto max/mediana sopra ogni punto ---
        SINK_RATIO_THRESHOLD = 3.0
        for xpos, yval, r in zip(x, y, ratios):
            if r is None:
                continue
            color = "tab:red" if r >= SINK_RATIO_THRESHOLD else "teal"
            ax.annotate(
                f"×{r:.2f}", xy=(xpos, yval), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=6, color=color, fontweight="bold",
            )
    else:
        print("Attenzione: nessun dato per il flusso vis_output.")
    ax.set_ylabel("Avg Max L2 Norm")
    ax.set_title("Visual tokens (mask decoder)", pad=22)
    ax.grid(alpha=0.3)

    fig.suptitle("Mask decoder: query vs text vs vis tokens", fontsize=13, y=0.98)
    fig.tight_layout()
    fname = out_dir / "mask_decoder_flow.png"
    fig.savefig(fname, dpi=150)
    print(f"Salvato: {fname}")

    print("\nRiepilogo mask decoder:")
    print("  Query tokens:")
    for lbl, m, s, r in zip(labels, q_y, q_yerr, q_ratio):
        if m is not None:
            print(f"    {lbl}: mean_max={m:.3f}  std={s:.3f}  max/median={r:.2f}")
    print("  Text tokens:")
    for lbl, m, s, r in zip(labels, t_y, t_yerr, t_ratio):
        if m is not None:
            print(f"    {lbl}: mean_max={m:.3f}  std={s:.3f}  max/median={r:.2f}")
    print("  Vis output tokens:")
    for lbl, m, s, r in zip(vlabels, y, yerr, ratios):
        if m is not None:
            print(f"    {lbl}: mean_max={m:.3f}  std={s:.3f}  max/median={r:.2f}")


# ---------------------------------------------------------------------
# Plot 3: text encoder (Jina-BERT-v2), flusso sequenziale unico
# ---------------------------------------------------------------------
def plot_text_encoder(data, out_dir, n_layers=12,
                       base_prefix="lang_encoder.text_encoder.text_model.transformer"):
    """
    Flusso sequenziale: emb_ln (LayerNorm subito dopo l'embedding, prima
    di qualsiasi layer transformer) -> layer0.norm1 -> layer0.norm2 ->
    layer1.norm1 -> ... -> layer{n_layers-1}.norm2.

    norm1/norm2 sono tipicamente le LayerNorm pre-attention e pre-FFN (o
    post-attention/post-FFN, a seconda della convenzione pre-norm vs
    post-norm del modello) di ciascun layer transformer -- qui trattate
    come due punti distinti nella sequenza, analogamente a come il pixel
    decoder viene espanso in cross/ffn/gate.
    """
    positions, y, yerr, ratios, labels = [], [], [], [], []
    boundaries = []
    pos = 0

    # --- emb_ln: punto zero della sequenza, prima del primo layer ---
    emb_ln_pattern = f"{base_prefix}.emb_ln"
    entries = entries_matching(data, emb_ln_pattern)
    m, s, r = stats_from_entries(entries)
    if m is not None:
        positions.append(pos)
        y.append(m)
        yerr.append(s)
        ratios.append(r)
        labels.append("emb_ln")
        pos += 1
    else:
        print(f"Attenzione: chiave '{emb_ln_pattern}' non trovata, salto emb_ln.")

    if pos > 0:
        boundaries.append(pos - 0.5)

    # --- layer0..layer{n_layers-1}, ciascuno espanso in norm1 / norm2 ---
    for layer_idx in range(n_layers):
        layer_has_data = False
        for sub_label in ("norm1", "norm2"):
            full_pattern = f"{base_prefix}.encoder.layers.{layer_idx}.{sub_label}"
            entries = entries_matching(data, full_pattern)
            m, s, r = stats_from_entries(entries)
            if m is None:
                continue
            layer_has_data = True
            positions.append(pos)
            y.append(m)
            yerr.append(s)
            ratios.append(r)
            labels.append(f"L{layer_idx}\n{sub_label}")
            pos += 1
        if layer_has_data:
            boundaries.append(pos - 0.5)

    if boundaries:
        boundaries = boundaries[:-1]  # l'ultimo confine e' la fine del grafico, non serve

    if len(positions) == 0:
        print("Nessun dato per il plot del text encoder.")
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.errorbar(positions, y, yerr=yerr, marker="o", capsize=4, color="tab:blue")

    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel("Avg Max L2 Norm")
    ax.set_title("Text encoder (Jina-BERT-v2): emb_ln + 12 layer (norm1/norm2)", pad=20)
    ax.grid(alpha=0.3)

    # --- Annotazione del rapporto max/mediana sopra ogni punto ---
    SINK_RATIO_THRESHOLD = 3.0
    for xpos, yval, r in zip(positions, y, ratios):
        if r is None:
            continue
        color = "tab:red" if r >= SINK_RATIO_THRESHOLD else "teal"
        ax.annotate(
            f"×{r:.2f}",
            xy=(xpos, yval),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color=color,
            fontweight="bold",
        )

    fig.tight_layout()
    fname = out_dir / "text_encoder_flow.png"
    fig.savefig(fname, dpi=150)
    print(f"Salvato: {fname}")

    print("\nRiepilogo text encoder:")
    for lbl, m, s, r in zip(labels, y, yerr, ratios):
        print(f"  {lbl.replace(chr(10), ' ')}: mean_max={m:.3f}  std={s:.3f}  max/median={r:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norms_file", type=str, default="tests/scanrefer_scannet_anchor_val_single_batched_hook_norms.pt")
    parser.add_argument("--out_dir", type=str, default="./plots")
    parser.add_argument("--n_mask_decoder_layers", type=int, default=8)
    parser.add_argument("--n_pixel_decoder_blocks", type=int, default=3)
    parser.add_argument("--n_queries", type=int, default=100,
                         help="Numero di query nella sequenza combinata query+text del mask decoder "
                              "(assunte come prime n_queries posizioni; il testo occupa il resto).")
    parser.add_argument("--text_norms_file", type=str, default=None,
                         help="File .pt separato con le norme del text encoder (se salvato a parte, "
                              "come nel caso di un hook batch dedicato). Se omesso, questo plot viene saltato.")
    parser.add_argument("--n_text_encoder_layers", type=int, default=12)
    parser.add_argument("--skip_visual_backbone", action="store_true",
                         help="Salta il plot della visual backbone (DINO + pixel decoder).")
    parser.add_argument("--skip_mask_decoder", action="store_true",
                         help="Salta il plot del mask decoder (query/text/vis output).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carica il file principale solo se serve almeno uno dei due plot che lo usano
    needs_main_file = not args.skip_visual_backbone or not args.skip_mask_decoder
    if needs_main_file:
        data = torch.load(args.norms_file, map_location="cpu")
        print(f"Loaded norms data from {args.norms_file}, {len(data)} modules found.")

        if not args.skip_visual_backbone:
            plot_visual_backbone(data, out_dir, n_pixel_decoder_blocks=args.n_pixel_decoder_blocks)
            print()
        else:
            print("Salto il plot della visual backbone (--skip_visual_backbone).")

        if not args.skip_mask_decoder:
            plot_mask_decoder(data, out_dir, n_layers=args.n_mask_decoder_layers,
                               n_queries=args.n_queries)
        else:
            print("Salto il plot del mask decoder (--skip_mask_decoder).")
    else:
        print("Salto il caricamento del file principale (entrambi i plot che lo usano sono disattivati).")

    if args.text_norms_file is not None:
        print()
        text_data = torch.load(args.text_norms_file, map_location="cpu")
        print(f"Loaded text encoder norms data from {args.text_norms_file}, {len(text_data)} modules found.")
        plot_text_encoder(text_data, out_dir, n_layers=args.n_text_encoder_layers)


if __name__ == "__main__":
    main()