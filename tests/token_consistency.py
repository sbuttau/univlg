"""
Verifica se il token con norma massima resta lo stesso (stessa posizione
nella sequenza di patch) attraverso le 4 risoluzioni/profondita' di DINOv2,
per ciascuna vista (elemento del batch B*V).

Dato che tutte e 4 le risoluzioni hanno lo stesso numero di token (773,
non cambia la risoluzione spaziale, solo la profondita' da cui vengono
estratte), il confronto e' diretto: stesso indice di vista, stesso indice
di token possibile, attraverso i 4 layer.

Interpretazione:
- Se lo stesso indice di token e' il massimo in piu' risoluzioni per la
  stessa vista -> il sink e' persistente, la stessa posizione domina lungo
  la rete (solo "diluita" in termini relativi dalla crescita generale
  della norma, non eliminata).
- Se l'indice cambia ad ogni risoluzione -> il fenomeno e' locale a quel
  layer specifico, non un sink che attraversa la rete.

Uso:
    python check_token_consistency.py /path/to/filtered_norms.pt
"""

import argparse
import torch
import numpy as np
from collections import Counter

from norm_hooks import split_post_dino_by_resolution


def get_tensor(entry):
    if isinstance(entry, dict):
        return entry["norms"]
    return entry


def argmax_per_view(tensor):
    """
    tensor: shape (B*V, N) -- norme per-token, un layer/risoluzione.
    Ritorna: array di lunghezza B*V con l'indice del token (0..N-1) di
    norma massima, per ciascuna vista.
    """
    t = tensor.float()
    return t.argmax(dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("norms_file", type=str)
    parser.add_argument(
        "--dino_key", type=str,
        default="visual_backbone.backbone.dinov2.inner.norm",
    )
    args = parser.parse_args()

    data = torch.load(args.norms_file, map_location="cpu")

    if args.dino_key not in data:
        print(f"Chiave '{args.dino_key}' non trovata nel file.")
        print("Chiavi disponibili:", list(data.keys())[:20])
        return

    by_res = split_post_dino_by_resolution(data[args.dino_key], n_resolutions=4)

    # Per ora assumiamo una sola scena (una entry per risoluzione);
    # se ce ne sono di piu', analizziamo la prima e stampiamo un avviso.
    n_scenes = len(by_res[0])
    if n_scenes > 1:
        print(f"Attenzione: trovate {n_scenes} scene. Analizzo solo la prima; "
              f"per un'analisi completa su piu' scene, estendi il loop sotto.")

    argmax_per_resolution = {}
    for res_idx in sorted(by_res.keys()):
        entry = by_res[res_idx][0]  # prima scena
        tensor = get_tensor(entry)
        argmax_per_resolution[res_idx] = argmax_per_view(tensor)

    n_views = len(argmax_per_resolution[0])
    n_resolutions = len(argmax_per_resolution)

    print(f"\nConfronto argmax (indice del token con norma massima) per vista, "
          f"attraverso {n_resolutions} risoluzioni, su {n_views} viste:\n")

    header = "vista  " + "  ".join(f"res{r}" for r in sorted(argmax_per_resolution.keys()))
    print(header)
    print("-" * len(header))

    same_across_all = 0
    for view_idx in range(n_views):
        row = [argmax_per_resolution[r][view_idx] for r in sorted(argmax_per_resolution.keys())]
        is_same = len(set(row)) == 1
        if is_same:
            same_across_all += 1
        marker = "  <-- STESSO TOKEN in tutte le risoluzioni" if is_same else ""
        print(f"{view_idx:5d}  " + "  ".join(f"{v:4d}" for v in row) + marker)

    print(f"\n{same_across_all}/{n_views} viste hanno lo STESSO indice di token "
          f"come massimo in tutte le {n_resolutions} risoluzioni "
          f"({100*same_across_all/n_views:.1f}%).")

    # Riepilogo aggiuntivo: quali indici di token ricorrono piu' spesso come
    # massimo, per ciascuna risoluzione separatamente
    print("\nToken piu' frequentemente massimo, per risoluzione:")
    for res_idx in sorted(argmax_per_resolution.keys()):
        counts = Counter(argmax_per_resolution[res_idx].tolist())
        most_common = counts.most_common(3)
        print(f"  res{res_idx}: {most_common}  (indice_token: n_volte_e'_il_massimo)")


if __name__ == "__main__":
    main()