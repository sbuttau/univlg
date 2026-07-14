"""
Cattura le norme (e altre statistiche) del residual stream PRIMA di ogni
nn.LayerNorm del modello, usando forward pre-hook -- senza modificare
il codice del modello.

Come funziona:
- register_forward_pre_hook cattura l'INPUT di un modulo, prima che venga eseguito.
- Per un nn.LayerNorm, l'input e' esattamente il residual stream grezzo,
  prima della normalizzazione -- quello che ti serve per cercare sink.
- Registrando l'hook su OGNI LayerNorm del modello (o su un sottoinsieme
  filtrato per nome), ottieni automaticamente tutti i checkpoint che ti
  interessano, senza toccare nessun forward().

Uso tipico:
    hook_manager = NormHookManager(model, name_filter="cross_view_attn")
    hook_manager.attach()

    # ... lancia l'eval normalmente, senza modifiche al loop ...

    hook_manager.detach()  # importante: rimuove gli hook a fine eval
    torch.save(hook_manager.get_data(), "norms.pt")
"""

import torch
import torch.nn as nn
from collections import defaultdict


def interpret_module_name(name):
    """
    Data il nome completo di un modulo LayerNorm (da named_modules()),
    restituisce una descrizione leggibile di COSA rappresenta il suo input
    (cioe' il punto della pipeline catturato dall'hook), basandosi sulla
    struttura del forward che abbiamo ispezionato nel codice di UniVLG.

    Questa mappatura va aggiornata se si aggiungono hook su parti del
    modello diverse da CrossViewPAnet / DINOv2.
    """
    if "cross_view_attention_layers" in name and name.endswith("norm"):
        return "post-attention (cross-view), pre-norm interno"
    if "ffn_layers" in name and name.endswith("norm"):
        return "post-FFN (cross-view), pre-norm interno"
    if "layer_norms" in name:
        return "output finale del layer (post-FFN+attn), pre-norm esterno/gate"
    if "dinov2" in name and "norm1" in name:
        return "DINOv2: pre-attention (norm1, pre-norm ViT)"
    if "dinov2" in name and "norm2" in name:
        return "DINOv2: pre-MLP (norm2, pre-norm ViT)"
    return "sconosciuto: aggiornare interpret_module_name()"


def print_summary_for_dict(data):
    """
    Stampa un riepilogo leggibile per un dizionario {module_name: entries},
    sia esso self.data di un NormHookManager, sia un sottoinsieme filtrato
    restituito da NormHookManager.select().

    Gestisce sia entries in formato "con metadati" (dict con 'norms',
    'input_shape', ecc. -- prodotte dai LayerNorm hook) sia entries in
    formato "solo tensore" (prodotte da attach_dino_hook, che salva
    direttamente il tensore di norme senza metadati aggiuntivi).
    """
    print(f"\n{'modulo':70s} {'n_call':>8s} {'in_shape':>18s} {'ch_dim':>7s} {'norm_shape':>16s} {'ambig':>6s}  interpretazione")
    print("-" * 165)
    for name, entries in data.items():
        if len(entries) == 0:
            continue
        e = entries[0]  # prima chiamata come esempio
        meaning = interpret_module_name(name)

        if isinstance(e, dict):
            in_shape = str(e["input_shape"])
            norm_shape = str(tuple(e["norms"].shape))
            ch_dim = str(e["channel_dim_used"])
            ambig = "SI" if e["ambiguous"] else ""
        else:
            # entry "solo tensore" (es. da attach_dino_hook)
            in_shape = "n/d"
            norm_shape = str(tuple(e.shape))
            ch_dim = "n/d"
            ambig = ""

        print(
            f"{name:70s} {len(entries):8d} {in_shape:>18s} "
            f"{ch_dim:>7s} {norm_shape:>16s} {ambig:>6s}  {meaning}"
        )
def split_post_dino_by_resolution(entries, n_resolutions=4):
    per_resolution = {r: [] for r in range(n_resolutions)}
    for i, entry in enumerate(entries):
        res_idx = i % n_resolutions
        per_resolution[res_idx].append(entry)
    return per_resolution

class NormHookManager:
    def __init__(self, model, name_filter=None):
        """
        Args:
            model: il modello (o sottomodulo) su cui cercare i LayerNorm.
            name_filter: se fornito (stringa), registra hook solo sui
                LayerNorm il cui nome completo (es. "visual_backbone.
                pixel_decoder.cross_view_attn.2.layer_norms.0") contiene
                questa sottostringa. Utile per limitarsi a una parte
                specifica del modello.
        """
        self.model = model
        self.name_filter = name_filter
        self.handles = []
        # data[module_name] = lista di tensori di norme, uno per ogni
        # chiamata (cioe' uno per ogni forward pass / scena, nell'ordine
        # in cui il dataloader le processa)
        self.data = defaultdict(list)

    def _make_hook(self, module_name, module):
        # Il canale su cui il LayerNorm normalizza non e' sempre l'ultima
        # dimensione del tensore: dipende da come il codice a monte ha
        # fatto reshape/permute prima di chiamare il LayerNorm. Deduciamo
        # la dimensione corretta dal suo normalized_shape, invece di
        # assumere sempre dim=-1.
        expected_c = module.normalized_shape[-1]

        def hook(m, inputs):
            with torch.no_grad():
                x = inputs[0]
                input_shape = tuple(x.shape)
                ambiguous = False
                candidate_dims = [d for d in range(x.dim()) if x.shape[d] == expected_c]

                if x.shape[-1] == expected_c and len(candidate_dims) == 1:
                    # Caso standard e NON ambiguo: canali in ultima posizione,
                    # e nessun'altra dimensione ha la stessa size.
                    channel_dim = x.dim() - 1
                elif len(candidate_dims) == 0:
                    print(
                        f"[NormHookManager] Attenzione: impossibile determinare "
                        f"la dimensione canale per '{module_name}' "
                        f"(shape={input_shape}, expected_c={expected_c}). Salto."
                    )
                    return
                else:
                    if len(candidate_dims) > 1:
                        ambiguous = True
                    channel_dim = next((d for d in candidate_dims if d != 0), candidate_dims[0])

                norms = x.norm(dim=channel_dim)

                # Salviamo, insieme alla norma, i metadati per poter verificare
                # a posteriori se la scelta e' stata ambigua o no, senza dover
                # ricalcolare tutto a mano.
                self.data[module_name].append({
                    "norms": norms.detach().cpu(),
                    "input_shape": input_shape,
                    "channel_dim_used": channel_dim,
                    "expected_c": expected_c,
                    "ambiguous": ambiguous,
                })

        return hook

    def attach_dino_hook(self, dino_module_path="visual_backbone.backbone.dinov2", key_prefix="post_dino"):
        """
        Registra un forward hook (non pre-hook) sul wrapper DINOv2, per
        catturare il suo output -- la normalizzazione di DINO avviene
        internamente dentro forward_intermediates() e non e' esposta come
        un nn.LayerNorm separato, quindi qui serve un hook dedicato invece
        del meccanismo generico basato su LayerNorm.

        L'output atteso e' una lista di tensori (uno per risoluzione/layer),
        shape (B*V, C, H, W) per via di output_fmt='NCHW' -- i canali sono
        quindi in posizione 1, non nell'ultima dimensione.

        I dati vengono salvati in self.data con chiavi "{key_prefix}/L{i}",
        cosi' si integrano con tutti gli altri metodi (select, print_summary,
        get_data, ecc.) esattamente come i LayerNorm hook.
        """
        dino_module = self.model
        for part in dino_module_path.split("."):
            dino_module = getattr(dino_module, part)

        def hook(module, inputs, output):
            with torch.no_grad():
                for i, feat in enumerate(output):
                    norms = feat.norm(dim=1).detach().cpu()  # canali in posizione 1 (NCHW)
                    self.data[f"{key_prefix}/L{i}"].append(norms)

        handle = dino_module.register_forward_hook(hook)
        self.handles.append(handle)
        print(f"Hook DINO registrato su '{dino_module_path}'.")

    def attach(self):
        n_hooked = 0
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.LayerNorm):
                continue
            if self.name_filter is not None and self.name_filter not in name:
                continue
            handle = module.register_forward_pre_hook(self._make_hook(name, module))
            self.handles.append(handle)
            n_hooked += 1
        print(f"Hook registrati su {n_hooked} moduli LayerNorm.")

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_data(self):
        """Restituisce un dict normale (non defaultdict) pronto per torch.save,
        con TUTTI i metadati (norme + shape + flag ambiguita')."""
        return dict(self.data)

    def get_norms_only(self):
        """Restituisce solo i tensori di norma, senza metadati -- comodo per
        alimentare direttamente lo script di plotting (plot_norms_debug.py),
        che si aspetta liste di tensori, non liste di dict."""
        return {
            name: [entry["norms"] for entry in entries]
            for name, entries in self.data.items()
        }

    def print_summary(self):
        print_summary_for_dict(self.data)

    def select(self, include_patterns, exclude_patterns=None):
        """
        Restituisce un nuovo dict {module_name: entries} contenente solo i
        moduli il cui nome contiene ALMENO UNO dei substring in
        include_patterns, e NESSUNO di quelli in exclude_patterns.
        Non modifica self.data.
        """
        exclude_patterns = exclude_patterns or []
        selected = {}
        for name, entries in self.data.items():
            if any(p in name for p in include_patterns) and not any(p in name for p in exclude_patterns):
                selected[name] = entries
        return selected

    def print_ambiguous(self):
        """Stampa solo i moduli per cui la scelta della dimensione canale
        e' stata ambigua (piu' dimensioni con la stessa size), da
        verificare manualmente con priorita'."""
        print("\nModuli con dimensione canale AMBIGUA (verificare manualmente):")
        found = False
        for name, entries in self.data.items():
            if len(entries) == 0:
                continue
            e = entries[0]
            if e["ambiguous"]:
                found = True
                print(
                    f"  {name}: input_shape={e['input_shape']}, "
                    f"expected_c={e['expected_c']}, channel_dim_used={e['channel_dim_used']}"
                )
        if not found:
            print("  Nessuno -- tutte le scelte sono state univoche.")


# ---------------------------------------------------------------------
# Esempio di utilizzo end-to-end
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)