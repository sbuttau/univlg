import argparse
import torch
import matplotlib.pyplot as plt
from collections import Counter


def get_norm_tensor(entry):
    return entry["norms"] if isinstance(entry, dict) else entry


def seq_length_from_mask(mask):
    # mask: True = padding, False = real token -> count the False
    return (~mask).sum(dim=-1)


def argmax_per_scene(data, module_name):
    entries = data[module_name]
    indices = []
    for entry in entries:
        norms = get_norm_tensor(entry)
        flat = norms.reshape(-1, norms.shape[-1])
        idx = flat.argmax(dim=-1)
        indices.append(idx)
    return indices


def check_second_peak_is_sep(data, module_name, seq_lengths, exclude_idx=0):
    entries = data[module_name]
    match_count, total = 0, 0
    for i, entry in enumerate(entries):
        norms = get_norm_tensor(entry)
        seq = (norms[0] if norms.dim() > 1 else norms).clone()
        seq[exclude_idx] = -float("inf")
        second_peak_idx = seq.argmax().item()

        L = seq_lengths[i]
        L = L[0].item() if L.dim() > 0 else L.item()
        expected_sep_idx = L - 1

        match_count += (second_peak_idx == expected_sep_idx)
        total += 1
    pct = 100 * match_count / total
    print(f"  second peak == SEP position: {match_count}/{total} ({pct:.1f}%)")
    return match_count / total


def plot_activation_profile(data, module_names, scene_idx=0, save_path=None):
    # ... (invariato)
    ...


def plot_overlay_with_sep(data, module_name, masks, n_scenes=30, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(min(n_scenes, len(data[module_name]))):
        norms = get_norm_tensor(data[module_name][i])
        seq = (norms[0] if norms.dim() > 1 else norms).numpy()
        ax.plot(range(len(seq)), seq, linewidth=0.7, alpha=0.3, color="tab:blue")

        L = seq_length_from_mask(masks[i])
        L = L[0].item() if L.dim() > 0 else L.item()
        ax.axvline(L - 1, color="tab:red", alpha=0.15, linewidth=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("token position")
    ax.set_ylabel("norm (log scale)")
    ax.set_title(f"{module_name} — CLS/SEP peaks over {n_scenes} scenes")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def main(data_path, mask_path, suspect_layers):
    data = torch.load(data_path)
    masks = torch.load(mask_path)

    for name in suspect_layers:
        assert name in data, f"'{name}' not found."
    assert len(masks) == len(data[suspect_layers[0]]), "mismatch tra mask e norme!"

    seq_lengths = [seq_length_from_mask(m) for m in masks]

    # ... blocco argmax/coerenza invariato ...

    print("\nSEP check per layer:")
    for name in suspect_layers:
        print(name)
        check_second_peak_is_sep(data, name, seq_lengths)

    plot_activation_profile(data, suspect_layers, scene_idx=0, save_path="sink_profile_scene0.png")
    plot_overlay_with_sep(data, suspect_layers[0], masks, n_scenes=30, save_path="sink_profile_overlay_sep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="norms.pt")
    parser.add_argument("--masks", type=str, default="attention_masks.pt")
    args = parser.parse_args()

    suspect_layers = [
        "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.0.norm2",
        "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.7.norm2",
        "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.8.norm2",
        "mask_decoder.lang_encoder.text_encoder.text_model.transformer.encoder.layers.9.norm2",
    ]
    main(args.data, args.masks, suspect_layers)