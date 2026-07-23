import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 1. Load your newly structured JSON data
json_path = "analysis_plots/nr3d_ref_scannet_anchor_val_single_batched_test_results.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Ensure the output directory exists
os.makedirs("analysis_plots", exist_ok=True)

# 2. Dynamically extract and systematically sort the ordered list of sublayers
sample_entry = next(item for item in data if "logged_norms" in item)
raw_layers = list(sample_entry["logged_norms"].keys())
layers = sorted(raw_layers, key=lambda x: (int(x.split('_')[0][1:]), x.split('_')[1]))

# Clean labels to display only the sub-layer stage on the x-axis
x_labels = [k.replace("1_Cross_Attn", "Cross").replace("2_Self_Attn", "Self").replace("3_Main_FFN", "FFN").split('_')[-1] for k in layers]
x_indices = np.arange(len(layers))

# Initialize data accumulators for sublayers
metrics_template = lambda: {layer: [] for layer in layers}
sub_data = {
    "q_norms": metrics_template(), "t_norms": metrics_template(),
    "q_feats": metrics_template(), "t_feats": metrics_template()
}

# 3. Populate lists using ALL samples
for entry in data:
    if "logged_norms" not in entry or not entry["logged_norms"]:
        continue
    norms_dict = entry["logged_norms"]
    for layer in layers:
        if layer in norms_dict:
            sub_data["q_norms"][layer].append(norms_dict[layer]["max_query_norm"])
            sub_data["t_norms"][layer].append(norms_dict[layer]["max_text_norm"])
            sub_data["q_feats"][layer].append(norms_dict[layer]["max_query_feature"])
            sub_data["t_feats"][layer].append(norms_dict[layer]["max_text_feature"])

# 4. Compute fine-grained statistics
def compute_stats(data_dict, metric_key):
    means, stds = [], []
    for layer in layers:
        values = np.array(data_dict[metric_key][layer])
        means.append(np.mean(values) if len(values) > 0 else 0.0)
        stds.append(np.std(values) if len(values) > 0 else 0.0)
    return np.array(means), np.array(stds)

q_norm_mean, q_norm_std = compute_stats(sub_data, "q_norms")
t_norm_mean, t_norm_std = compute_stats(sub_data, "t_norms")
q_feat_mean, q_feat_std = compute_stats(sub_data, "q_feats")
t_feat_mean, t_feat_std = compute_stats(sub_data, "t_feats")

# 5. Map macro-layer spans to compute horizontal average lines and center points
macro_spans = defaultdict(list)
for idx, key in enumerate(layers):
    layer_num = int(key.split('_')[0][1:])
    macro_spans[layer_num].append(idx)

# ========================================================
# 📍 CALCOLO DEI LIMITI ASSE Y GLOBALI (Forzatura di Scala)
# ========================================================
def get_global_limits(y_query, y_text):
    max_val = max(max(y_query), max(y_text))
    min_val = min(min(y_query), min(y_text))
    return min_val - (abs(min_val) * 0.05), max_val + (abs(max_val) * 0.15)

norm_ylim = get_global_limits(q_norm_mean, t_norm_mean)
feat_ylim = get_global_limits(q_feat_mean, t_feat_mean)


# 6. Core Logic for Boundaries and Layout Text
def apply_common_layout(ax, ylim_tuple, title_text, y_label):
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.2)
    
    # Vertical Boundaries
    for idx, key in enumerate(layers):
        if idx > 0 and int(layers[idx-1].split('_')[0][1:]) != int(key.split('_')[0][1:]):
            ax.axvline(x=idx - 0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            
    # Layer text placement based on ylim
    for layer_num, indices in macro_spans.items():
        mid_x = np.mean(indices)
        ax.text(mid_x, ylim_tuple[1] - (ylim_tuple[1] * 0.08), f"Layer {layer_num}", 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#444444')
                
    ax.set_ylim(ylim_tuple)

def plot_base_sublayers(ax, sub_mean, sub_std, color, label, is_background=False):
    alpha_line = 0.35 if is_background else 1.0
    alpha_fill = 0.06 if is_background else 0.15
    linewidth = 1 if is_background else 1.8
    markersize = 4 if is_background else 5
    
    ax.plot(x_indices, sub_mean, label=label, color=color, marker='o', markersize=markersize, linewidth=linewidth, alpha=alpha_line, zorder=2)
    ax.fill_between(x_indices, sub_mean - sub_std, sub_mean + sub_std, color=color, alpha=alpha_fill, zorder=1)

def plot_macro_trend(ax, sub_mean, color, label):
    macro_x, macro_y = [], []
    for layer_num, indices in macro_spans.items():
        macro_x.append(np.mean(indices))
        macro_y.append(np.mean(sub_mean[indices]))
    ax.plot(macro_x, macro_y, color=color, linewidth=3, linestyle='-', label=label, zorder=4)
    ax.scatter(macro_x, macro_y, color=color, s=70, edgecolor='black', linewidth=1, zorder=5)


# --- GENERAZIONE DEI 4 PLOT ALLINEATI ---

# 📊 1. L2 NORMS (Solo Sublayers - Linee Accese)
fig, ax = plt.subplots(figsize=(12, 6.5))
plot_base_sublayers(ax, q_norm_mean, q_norm_std, color="#1f77b4", label="Query Tokens", is_background=False)
plot_base_sublayers(ax, t_norm_mean, t_norm_std, color="#ff7f0e", label="Text Tokens", is_background=False)
apply_common_layout(ax, norm_ylim, "Avg Max L2 Norms", "L2 Norm Value")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("analysis_plots/plots/nr3d_sublayers_only_norms.png", dpi=300)
plt.close()

# 📊 2. L2 NORMS (Sublayers Background + Macro Trend Line)
fig, ax = plt.subplots(figsize=(12, 6.5))
plot_base_sublayers(ax, q_norm_mean, q_norm_std, color="#9ecae1", label="Query Tokens", is_background=True)
plot_base_sublayers(ax, t_norm_mean, t_norm_std, color="#fdbb84", label="Text Tokens", is_background=True)
plot_macro_trend(ax, q_norm_mean, color="#084594", label="Query Tokens (mean)")
plot_macro_trend(ax, t_norm_mean, color="#e6550d", label="Text Tokens (mean)")
apply_common_layout(ax, norm_ylim, "Avg Max L2 Norms", "L2 Norm Value")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("analysis_plots/plots/nr3d_combined_sublayers_macro_norms.png", dpi=300)
plt.close()


# 📊 3. ABSOLUTE FEATURES (Solo Sublayers - Linee Accese)
fig, ax = plt.subplots(figsize=(12, 6.5))
plot_base_sublayers(ax, q_feat_mean, q_feat_std, color="#1f77b4", label="Query Tokens", is_background=False)
plot_base_sublayers(ax, t_feat_mean, t_feat_std, color="#ff7f0e", label="Text Tokens", is_background=False)
apply_common_layout(ax, feat_ylim, "Avg Max Absolute Features", "Absolute Max Feature Value")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("analysis_plots/plots/nr3d_sublayers_only_features.png", dpi=300)
plt.close()

# 📊 4. ABSOLUTE FEATURES (Sublayers Background + Macro Trend Line)
fig, ax = plt.subplots(figsize=(12, 6.5))
plot_base_sublayers(ax, q_feat_mean, q_feat_std, color="#9ecae1", label="Query Tokens", is_background=True)
plot_base_sublayers(ax, t_feat_mean, t_feat_std, color="#fdbb84", label="Text Tokens", is_background=True)
plot_macro_trend(ax, q_feat_mean, color="#084594", label="Query Tokens (mean)")
plot_macro_trend(ax, t_feat_mean, color="#e6550d", label="Text Tokens (mean)")
apply_common_layout(ax, feat_ylim, "Avg Max Absolute Features", "Absolute Max Feature Value")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("analysis_plots/plots/nr3d_combined_sublayers_macro_features.png", dpi=300)
plt.close()

print("🏁 Execution complete. 4 standalone plots saved in 'analysis_plots/plots/'. Scales are strictly locked.")