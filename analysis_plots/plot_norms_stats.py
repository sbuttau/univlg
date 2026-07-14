import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your newly structured JSON data
json_path = "analysis_plots/scanrefer_scannet_val_scene0046_debug_batched_test_results.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Ensure the output directory exists
os.makedirs("analysis_plots", exist_ok=True)

# 2. Dynamically extract and systematically sort the ordered list of layers
sample_entry = next(item for item in data if "logged_norms" in item)
raw_layers = list(sample_entry["logged_norms"].keys())
layers = sorted(raw_layers, key=lambda x: (int(x.split('_')[0][1:]), x.split('_')[1]))

# Clean labels to display only the sub-layer stage on the x-axis
x_labels = [k.replace("1_Cross_Attn", "Cross").replace("2_Self_Attn", "Self").replace("3_Main_FFN", "FFN").split('_')[-1] for k in layers]
x_indices = np.arange(len(layers))

# Initialize unified data accumulators
metrics_template = lambda: {layer: [] for layer in layers}
all_data = {
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
            all_data["q_norms"][layer].append(norms_dict[layer]["max_query_norm"])
            all_data["t_norms"][layer].append(norms_dict[layer]["max_text_norm"])
            all_data["q_feats"][layer].append(norms_dict[layer]["max_query_feature"])
            all_data["t_feats"][layer].append(norms_dict[layer]["max_text_feature"])

# 4. Helper function to compute Mean and Std Dev arrays
def compute_trajectory_stats(group_dict, metric_key):
    means, stds = [], []
    for layer in layers:
        values = np.array(group_dict[metric_key][layer])
        means.append(np.mean(values) if len(values) > 0 else 0.0)
        stds.append(np.std(values) if len(values) > 0 else 0.0)
    return np.array(means), np.array(stds)

# Extract unified statistics
all_q_mean, all_q_std = compute_trajectory_stats(all_data, "q_norms")
all_t_mean, all_t_std = compute_trajectory_stats(all_data, "t_norms")

all_q_feat_mean, all_q_feat_std = compute_trajectory_stats(all_data, "q_feats")
all_t_feat_mean, all_t_feat_std = compute_trajectory_stats(all_data, "t_feats")


# 5. Core logic to inject vertical boundaries and text into subplots dynamically
def apply_layer_grouping_annotations(ax, y_query, y_text):
    max_val = max(max(y_query), max(y_text))
    min_val = min(min(y_query), min(y_text))
    
    current_layer = None
    layer_subindices = []
    
    for idx, key in enumerate(layers):
        layer_num = int(key.split('_')[0][1:])
        if current_layer is None:
            current_layer = layer_num
            
        if layer_num != current_layer:
            ax.axvline(x=idx - 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
            mid_x = np.mean(layer_subindices)
            ax.text(mid_x, max_val + (max_val * 0.02), f"Layer {current_layer}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')
            
            current_layer = layer_num
            layer_subindices = [idx]
        else:
            layer_subindices.append(idx)
            
    if layer_subindices:
        mid_x = np.mean(layer_subindices)
        ax.text(mid_x, max_val + (max_val * 0.02), f"Layer {current_layer}", 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')
        
    ax.set_ylim(min_val - (abs(min_val) * 0.05), max_val + (abs(max_val) * 0.12))


# --- FIGURE 1: UNIFIED L2 NORMS COMPILATION ---
fig1, ax1 = plt.subplots(figsize=(10, 6.5))

# Object Queries
ax1.plot(x_indices, all_q_mean, label="Object Queries", color="#1f77b4", marker='o', linewidth=2, zorder=3)
ax1.fill_between(x_indices, all_q_mean - all_q_std, all_q_mean + all_q_std, color="#1f77b4", alpha=0.25, zorder=2)

# Text Tokens Context (Arancione forzato e Std Dev più visibile con contorno tratteggiato fine)
ax1.plot(x_indices, all_t_mean, label="Text Tokens Context", color="#ff7f0e", marker='^', linewidth=1.5, linestyle="--", zorder=3)
ax1.fill_between(x_indices, all_t_mean - all_t_std, all_t_mean + all_t_std, color="#ff7f0e", alpha=0.25, zorder=2)

ax1.set_title("Global L2 Norm Trajectory across Decoder Layers", fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel("Average Max L2 Norm", fontsize=10)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_layer_grouping_annotations(ax1, all_q_mean, all_t_mean)

plt.tight_layout()
plt.savefig("analysis_plots/plots/scene46_max_norms_trajectory.png", dpi=300)
plt.close()


# --- FIGURE 2: UNIFIED ABSOLUTE FEATURE COMPILATION ---
fig2, ax2 = plt.subplots(figsize=(10, 6.5))

# Object Queries
ax2.plot(x_indices, all_q_feat_mean, label="Object Queries", color="#1f77b4", marker='o', linewidth=2, zorder=3)
ax2.fill_between(x_indices, all_q_feat_mean - all_q_feat_std, all_q_feat_mean + all_q_feat_std, color="#1f77b4", alpha=0.25, zorder=2)

# Text Tokens Context (Arancione forzato)
ax2.plot(x_indices, all_t_feat_mean, label="Text Tokens Context", color="#ff7f0e", marker='^', linewidth=1.5, linestyle="--", zorder=3)
ax2.fill_between(x_indices, all_t_feat_mean - all_t_feat_std, all_t_feat_mean + all_t_feat_std, color="#ff7f0e", alpha=0.25, zorder=2)

ax2.set_title("Global Max Absolute Feature Trajectory across Decoder Layers", fontsize=12, fontweight='bold', pad=15)
ax2.set_ylabel("Absolute Max Feature Value", fontsize=10)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_layer_grouping_annotations(ax2, all_q_feat_mean, all_t_feat_mean)

plt.tight_layout()
plt.savefig("analysis_plots/plots/scene46_max_feat_trajectory.png", dpi=300)
plt.close()

print("📊 Global unified plots generated and saved to 'analysis_plots/' ")