import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your newly structured JSON data
json_path = "analysis_plots/scanrefer_scannet_anchor_val_single_batched_test_results.json"
with open(json_path, "r") as f:
    data = json.load(f)

# 2. Dynamically extract and systematically sort the ordered list of layers
sample_entry = next(item for item in data if "logged_norms" in item)
raw_layers = list(sample_entry["logged_norms"].keys())
# Garantisce l'ordinamento numerico e per sotto-layer (L0_1, L0_2, ecc.)
layers = sorted(raw_layers, key=lambda x: (int(x.split('_')[0][1:]), x.split('_')[1]))

# Clean labels to display only the sub-layer stage on the x-axis
x_labels = [k.replace("1_Cross_Attn", "Cross").replace("2_Self_Attn", "Self").replace("3_Main_FFN", "FFN").split('_')[-1] for k in layers]
x_indices = np.arange(len(layers))

# Initialize data accumulators
metrics_template = lambda: {layer: [] for layer in layers}
pos_data = {
    "q_norms": metrics_template(), "t_norms": metrics_template(),
    "q_feats": metrics_template(), "t_feats": metrics_template()
}
neg_data = {
    "q_norms": metrics_template(), "t_norms": metrics_template(),
    "q_feats": metrics_template(), "t_feats": metrics_template()
}

# 3. Populate lists separating Positives (success=1) and Negatives (success=0)
for entry in data:
    if "logged_norms" not in entry or not entry["logged_norms"]:
        continue
        
    is_success = entry["success"] == 1
    target_group = pos_data if is_success else neg_data
    norms_dict = entry["logged_norms"]
    
    for layer in layers:
        if layer in norms_dict:
            target_group["q_norms"][layer].append(norms_dict[layer]["max_query_norm"])
            target_group["t_norms"][layer].append(norms_dict[layer]["max_text_norm"])
            target_group["q_feats"][layer].append(norms_dict[layer]["max_query_feature"])
            target_group["t_feats"][layer].append(norms_dict[layer]["max_text_feature"])

# 4. Helper function to compute Mean and Std Dev arrays
def compute_trajectory_stats(group_dict, metric_key):
    means, stds = [], []
    for layer in layers:
        values = np.array(group_dict[metric_key][layer])
        means.append(np.mean(values) if len(values) > 0 else 0.0)
        stds.append(np.std(values) if len(values) > 0 else 0.0)
    return np.array(means), np.array(stds)

# Extract statistics
pos_q_mean, pos_q_std = compute_trajectory_stats(pos_data, "q_norms")
pos_t_mean, pos_t_std = compute_trajectory_stats(pos_data, "t_norms")
neg_q_mean, neg_q_std = compute_trajectory_stats(neg_data, "q_norms")
neg_t_mean, neg_t_std = compute_trajectory_stats(neg_data, "t_norms")

pos_q_feat_mean, pos_q_feat_std = compute_trajectory_stats(pos_data, "q_feats")
pos_t_feat_mean, pos_t_feat_std = compute_trajectory_stats(pos_data, "t_feats")
neg_q_feat_mean, neg_q_feat_std = compute_trajectory_stats(neg_data, "q_feats")
neg_t_feat_mean, neg_t_feat_std = compute_trajectory_stats(neg_data, "t_feats")


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
            # Draw boundary line on the specific axis
            ax.axvline(x=idx - 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
            
            # Place label at the center of the completed block
            mid_x = np.mean(layer_subindices)
            ax.text(mid_x, max_val + (max_val * 0.02), f"Layer {current_layer}", 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#444444')
            
            current_layer = layer_num
            layer_subindices = [idx]
        else:
            layer_subindices.append(idx)
            
    if layer_subindices:
        mid_x = np.mean(layer_subindices)
        ax.text(mid_x, max_val + (max_val * 0.02), f"Layer {current_layer}", 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#444444')
        
    ax.set_ylim(min_val - (abs(min_val) * 0.05), max_val + (abs(max_val) * 0.12))


# --- FIGURE 1: SIDE-BY-SIDE L2 NORMS COMPILATION ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

# Left Side: Positives
ax1.plot(x_indices, pos_q_mean, label="Object Queries", color="#1f77b4", marker='o', linewidth=2, zorder=3)
ax1.fill_between(x_indices, pos_q_mean - pos_q_std, pos_q_mean + pos_q_std, color="#1f77b4", alpha=0.15, zorder=2)
ax1.plot(x_indices, pos_t_mean, label="Text Tokens Context", color="#ff7f0e", marker='^', linewidth=1.5, linestyle="--", zorder=3)
ax1.fill_between(x_indices, pos_t_mean - pos_t_std, pos_t_mean + pos_t_std, color="#ff7f0e", alpha=0.15, zorder=2)
ax1.set_title("Positive Samples (Successful Groundings)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Average Max L2 Norm", fontsize=10)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_layer_grouping_annotations(ax1, pos_q_mean, pos_t_mean)

# Right Side: Negatives
ax2.plot(x_indices, neg_q_mean, label="Object Queries", color="#1f77b4", marker='o', linewidth=2, zorder=3)
ax2.fill_between(x_indices, neg_q_mean - neg_q_std, neg_q_mean + neg_q_std, color="#1f77b4", alpha=0.15, zorder=2)
ax2.plot(x_indices, neg_t_mean, label="Text Tokens Context", color="#d62728", marker='^', linewidth=1.5, linestyle="--", zorder=3)
ax2.fill_between(x_indices, neg_t_mean - neg_t_std, neg_t_mean + neg_t_std, color="#d62728", alpha=0.15, zorder=2)
ax2.set_title("Negative Samples (Failed Groundings)", fontsize=12, fontweight='bold')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_layer_grouping_annotations(ax2, neg_q_mean, neg_t_mean)

fig1.suptitle("Fine-Grained L2 Norm Trajectory across Decoder Layers", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("analysis_plots/grouped_max_norms_trajectory_comparison.png", dpi=300)
plt.close()


# --- FIGURE 2: SIDE-BY-SIDE ABSOLUTE FEATURE COMPILATION ---
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

# Left Side: Positives
ax3.plot(x_indices, pos_q_feat_mean, label="Object Queries", color="#1f77b4", marker='o', linewidth=2, zorder=3)
ax3.fill_between(x_indices, pos_q_feat_mean - pos_q_feat_std, pos_q_feat_mean + pos_q_feat_std, color="#1f77b4", alpha=0.15, zorder=2)
ax3.plot(x_indices, pos_t_feat_mean, label="Text Tokens Context", color="#ff7f0e", marker='^', linewidth=1.5, linestyle="--", zorder=3)
ax3.fill_between(x_indices, pos_t_feat_mean - pos_t_feat_std, pos_t_feat_mean + pos_t_feat_std, color="#ff7f0e", alpha=0.15, zorder=2)
ax3.set_title("Positive Samples (Successful Groundings)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Absolute Max Feature Value", fontsize=10)
ax3.set_xticks(x_indices)
ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax3.grid(True, linestyle=':', alpha=0.3)
ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_layer_grouping_annotations(ax3, pos_q_feat_mean, pos_t_feat_mean)

# Right Side: Negatives
ax4.plot(x_indices, neg_q_feat_mean, label="Object Queries", color="#1f77b4", marker='o', linewidth=2, zorder=3)
ax4.fill_between(x_indices, neg_q_feat_mean - neg_q_feat_std, neg_q_feat_mean + neg_q_feat_std, color="#1f77b4", alpha=0.15, zorder=2)
ax4.plot(x_indices, neg_t_feat_mean, label="Text Tokens Context", color="#d62728", marker='^', linewidth=1.5, linestyle="--", zorder=3)
ax4.fill_between(x_indices, neg_t_feat_mean - neg_t_feat_std, neg_t_feat_mean + neg_t_feat_std, color="#d62728", alpha=0.15, zorder=2)
ax4.set_title("Negative Samples (Failed Groundings)", fontsize=12, fontweight='bold')
ax4.set_xticks(x_indices)
ax4.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax4.grid(True, linestyle=':', alpha=0.3)
ax4.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_layer_grouping_annotations(ax4, neg_q_feat_mean, neg_t_feat_mean)

fig2.suptitle("Fine-Grained Max Absolute Feature Trajectory across Decoder Layers", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("analysis_plots/grouped_max_feat_trajectory_comparison.png", dpi=300)
plt.close()

print("📊 Both side-by-side grouped plots generated and saved to 'analysis_plots/'.")