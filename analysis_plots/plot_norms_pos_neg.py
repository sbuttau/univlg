import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your newly structured JSON data
json_path = "analysis_plots/scanrefer_scannet_anchor_val_multiple_batched_test_results.json"
with open(json_path, "r") as f:
    data = json.load(f)

os.makedirs("analysis_plots", exist_ok=True)

# 2. Dynamically extract and systematically sort the ordered list of layers
sample_entry = next(item for item in data if "logged_norms" in item)
raw_layers = list(sample_entry["logged_norms"].keys())
layers = sorted(raw_layers, key=lambda x: (int(x.split('_')[0][1:]), x.split('_')[1]))

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

pos_count = 0
neg_count = 0

# 3. Populate lists separating Positives (success=1) and Negatives (success=0)
for entry in data:
    if "logged_norms" not in entry or not entry["logged_norms"]:
        continue
        
    is_success = entry["success"] == 1
    if is_success:
        pos_count += 1
    else:
        neg_count += 1
        
    target_group = pos_data if is_success else neg_data
    norms_dict = entry["logged_norms"]
    
    for layer in layers:
        if layer in norms_dict:
            target_group["q_norms"][layer].append(norms_dict[layer]["max_query_norm"])
            target_group["t_norms"][layer].append(norms_dict[layer]["max_text_norm"])
            target_group["q_feats"][layer].append(norms_dict[layer]["max_query_feature"])
            target_group["t_feats"][layer].append(norms_dict[layer]["max_text_feature"])

print("=" * 40)
print(f"📊 DATASET DISTRIBUTION OVERVIEW:")
print(f"   - Positive Samples (success=1): {pos_count}")
print(f"   - Negative Samples (success=0): {neg_count}")
print("=" * 40)

# 4. Compute Mean and Std Dev arrays
def compute_trajectory_stats(group_dict, metric_key):
    means, stds = [], []
    for layer in layers:
        values = np.array(group_dict[metric_key][layer])
        means.append(np.mean(values) if len(values) > 0 else 0.0)
        stds.append(np.std(values) if len(values) > 0 else 0.0)
    return np.array(means), np.array(stds)

pos_q_mean, pos_q_std = compute_trajectory_stats(pos_data, "q_norms")
pos_t_mean, pos_t_std = compute_trajectory_stats(pos_data, "t_norms")
neg_q_mean, neg_q_std = compute_trajectory_stats(neg_data, "q_norms")
neg_t_mean, neg_t_std = compute_trajectory_stats(neg_data, "t_norms")

pos_q_feat_mean, pos_q_feat_std = compute_trajectory_stats(pos_data, "q_feats")
pos_t_feat_mean, pos_t_feat_std = compute_trajectory_stats(pos_data, "t_feats")
neg_q_feat_mean, neg_q_feat_std = compute_trajectory_stats(neg_data, "q_feats")
neg_t_feat_mean, neg_t_feat_std = compute_trajectory_stats(neg_data, "t_feats")


# 5. Core logic for layout annotations
def apply_synchronized_annotations(ax, ylim_tuple):
    ymin, ymax = ylim_tuple
    current_layer = None
    layer_subindices = []
    
    for idx, key in enumerate(layers):
        layer_num = int(key.split('_')[0][1:])
        if current_layer is None:
            current_layer = layer_num
            
        if layer_num != current_layer:
            ax.axvline(x=idx - 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
            mid_x = np.mean(layer_subindices)
            ax.text(mid_x, ymax - (abs(ymax) * 0.08), f"Layer {current_layer}", 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#444444')
            
            current_layer = layer_num
            layer_subindices = [idx]
        else:
            layer_subindices.append(idx)
            
    if layer_subindices:
        mid_x = np.mean(layer_subindices)
        ax.text(mid_x, ymax - (abs(ymax) * 0.08), f"Layer {current_layer}", 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#444444')
        
    ax.set_ylim(ymin, ymax)


# 6. Global Bounds Pre-Calculation
def calculate_global_bounds(pos_m, pos_s, neg_m, neg_s, pos_m2, pos_s2, neg_m2, neg_s2):
    all_values = np.concatenate([pos_m, neg_m, pos_m2, neg_m2])
    max_val = np.max(all_values) if len(all_values) > 0 else 1.0
    min_val = np.min(all_values) if len(all_values) > 0 else 0.0
    return min_val - (abs(min_val) * 0.05), max_val + (abs(max_val) * 0.15)

global_norm_limits = calculate_global_bounds(pos_q_mean, pos_q_std, neg_q_mean, neg_q_std, pos_t_mean, pos_t_std, neg_t_mean, neg_t_std)
global_feat_limits = calculate_global_bounds(pos_q_feat_mean, pos_q_feat_std, neg_q_feat_mean, neg_q_feat_std, pos_t_feat_mean, pos_t_feat_std, neg_t_feat_mean, neg_t_feat_std)


# 🎯 NEW: Unified wrapper to enforce high-visibility Standard Deviation bands
def plot_with_enhanced_std(ax, x, mean, std, label, color, marker, linestyle="-"):
    # Main Trajectory Line
    ax.plot(x, mean, label=label, color=color, marker=marker, linewidth=2, linestyle=linestyle, zorder=4)
    
    # Enhanced Fill Variance Band (Alpha aumentato a 0.22 per visibilità ottimale)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22, zorder=2)
    
    # Explicit Boundary Lines for the Std Dev (Rende visibile la forma della varianza)
    ax.plot(x, mean - std, color=color, linestyle=":", linewidth=0.8, alpha=0.6, zorder=3)
    ax.plot(x, mean + std, color=color, linestyle=":", linewidth=0.8, alpha=0.6, zorder=3)


# --- FIGURE 1: SIDE-BY-SIDE L2 NORMS COMPILATION ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

# Left Side: Positives
plot_with_enhanced_std(ax1, x_indices, pos_q_mean, pos_q_std, "Object Queries", "#1f77b4", 'o')
plot_with_enhanced_std(ax1, x_indices, pos_t_mean, pos_t_std, "Text Tokens Context", "#ff7f0e", '^', linestyle="--")
ax1.set_title("Positive Samples (Successful Groundings)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Average Max L2 Norm", fontsize=10)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_synchronized_annotations(ax1, global_norm_limits)

# Right Side: Negatives
plot_with_enhanced_std(ax2, x_indices, neg_q_mean, neg_q_std, "Object Queries", "#1f77b4", 'o')
plot_with_enhanced_std(ax2, x_indices, neg_t_mean, neg_t_std, "Text Tokens Context", "#ff7f0e", '^', linestyle="--")
ax2.set_title("Negative Samples (Failed Groundings)", fontsize=12, fontweight='bold')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_synchronized_annotations(ax2, global_norm_limits)

fig1.suptitle("Avg Max L2 Norm", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("analysis_plots/plots/scanrefer_val_multiple_pos_neg_max_norms_trajectory_comparison.png", dpi=300)
plt.close()


# --- FIGURE 2: SIDE-BY-SIDE ABSOLUTE FEATURE COMPILATION ---
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

# Left Side: Positives
plot_with_enhanced_std(ax3, x_indices, pos_q_feat_mean, pos_q_feat_std, "Object Tokens", "#1f77b4", 'o')
plot_with_enhanced_std(ax3, x_indices, pos_t_feat_mean, pos_t_feat_std, "Text Tokens", "#ff7f0e", '^', linestyle="--")
ax3.set_title("Positive Samples (Successful cases)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Absolute Max Feature Value", fontsize=10)
ax3.set_xticks(x_indices)
ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax3.grid(True, linestyle=':', alpha=0.3)
ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_synchronized_annotations(ax3, global_feat_limits)

# Right Side: Negatives
print(f"Mean: {neg_q_feat_mean}, Std: {neg_q_feat_std}")
plot_with_enhanced_std(ax4, x_indices, neg_q_feat_mean, neg_q_feat_std, "Query Tokens", "#1f77b4", 'o')
plot_with_enhanced_std(ax4, x_indices, neg_t_feat_mean, neg_t_feat_std, "Text Tokens", "#ff7f0e", '^', linestyle="--")
ax4.set_title("Negative Samples (Failure cases)", fontsize=12, fontweight='bold')
ax4.set_xticks(x_indices)
ax4.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax4.grid(True, linestyle=':', alpha=0.3)
ax4.legend(loc="lower right", framealpha=0.9, fontsize=9)
apply_synchronized_annotations(ax4, global_feat_limits)

fig2.suptitle("Avg Max Absolute Features", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("analysis_plots/plots/scanrefer_val_multiple_pos_neg_max_feat_trajectory_comparison.png", dpi=300)
plt.close()

print("📈 High-visibility variance plots updated and saved.")