import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_layer_grouped_plot(x_indices, y_query, y_text, sorted_keys, x_labels, title, ylabel, output_path):
    """Helper function to isolate plotting logic and prevent layout corruption."""
    plt.figure(figsize=(15, 6))

    # Plot continuous lines to preserve the fine-grained sawtooth dynamics
    plt.plot(x_indices, y_query, marker='o', color='#1f77b4', linewidth=2, label="Object Queries", zorder=3)
    plt.plot(x_indices, y_text, marker='^', color='#ff7f0e', linewidth=1.5, linestyle="--", label="Text Tokens Context", zorder=3)

    # Dynamically find the local peak for text placement boundary
    max_val = max(max(y_query), max(y_text))
    min_val = min(min(y_query), min(y_text))

    # Group sub-layers into unified vertical Layer spans
    current_layer = None
    layer_subindices = []
    
    for idx, key in enumerate(sorted_keys):
        layer_num = int(key.split('_')[0][1:])
        if current_layer is None:
            current_layer = layer_num
            
        if layer_num != current_layer:
            # Draw a vertical separation boundary before the new layer starts
            plt.axvline(x=idx - 0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1.2)
            
            # Label the completed layer at the center of its sub-layers
            mid_x = np.mean(layer_subindices)
            plt.text(mid_x, max_val + (max_val * 0.01), f"Layer {current_layer}", 
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')
            
            current_layer = layer_num
            layer_subindices = [idx]
        else:
            layer_subindices.append(idx)
            
    if layer_subindices:
        mid_x = np.mean(layer_subindices)
        plt.text(mid_x, max_val + (max_val * 0.01), f"Layer {current_layer}", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')

    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.xlabel("Sub-layers", fontsize=10, labelpad=10)
    plt.ylabel(ylabel, fontsize=10)

    plt.xticks(x_indices, x_labels, rotation=45, ha='right', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.3)
    
    # Position the legend safely in the lower right using a solid background frame
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    
    # Set explicit margins based on the specific metric range
    plt.ylim(min_val - (abs(min_val) * 0.05), max_val + (abs(max_val) * 0.08))
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close() # Crucial to free memory and prevent cross-figure leakage

def main():
    parser = argparse.ArgumentParser(description="Plot Fine-Grained Metrics with Clean Layer Grouping.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON metrics file.")
    parser.add_argument("--output_dir", type=str, default="/workspaces/univlg/analysis_plots", help="Directory to save the plots.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found at: {args.input}")

    with open(args.input, "r") as f:
        data = json.load(f)

    # Sort entries systematically by Layer and Sublayer Stage
    sorted_keys = sorted(data.keys(), key=lambda x: (int(x.split('_')[0][1:]), x.split('_')[1]))

    # Clean labels to display only the sub-layer stage
    x_labels = [k.replace("1_Cross_Attn", "Cross").replace("2_Self_Attn", "Self").replace("3_Main_FFN", "FFN").split('_')[-1] for k in sorted_keys]
    x_indices = np.arange(len(sorted_keys))

    os.makedirs(args.output_dir, exist_ok=True)

    # --- FIGURE 1: L2 NORM PROFILE ---
    query_norm_y = [data[k]["avg_max_query_norm"] for k in sorted_keys]
    text_norm_y = [data[k]["avg_max_text_norm"] for k in sorted_keys]
    norm_out_path = os.path.join(args.output_dir, "grouped_max_norms_scanrefer_val_scene041.png")
    
    generate_layer_grouped_plot(
        x_indices=x_indices, y_query=query_norm_y, y_text=text_norm_y,
        sorted_keys=sorted_keys, x_labels=x_labels,
        title="Average Max $L_2$ Norm (1000 samples)",
        ylabel="Average Max $L_2$ Norm", output_path=norm_out_path
    )
    print(f"📊 Figure 1 (Norms) successfully compiled and saved to: {norm_out_path}")

    # --- FIGURE 2: ABSOLUTE FEATURE CHANNEL PROFILE ---
    query_max_y = [data[k]["avg_max_query_feature"] for k in sorted_keys]
    text_max_y = [data[k]["avg_max_text_feature"] for k in sorted_keys]
    feat_out_path = os.path.join(args.output_dir, "grouped_max_features_scanrefer_val_scene041.png")
    
    generate_layer_grouped_plot(
        x_indices=x_indices, y_query=query_max_y, y_text=text_max_y,
        sorted_keys=sorted_keys, x_labels=x_labels,
        title="Max Absolute Feature Values (1000 samples)",
        ylabel="Absolute Max Feature Value", output_path=feat_out_path
    )
    print(f"📊 Figure 2 (Features) successfully compiled and saved to: {feat_out_path}")

if __name__ == "__main__":
    main()