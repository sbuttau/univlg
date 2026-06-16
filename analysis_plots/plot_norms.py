import json
import matplotlib.pyplot as plt

# Load your profile extraction data
with open("/workspaces/univlg/analysis_plots/fine_grained_norms.json", "r") as f:
    data = json.load(f)

# Sort entries systematically by Layer and Sublayer Stage
sorted_keys = sorted(data.keys(), key=lambda x: (int(x.split('_')[0][1:]), x.split('_')[1]))

x_labels = [k.replace("1_Vision_Cross", "Cross").replace("2_Self_Attn", "Self").replace("3_Main_FFN", "FFN") for k in sorted_keys]
query_y = [data[k]["avg_max_query_norm"] for k in sorted_keys]
text_y = [data[k]["avg_max_text_norm"] for k in sorted_keys]

plt.figure(figsize=(14, 5))

# Plot line charts matching the format of Figure 2
plt.plot(range(len(sorted_keys)), query_y, marker='o', color='#1f77b4', linewidth=2, label="Object Queries")
plt.plot(range(len(sorted_keys)), text_y, marker='^', color='#ff7f0e', linewidth=1.5, linestyle="--", label="Text Tokens Context")

plt.title("Fine-Grained Outlier Max Norm Profile Across Decoder Sub-Layers", fontsize=12, fontweight='bold')
plt.xlabel("Decoder Pipeline Sub-layers (Layer_Stage)", fontsize=10)
plt.ylabel("Average Max $L_2$ Norm", fontsize=10)

# Set custom x-ticks showing sub-layers clearly
plt.xticks(range(len(sorted_keys)), x_labels, rotation=45, ha='right', fontsize=8)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("/workspaces/univlg/analysis_plots/reproduced_figure2.png", dpi=300)
print("📊 Reconstructed Figure 2 successfully compiled and saved to disk!")