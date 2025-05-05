import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ratio_histogram_from_txt(txt_file, output_path="ratio_histogram.png", bins=50):
    # Load data
    try:
        ratios = np.loadtxt("avg_ratios.txt")
    except Exception as e:
        print(f"Failed to load avg_ratios.txt: {e}")
        return

    # Filter invalid or non-positive data
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(ratios, bins=bins, edgecolor='black')
    plt.xlabel("Estimated Lower Bound / True Distance")
    plt.ylabel("Count")
    plt.title("Histogram of Distance Ratio Estimates")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved histogram to {output_path}")

if __name__ == "__main__":
    plot_ratio_histogram_from_txt("avg_ratios.txt")
