import matplotlib.pyplot as plt
import numpy as np
import os

source = '/data/vector_datasets/'
datasets = ['sift']
K=64

def plot_ratio_histogram_from_txt(txt_file, output_path, bins=80):
    # Load data
    try:
        ratios = np.loadtxt(txt_file)
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
    plt.title("Histogram of Distance Ratio Estimates with K = " + str(K))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved histogram to {output_path}")

if __name__ == "__main__":
    # string output_file = path + dataset + "_avg_ratios_" + to_string(K) + ".txt";
    output_path = "avg_ratios_" + datasets[0] + "_" + str(K) + ".png"
    plot_ratio_histogram_from_txt(source + datasets[0] + "/" + datasets[0] + "_avg_ratios_" + str(K) + ".txt", output_path) 
