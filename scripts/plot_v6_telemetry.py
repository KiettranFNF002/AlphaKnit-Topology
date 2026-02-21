import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

def plot_telemetry(history_path, output_dir="plots"):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return

    with open(history_path) as f:
        history = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    epochs = [row["epoch"] for row in history]
    margin = [row.get("struct_margin", 0) for row in history]
    acc = [row.get("struct_acc", 0) for row in history]
    phase_lag = [row.get("phase_lag", 1) for row in history]
    entropy = [row.get("train_entropy", 0) for row in history]
    
    # 1. Main Metrics Plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, margin, marker='o', color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Structural Logit Margin (Primary Order Parameter)")
    plt.ylabel("Margin")
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, phase_lag, marker='s', color='green')
    plt.title("Optimizer Phase Lag (cos θ)")
    plt.ylabel("Cosine Similarity")
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, acc, marker='^', color='orange')
    plt.title("Structural Accuracy (Top-1)")
    plt.ylabel("Accuracy")
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, entropy, marker='x', color='purple')
    plt.title("Structural Training Entropy")
    plt.ylabel("Entropy")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v6_metrics.png"))
    print(f"Saved metrics plot to {output_dir}/v6_metrics.png")
    
    # 2. Latent Phase Portrait (PCA)
    latent_vectors = [row.get("latent_vector") for row in history if row.get("latent_vector") is not None]
    if len(latent_vectors) >= 3:
        X = np.array(latent_vectors)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        plt.plot(coords[:, 0], coords[:, 1], linestyle='-', color='gray', alpha=0.3)
        plt.scatter(coords[:, 0], coords[:, 1], c=epochs[:len(coords)], cmap='viridis', s=100, edgecolors='black')
        
        # Annotate start and end
        plt.annotate("Start", (coords[0, 0], coords[0, 1]), xytext=(5, 5), textcoords='offset points')
        plt.annotate("Current", (coords[-1, 0], coords[-1, 1]), xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(label="Epoch")
        plt.title("Latent Phase Portrait (PCA Trajectory)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.2)
        
        plt.savefig(os.path.join(output_dir, "phase_portrait.png"))
        print(f"Saved phase portrait to {output_dir}/phase_portrait.png")
    else:
        print("Not enough latent data for Phase Portrait.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True, help="Path to training_history_*.json")
    parser.add_argument("--output", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()
    
    plot_telemetry(args.history, args.output)
