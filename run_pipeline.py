import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA

from src.dataset import build_dataset
from src.vae import VAE
from src.clustering import apply_kmeans, visualize_tsne, visualize_pca
from src.evaluation import evaluate_clustering
from src.lyrics_pipeline import build_lyrics_tfidf, cluster_lyrics


def train_vae(model, data, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = torch.tensor(data, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstructed, mu, logvar = model(data)

        recon_loss = torch.mean((reconstructed - data) ** 2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def main():
    print("Loading dataset...")
    X, y = build_dataset("data/audio/gtzan")
    print("X type:", type(X))
    print("X shape:", np.array(X).shape)
    print("First element:", X[0] if len(X) > 0 else "EMPTY")



    print("Training VAE...")
    vae = VAE(input_dim=X.shape[1])
    train_vae(vae, X)

    with torch.no_grad():
        _, mu, _ = vae(torch.tensor(X, dtype=torch.float32))
        latent = mu.numpy()

    # -------- VAE + KMeans --------
    print("Clustering (VAE latent)...")
    labels = apply_kmeans(latent)

    print("Evaluating VAE clustering...")
    scores = evaluate_clustering(latent, labels)
    print(scores)

    os.makedirs("results", exist_ok=True)
    visualize_tsne(latent, labels, "results/tsne_clusters.png")

    # -------- PCA + KMeans Baseline --------
    print("Running PCA baseline...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    pca_labels = apply_kmeans(X_pca, n_clusters=2)

    pca_scores = evaluate_clustering(X_pca, pca_labels)
    print("PCA baseline metrics:", pca_scores)

    visualize_pca(X, pca_labels, "results/pca_clusters.png")

    # -------- Lyrics Add-on (TF-IDF + KMeans) --------
    #print("Loading lyrics (TF-IDF)...")
    #lyrics_dir = os.path.join("data", "lyrics")

    #lyric_paths, X_lyrics = build_lyrics_tfidf(lyrics_dir)

    #print("Lyrics clustering...")
    #lyric_labels = cluster_lyrics(X_lyrics, k=2)

    #print("Evaluating lyrics clustering...")
    #lyric_scores = evaluate_clustering(X_lyrics, lyric_labels)
    #print("Lyrics metrics:", lyric_scores)

    print("Done! Results saved in 'results/' folder.")


if __name__ == "__main__":
    main()
