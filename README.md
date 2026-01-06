\# VAE Music Clustering (Audio + Lyrics) with PCA Baseline



\## What this project does

\- Extracts MFCC features from short audio clips

\- Trains a Variational Autoencoder (VAE) to learn latent representations

\- Clusters the latent space using K-Means

\- Compares results with a PCA + K-Means baseline

\- Adds a lightweight lyrics experiment using TF-IDF + K-Means

\- Visualizes audio clusters with t-SNE



\## Folder structure

\- `data/audio/` (not included in repo)

\- `data/lyrics/` (not included in repo)

\- `src/` source code

\- `results/` outputs (not included in repo)



\## Setup (Windows)

```bash

python -m venv venv

venv\\Scripts\\activate

pip install --upgrade pip

pip install -r requirements.txt



