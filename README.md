# VAE Music Clustering (Audio + Lyrics) with PCA Baseline

## What this project does
- Extracts MFCC features from short audio clips
- Trains a Variational Autoencoder (VAE) to learn latent representations
- Performs K-Means clustering in the learned latent space
- Compares results with a PCA + K-Means baseline
- (Optional) Lyrics experiment using TF-IDF + K-Means (if lyrics are available)
- Visualizes clusters using t-SNE (and PCA plot for baseline)

## Repository structure
- `src/` : source code
- `run_pipeline.py` : main runner script
- `results/` : generated figures (created after running)
- `data/` : local dataset folder (not pushed to GitHub)

## Data availability (not included in GitHub)
Audio and lyrics are not included in this repository due to storage/copyright.

Place files like:
```text
data/
└── audio/
    └── gtzan/
        ├── blues/
        ├── classical/
        ├── country/
        ├── disco/
        ├── hiphop/
        ├── jazz/
        ├── metal/
        ├── pop/
        ├── reggae/
        └── rock/

Setup (Windows):
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

Run:
python run_pipeline.py




