# VAE Music Clustering (Audio + Lyrics) with PCA Baseline

## What this project does
- Extracts MFCC features from short audio clips
- Trains a Variational Autoencoder (VAE) to learn latent representations
- Performs K-Means clustering in the learned latent space
- Compares clustering results with a PCA + K-Means baseline
- Includes a lightweight lyrics-based experiment using TF-IDF + K-Means
- Visualizes audio clusters using t-SNE

## Folder Structure
- `data/audio/` (not included in the repository)
- `data/lyrics/` (not included in the repository)
- `src/` source code
- `results/` generated outputs

## Data Availability and Setup

The audio files and song lyrics used in this project are not included in the repository due to
copyright and storage limitations.

To reproduce the experiments, organize your local dataset as follows:

```text
data/
├── audio/
│   ├── english/
│   │   └── song1.mp3
│   └── bangla/
│       └── song2.mp3
└── lyrics/
    └── song1.txt

Audio clips should be approximately 20–30 seconds long.  
Lyrics filenames must match the corresponding audio filenames.

Once the data is placed correctly, run the full pipeline using:
```bash
python run_pipeline.py

## Setup (Windows)
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py

