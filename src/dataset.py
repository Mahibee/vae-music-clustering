import os
import numpy as np
import librosa

def extract_mfcc(file_path, duration=30, sr=22050, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def build_dataset(audio_root):
    X = []
    y = []

    genre_folders = [
        d for d in os.listdir(audio_root)
        if os.path.isdir(os.path.join(audio_root, d))
    ]

    print(f"Found genres: {genre_folders}")

    for label, genre in enumerate(genre_folders):
        genre_path = os.path.join(audio_root, genre)

        for root, _, files in os.walk(genre_path):
            for file in files:
                if file.lower().endswith((".wav", ".mp3", ".au")):

                    file_path = os.path.join(root, file)
                    mfcc = extract_mfcc(file_path)

                    if mfcc is not None:
                        X.append(mfcc)
                        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {X.shape[0]} audio files")
    print(f"Feature dimension: {X.shape[1] if X.shape[0] > 0 else 'N/A'}")

    return X, y
