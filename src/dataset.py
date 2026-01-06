import os
import numpy as np
import librosa

def list_audio_files(root_dir):
    audio_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".mp3", ".wav")):
                audio_files.append(os.path.join(root, file))
    return audio_files

def extract_mfcc(file_path, duration=30, sr=22050, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

def build_dataset(audio_root):
    X = []
    y = []

    for label, lang in enumerate(["english", "bangla"]):
        folder = os.path.join(audio_root, lang)
        files = list_audio_files(folder)

        for file in files:
            mfcc = extract_mfcc(file)
            X.append(mfcc)
            y.append(label)

    return np.array(X), np.array(y)
