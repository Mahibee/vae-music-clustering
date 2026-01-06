import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def list_lyrics_files(lyrics_root):
    return [
        os.path.join(lyrics_root, f)
        for f in os.listdir(lyrics_root)
        if f.lower().endswith(".txt")
    ]

def load_lyrics_text(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return text

def build_lyrics_tfidf(lyrics_root):
    paths = list_lyrics_files(lyrics_root)

    texts = []
    valid_paths = []

    for p in paths:
        t = load_lyrics_text(p)
        if len(t.split()) > 3:   # <-- ignore empty/bad files
            texts.append(t)
            valid_paths.append(p)

    if len(texts) < 2:
        raise ValueError("Not enough valid lyrics files to build TF-IDF")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b"  # accept all words
    )

    X = vectorizer.fit_transform(texts).toarray()
    return valid_paths, X

def cluster_lyrics(X, k=2):
    k = min(k, len(X))  # safety
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)
