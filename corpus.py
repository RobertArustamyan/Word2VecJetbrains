import re
import numpy as np
from collections import Counter

STOPWORDS = {"the", "is", "are", "was", "and", "of", "to",
             "in", "that", "it", "from", "with", "on", "by"}

def load_text(path: str, max_chars: int = None) -> str:
    """read text from a file, with possibility of truncating to max_chars."""
    with open(path, "r") as f:
        if max_chars:
            return f.read(max_chars)
        return f.read()

def tokenize(text: str) -> list[str]:
    """lowercases text, splits into words, and removes stopwords and single-char tokens."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]

def build_vocab(tokens: list[str], min_count: int = 2) -> tuple:
    """Builds word-to-index and index-to-word mappings and a frequency array"""
    counts = Counter(tokens)
    vocab_words = [w for w, c in counts.most_common() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = vocab_words
    freq_array = np.array([counts[w] for w in vocab_words], dtype=np.float64)
    return word2idx, idx2word, freq_array

def tokens_to_ids(tokens: list[str], word2idx: dict) -> list[int]:
    """Converts a list of string tokens to integer indices"""
    return [word2idx[w] for w in tokens if w in word2idx]

def subsample(token_ids: list[int], freq_array: np.ndarray, t: float = 1e-4) -> list[int]:
    """Randomly discards frequent tokens to reduce their influence on training"""
    total = freq_array.sum()
    keep_probs = np.sqrt(t / (freq_array / total))
    keep_probs = np.minimum(keep_probs, 1.0)
    return [idx for idx in token_ids if np.random.random() < keep_probs[idx]]


if __name__ == "__main__":
    text = load_text("data/sample1.txt")
    tokens = tokenize(text)
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=2)
    token_ids = tokens_to_ids(tokens, word2idx)
    token_ids = subsample(token_ids, freq_array)

    print(f"Unique tokens    : {len(tokens)}")
    print(f"Vocabulary size  : {len(idx2word)}")
    print(f"Tokens after sub : {len(token_ids)}")