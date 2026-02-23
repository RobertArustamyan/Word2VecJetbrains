import re
import numpy as np
from collections import Counter


def load_text(path: str) -> str:
    """Reads raw text from file."""
    with open(path, "r") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def build_vocab(tokens: list[str], min_count: int = 2) -> tuple:
    """
    Build vocabulary from token list.
    """
    counts = Counter(tokens)
    vocab_words = [w for w, c in counts.most_common() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = vocab_words
    freq_array = np.array([counts[w] for w in vocab_words], dtype=np.float64)

    return word2idx, idx2word, freq_array


def tokens_to_ids(tokens: list[str], word2idx: dict) -> list[int]:
    """Convert string tokens to integer ids, skipping unknown words."""
    return [word2idx[w] for w in tokens if w in word2idx]


def subsample(token_ids: list[int], freq_array: np.ndarray, t: float = 1e-4) -> list[int]:
    """
    Discard frequent words with probability: P(discard) = 1 - sqrt(t / freq(w))
    """
    total = freq_array.sum()
    keep_probs = np.sqrt(t / (freq_array / total))
    keep_probs = np.minimum(keep_probs, 1.0)

    return [idx for idx in token_ids if np.random.random() < keep_probs[idx]]


if __name__ == "__main__":
    text = load_text("data/sample1.txt")
    tokens = tokenize(text)
    print(tokens)
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=2)
    token_ids = tokens_to_ids(tokens, word2idx)
    token_ids = subsample(token_ids, freq_array)

    print(f"Tokens: {tokens}")
    print(f"Unique tokens : {len(tokens)}")
    print(f"Vocabulary size : {len(idx2word)}")
    print(f"Tokens after sub : {len(token_ids)}")

