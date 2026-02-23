import numpy as np

from word2vec import Word2Vec
from corpus import tokenize, tokens_to_ids, subsample
from utils import get_negative_sampler


def generate_pairs(token_ids: list[int], window: int = 5):
    """Yields all (center, context) pairs from the token sequence using random window size"""
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        actual_window = np.random.randint(1, window + 1)
        left  = max(0, i - actual_window)
        right = min(n, i + actual_window + 1)
        for j in range(left, right):
            if j != i:
                yield center, token_ids[j]


def train(text: str, vocab_size: int, word2idx: dict, idx2word: list, freq_array: np.ndarray,
          embed_dim: int = 100, n_negative: int = 5, window: int = 5,
          n_epochs: int = 10, lr_start: float = 0.025, lr_min: float = 0.0001):
    model = Word2Vec(vocab_size, embed_dim, n_negative, lr_start)
    neg_sample = get_negative_sampler(freq_array)

    token_ids = tokens_to_ids(tokenize(text), word2idx)
    token_ids = subsample(token_ids, freq_array)

    all_pairs = list(generate_pairs(token_ids, window=window))
    total_steps = len(all_pairs) * n_epochs
    print(f"Total training steps: {total_steps:,}")
    # Training loop
    step = 0
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        np.random.shuffle(all_pairs)

        for center, context in all_pairs:
            progress = step / max(total_steps - 1, 1)
            model.learning_rate = max(lr_min, lr_start * (1 - progress))

            neg_idxs = neg_sample(n_negative, exclude={center, context})
            loss, p, q = model.forward(center, context, neg_idxs)
            model.backward(center, context, neg_idxs, p, q)

            epoch_loss += loss
            step += 1

        avg_loss = epoch_loss / len(all_pairs)
        print(f"Epoch {epoch:2d}/{n_epochs} loss: {avg_loss:.5f} lr: {model.learning_rate:.5f}")

    return model