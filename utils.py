import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid that avoids overflow for large negative inputs."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def cosine_similarity(v1, v2):
    """Measures the cos between two vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def get_negative_sampler(freq_array: np.ndarray, power: float = 0.75):
    """Returns a sampler that draws negative word indices proportional to smoothed word frequency"""
    weights = freq_array ** power
    weights /= weights.sum()
    vocab_size = len(freq_array)

    def sample(k: int, exclude: set) -> np.ndarray:
        """Draws k negative indices, without indexes in the exclude set."""
        samples = []
        while len(samples) < k:
            draw = np.random.choice(vocab_size, size=k * 2, p=weights)
            for idx in draw:
                if idx not in exclude and len(samples) < k:
                    samples.append(idx)
        return np.array(samples, dtype=np.int32)

    return sample