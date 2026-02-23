import numpy as np

from utils import sigmoid


class Word2Vec:
    def __init__(self, vocab_size, embedding_dimension, n_negative, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.n_negative = n_negative
        self.learning_rate = learning_rate

        # initializing matrices with Xavier initialization
        interval_bound = np.sqrt(6 / (embedding_dimension + vocab_size))
        self.W_in = np.random.uniform(-interval_bound, interval_bound, (vocab_size, embedding_dimension))
        self.W_out = np.random.uniform(-interval_bound, interval_bound, (vocab_size, embedding_dimension))

    def forward(self, center_idx, context_idx, neg_idxs):
        """compute loss and sigmoid probabilities for one training pair."""
        v_c = self.W_in[center_idx]
        v_o = self.W_out[context_idx]
        v_neg = self.W_out[neg_idxs]

        pos_score = np.dot(v_c, v_o)
        neg_scores = v_neg @ v_c

        p = sigmoid(pos_score)
        q = sigmoid(neg_scores)

        loss = -np.log(p + 1e-10) - np.sum(np.log(1 - q + 1e-10))
        return loss, p, q

    def backward(self, center_idx, context_idx, neg_idxs, p, q):
        """Compute gradients and updates the affected embedding rows"""
        v_c = self.W_in[center_idx]
        v_o = self.W_out[context_idx]
        v_neg = self.W_out[neg_idxs]

        grad_v_c = (p - 1) * v_o + q @ v_neg
        grad_v_o = (p - 1) * v_c
        grad_v_neg = np.outer(q, v_c)

        self.W_in[center_idx] -= self.learning_rate * grad_v_c
        self.W_out[context_idx] -= self.learning_rate * grad_v_o
        self.W_out[neg_idxs] -= self.learning_rate * grad_v_neg

    def most_similar(self, word_idx, idx2word, top_n):
        """Returns the top_n most similar words by cosine similarity in W_in."""
        query = self.W_in[word_idx]

        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        normed = self.W_in / norms
        query_normed = query / (np.linalg.norm(query) + 1e-10)

        sims = normed @ query_normed
        sims[word_idx] = -1

        top_indices = np.argsort(sims)[::-1][:top_n]
        return [(idx2word[i], float(sims[i])) for i in top_indices]