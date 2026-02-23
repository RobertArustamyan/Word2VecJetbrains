from corpus import load_text, tokenize, build_vocab
from trainer import train


def main():
    # Parameters for model training
    TEXT_PATH  = "data/text8"
    EMBED_DIM  = 100
    N_NEGATIVE = 5
    WINDOW     = 5
    N_EPOCHS   = 10
    LR_START   = 0.025
    LR_MIN     = 0.0001
    MIN_COUNT  = 10

    text = load_text(TEXT_PATH, max_chars=2_000_000)
    tokens = tokenize(text)
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=MIN_COUNT)

    print(f"Vocabulary size : {len(idx2word)}")
    print(f"Total tokens : {len(tokens)}")

    model = train(text, len(idx2word), word2idx, idx2word, freq_array, EMBED_DIM, N_NEGATIVE, WINDOW, N_EPOCHS, LR_START, LR_MIN)

    print("Nearest neighbours")
    probe_words = ["learning", "neural", "model", "data", "word"] # will show top 5 nearest results for given set of words
    for word in probe_words:
        if word in word2idx:
            neighbours = model.most_similar(word2idx[word], idx2word, top_n=5)
            nb_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbours)
            print(f"{word:12s}: {nb_str}")


if __name__ == "__main__":
    main()