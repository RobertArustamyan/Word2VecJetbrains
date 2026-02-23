from corpus import load_text, tokenize, build_vocab
from trainer import train


def main():
    # Parameters for model training
    text_path = "data/text8"
    embed_dim = 100
    n_negative = 5
    window = 5
    n_epochs = 10
    lr_start = 0.025
    lr_min = 0.0001
    min_count = 10

    text = load_text(text_path, max_chars=2_000_000)
    tokens = tokenize(text)
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=min_count)

    print(f"Vocabulary size : {len(idx2word)}")
    print(f"Total tokens : {len(tokens)}")

    model = train(text, len(idx2word), word2idx, idx2word, freq_array, embed_dim, n_negative, window, n_epochs, lr_start, lr_min)

    print("Nearest neighbours")
    probe_words = ["learning", "neural", "model", "data", "word"] # will show top 5 nearest results for given set of words
    for word in probe_words:
        if word in word2idx:
            neighbours = model.most_similar(word2idx[word], idx2word, top_n=5)
            nb_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbours)
            print(f"{word:12s}: {nb_str}")


if __name__ == "__main__":
    main()