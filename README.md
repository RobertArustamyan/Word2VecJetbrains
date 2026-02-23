# Word2Vec — Skip-Gram with Negative Sampling with Numpy

## Project Structure

```
word2vec/
├── word2vec.py      — Word2Vec class: embedding matrices, forward pass, backward pass, nearest neighbours
├── corpus.py        — Text loading, tokenization, vocabulary building, subsampling
├── trainer.py       — Training loop, skip-gram pair generation, learning rate decay
├── utils.py         — Sigmoid, cosine similarity, negative sampler
├── train.py         — Entry point: wires everything together and prints nearest neighbours
└── data/
    └── text8        — Training corpus
```

## Data

Download text8 from http://mattmahoney.net/dc/text8.zip and extract into `data/`.

```bash
wget http://mattmahoney.net/dc/text8.zip -P data/
unzip data/text8.zip -d data/
```

## File Descriptions

**`corpus.py`** — Loads raw text from disk, tokenizes it, builds the vocabulary with word-to-index and index-to-word mappings, converts tokens to integer ids, and subsamples frequent words.

**`utils.py`** — Numerically stable sigmoid function, cosine similarity between two vectors, and a negative sampler that draws word indices proportional to `frequency^0.75`.

**`word2vec.py`** — Holds the two embedding matrices `W_in` and `W_out`. Implements the SGNS forward pass (loss and probabilities) and backward pass (gradients and SGD update). Also provides nearest-neighbour lookup by cosine similarity.

**`trainer.py`** — Generates (center, context) skip-gram pairs using a variable-length window, then runs the training loop with linear learning rate decay across all epochs.

**`train.py`** — Entry point. Loads and preprocesses data, calls the trainer, and prints nearest neighbours for a set of probe words.