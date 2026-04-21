from collections import Counter

def bag_of_words(corpus):
    """
    Returns: dict
    """
    vocab_words = sorted(set(word for doc in corpus for word in doc))
    vocab = {word: i for i, word in enumerate(vocab_words)}

    vectors = []
    for doc in corpus:
        vec = [0] * len(vocab)
        counts = Counter(doc)
        for word, count in counts.items():
            vec[vocab[word]] = count
        vectors.append(vec)

        
    return {"vocab": vocab, "vectors": vectors}