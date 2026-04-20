from collections import Counter
def build_vocabulary(corpus, max_vocab_size):
    """
    Returns: dict
    """
    vocab = Counter(word for sent in corpus for word in sent)
    top_items = dict(sorted(vocab.items(), key=lambda x: (-x[1], x[0]))[:max_vocab_size - 2])
    word2id = {'<pad>': 0, '<unk>': 1}
    word2id.update({word: i for i, (word, _) in enumerate(top_items.items(), start=2)})
    
    indexed_corpus = [
    [word2id.get(word, word2id['<unk>']) for word in sent]
    for sent in corpus]

    return {'vocab': word2id,'encoded': indexed_corpus}
    