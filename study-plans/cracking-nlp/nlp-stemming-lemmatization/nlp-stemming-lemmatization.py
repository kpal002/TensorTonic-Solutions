def stem_and_lemmatize(word, lemma_dict):
    """
    Returns: dict
    """
    
    output = dict.fromkeys(["stem","lemma"],word)
    
    # Stemming
    # 1. -ing, -ly, -ed: remove if remaining length >= 3
    for suf in ("ing", "ly", "ed","er"):
        if word.endswith(suf):
            stem = word[:-len(suf)]
            if len(stem) >= 3:
                output["stem"] = stem

    if word.endswith('s') and not word.endswith('ss'):

        stem = word[:-1]
        if len(stem) >=2:
            output["stem"] = stem

    if word.endswith('tion'):
        stem = word[:-4] + 'te'
        if len(stem) >=3:
            output["stem"] = stem
            

    #Lemmatization
    if word in lemma_dict.keys():
        output["lemma"] = lemma_dict[word]
    

    return output
        