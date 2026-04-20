def pad_and_truncate(sequences, max_length, pad_value=0):
    """
    Returns: list[list[int]]
    """
    new_seq = []
    for seq in sequences:
        if len(seq) > max_length:
            new_seq.append(seq[:max_length])
        elif len(seq) < max_length:
            new_seq.append(seq + [pad_value]*(max_length - len(seq)))

        else:
            new_seq.append(seq)

    return new_seq