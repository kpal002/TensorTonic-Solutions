import numpy as np

def embedding_similarity(query, candidates, metric="cosine", k=1):
    """
    Returns: list
    """
    q_arr = np.array(query, dtype=float)
    c_arr = np.array(candidates, dtype=float)

    if metric == "cosine":
        scores = c_arr @ q_arr / (np.linalg.norm(c_arr, axis=1) * np.linalg.norm(q_arr))

    elif metric == "dot":
        scores = c_arr @ q_arr

    else:
        scores = -np.linalg.norm(q_arr - c_arr, axis=1)


    top_2_indices = np.argsort(-scores, kind="stable")[:k]
    return top_2_indices
        
            
    