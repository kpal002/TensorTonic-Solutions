import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """

    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train, dtype=int)
    predictions = []
    
    for i in range(len(X_test)):
        distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = y_train[top_k_indices]

        labels, counts = np.unique(top_k_labels, return_counts=True)

        predicted_label = labels[np.argmax(counts)]

        predictions.append(int(predicted_label))

    return predictions
        
