import numpy as np
from mlp import MLP


def split_data(X, y, n_splits):
    """
    Split data into n_splits folds.
    
    Parameters:
    X: Features (should be a 2D numpy array)
    y: Labels (should be a 1D numpy array)
    n_splits: Number of splits for cross-validation
    
    Returns:
    List of (train_indices, val_indices) tuples
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // n_splits
    folds = []
    
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else len(X)
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, val_indices))
    
    return folds


def cross_validate(model_class, X, y, n_splits=5, **model_params):
    """
    Perform cross-validation on a given model.
    
    Parameters:
    model_class: Class of the model to be validated (e.g., MLP, Perceptron)
    X: Features (numpy array of shape [samples, features])
    y: Labels (numpy array of shape [samples,])
    n_splits: Number of splits for cross-validation
    **model_params: Parameters to initialize the model
    
    Returns:
    list of scores from each fold
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    fold_size = len(X) // n_splits
    scores = []

    for i in range(n_splits):
        start = i * fold_size
        end = len(X) if i == n_splits - 1 else (i + 1) * fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Initialize model
        model = model_class(input_size=X.shape[1], num_classes=np.unique(y).size, **model_params)
        
        # Special handling for MLP one-hot encoding
        
        if model_class.__name__ == 'MLP':
            y_train_encoded = np.eye(np.unique(y).size)[y_train]
            y_val_encoded = np.eye(np.unique(y).size)[y_val]
            model.train(X_train.T, y_train_encoded.T, X_val.T, y_val_encoded.T, epochs=200)
        else:
            model.train(X_train, y_train)

        # Evaluate the model
        if model_class.__name__ == 'MLP':
            predictions = model.predict(X_val.T)
            # For MLP, we need to compare against the original labels
            val_accuracy = np.mean(predictions == np.argmax(y_val_encoded, axis=1))
        else:
            predictions = model.predict(X_val)
            val_accuracy = np.mean(predictions == y_val)

        scores.append(val_accuracy)
        print(f"Fold {i + 1}/{n_splits}, Validation Accuracy: {val_accuracy:.4f}")

    return scores

def print_cv_results(scores):
    """
    Print cross-validation results.
    
    Parameters:
    scores: List of scores from cross-validation
    """
    scores = [float(score) for score in scores]
    print("\nCross-validation results:")
    print(f"Scores: {scores}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")