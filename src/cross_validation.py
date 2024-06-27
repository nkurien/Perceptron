import numpy as np

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
    model_class: Class of the model to be validated
    X: Features (should be a 2D numpy array)
    y: Labels (should be a 1D numpy array)
    n_splits: Number of splits for cross-validation
    **model_params: Parameters to initialize the model
    
    Returns:
    list of scores from each fold
    """
    folds = split_data(X, y, n_splits)
    scores = []
    
    for fold, (train_index, val_index) in enumerate(folds, 1):
        print(f"Fold {fold}/{n_splits}")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Initialize a new model for each fold
        model = model_class(**model_params)
        
        # Train the model
        model.train(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_val)
        score = np.mean(predictions == y_val)
        scores.append(score)
        
        print(f"Fold {fold} score: {score:.4f}")
    
    return scores

def print_cv_results(scores):
    """
    Print cross-validation results.
    
    Parameters:
    scores: List of scores from cross-validation
    """
     # Convert numpy floats to Python floats
    scores = [float(score) for score in scores]
    
    print("\nCross-validation results:")
    print(f"Scores: {(scores)}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")