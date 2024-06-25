import numpy as np

def shuffle_data(X, y, seed=None):
    """
    Shuffles the data samples along with their corresponding labels.

    Args:
        X (array-like): Data features, a 2D array of shape (n_samples, n_features).
        y (array-like): Data labels, a 1D array of shape (n_samples,).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: Shuffled data features and labels.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    return X[indices], y[indices]

def train_test_split(X, y, test_size=0.25, seed=None):  # Default test_size is now 0.25
    """
    Splits the dataset into training and test sets.

    Args:
        X (array-like): Data features, a 2D array of shape (n_samples, n_features).
        y (array-like): Data labels, a 1D array of shape (n_samples,).
        test_size (float or int, optional): If float, represents the proportion of 
            the dataset to include in the test split. If int, represents the 
            absolute number of test samples. Defaults to 0.25.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: Split data into training and test sets (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If test_size is not in the range 0 < test_size < 1 or not an int less than the number of samples.
    """
    # Shuffle is now done by default

    # Convert X and y to numpy arrays if they are not already
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input data cannot be empty.")

    if len(X) != len(y):
        raise ValueError("The number of samples in X and y must be equal.")
    X, y = shuffle_data(X, y, seed)

    num_samples = len(y)
    
    if isinstance(test_size, float):
        if 0.0 < test_size < 1.0:
            train_ratio = num_samples - int(num_samples * test_size)
        else:
            raise ValueError("test_size as a float must be in the range (0.0, 1.0)")

    elif isinstance(test_size, int):
        if 1 <= test_size < num_samples:
            train_ratio = num_samples - test_size
        else:
            raise ValueError("test_size as an int must be less than the number of samples")

    else:
        raise ValueError("Invalid test_size value")

    X_train, X_test = X[:train_ratio], X[train_ratio:]
    y_train, y_test = y[:train_ratio], y[train_ratio:]

    return X_train, X_test, y_train, y_test