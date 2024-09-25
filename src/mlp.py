import numpy as np

class MLP:
    def __init__(self, input_size, num_classes, hidden_layers=[128, 64], activation='relu'):
        self.layer_sizes = [input_size] + hidden_layers + [num_classes]
        if activation == 'relu':
            # He initialization for ReLU
            self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        elif activation == 'sigmoid':
            # Xavier initialization for sigmoid
            self.weights = [np.random.randn(y, x) * np.sqrt(1.0/x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        else:
            raise ValueError("Unsupported activation function")
        
        self.biases = [np.zeros((y, 1)) for y in self.layer_sizes[1:]]
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        else:
            raise ValueError("Unsupported activation function")
    
    def relu(self, z):
        """
        ReLU activation function.
        """
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """
        Derivative of the ReLU function.
        """
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """
        Numerically stable sigmoid function.
        """
        # Clip the input to avoid overflow in exp
        z = np.clip(z, -709, 709)  # exp(-709) is close to the smallest positive float
        
        # Use logaddexp for numerical stability
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))

    def sigmoid_derivative(self, z):
        """
        Derivative of the sigmoid function.
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        """
        Performs a forward pass through the network, calculating the activations for each layer.

        Args:
            X: Input data matrix (num_examples x features).

        Returns:
            The output of the final layer (num_examples x num_classes) after applying the softmax function.
        """

        self.activations = [X]  # Store activations for each layer
        self.zs = []             # Store weighted sums before activation for each layer

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Calculate the weighted sum of the previous layer's activation and bias
            z = np.dot(w, self.activations[-1]) + b
            self.zs.append(z)

            # Apply activation function (softmax for output layer, otherwise chosen activation)
            if i == len(self.weights) - 1:
                self.activations.append(self.softmax(z))
            else:
                self.activations.append(self.activation(z))

        return self.activations[-1]  # Return the output of the final layer

    def backward(self, X, y):
        """
        Performs a backward pass through the network, calculating the gradients for weights and biases.

        Args:
            X: Input data matrix (num_examples x features).
            y: Target labels matrix (num_examples x num_classes).

        Returns:
            A tuple containing lists of gradients for weights and biases: (nabla_b, nabla_w).
        """

        m = X.shape[1]  # Number of examples in the batch

        # Calculate the error (difference between output and target labels) in the output layer
        delta = self.activations[-1] - y

        # Initialize lists to store gradients for weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Backpropagate through layers (starting from the output layer)
        # Output layer: calculate gradients for weights and biases
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / m
        nabla_w[-1] = np.dot(delta, self.activations[-2].T) / m

        # Hidden layers: backpropagate errors and calculate gradients
        for l in range(2, len(self.layer_sizes)):
            # Propagate error back using weight transpose from the next layer
            delta = np.dot(self.weights[-l + 1].T, delta)

            # Apply activation derivative to the error for the current layer
            delta *= self.activation_derivative(self.zs[-l])

            # Calculate gradients for weights and biases in the current layer
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / m
            nabla_w[-l] = np.dot(delta, self.activations[-l - 1].T) / m

        return nabla_b, nabla_w  # Return the gradients for weights and biases

    
    def train(self, X, y, X_val=None, y_val=None, epochs=200, learning_rate=0.01, batch_size=32, lambda_reg=0.01, verbose=True):
        """
        Trains the MLP model using stochastic gradient descent with momentum.

        Args:
            X: Input data matrix.
            y: Target labels matrix.
            X_val: Optional validation data matrix.
            y_val: Optional validation target labels matrix.
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimization.
            batch_size: Size of mini-batches for stochastic gradient descent.
            lambda_reg: Regularization parameter for L2 regularization.
            verbose: Whether to print training progress.
        """

        n = X.shape[1]  # Number of examples
        adjusted_learning_rate = 0.001 if self.activation == self.relu else learning_rate

        # Initialize momentum terms
        v_w = [np.zeros_like(w) for w in self.weights]
        v_b = [np.zeros_like(b) for b in self.biases]
        momentum = 0.9  # Momentum parameter

        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]

                self.forward(X_batch)
                nabla_b, nabla_w = self.backward(X_batch, y_batch)

                # Update weights and biases with momentum
                for j in range(len(self.weights)):
                    v_w[j] = momentum * v_w[j] + (1 - momentum) * nabla_w[j]
                    self.weights[j] = (1 - adjusted_learning_rate * lambda_reg / n) * self.weights[j] - adjusted_learning_rate * v_w[j]

                for j in range(len(self.biases)):
                    v_b[j] = momentum * v_b[j] + (1 - momentum) * nabla_b[j]
                    self.biases[j] = self.biases[j] - adjusted_learning_rate * v_b[j]

            if verbose and (epoch + 1) % 10 == 0:
                train_accuracy = self.evaluate(X, y)
                if X_val is not None and y_val is not None:
                    val_accuracy = self.evaluate(X_val, y_val)
                    print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {train_accuracy:.4f}")

    def predict(self, X):
        # Forward pass
        output = self.forward(X)
        # Return the index of the maximum value along the first axis
        return np.argmax(output, axis=0)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == np.argmax(y, axis=0))

# Usage:
# mlp = MLP([784, 128, 64, 10])  # Input layer, two hidden layers, output layer
# mlp.train(X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.1, batch_size=32)