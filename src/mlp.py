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
        self.activations = [X]
        self.zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, self.activations[-1]) + b
            self.zs.append(z)
            if i == len(self.weights) - 1:
                self.activations.append(self.softmax(z))
            else:
                self.activations.append(self.activation(z))
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]  # number of examples in the batch
        delta = self.activations[-1] - y  # error in the output layer
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Output layer
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / m
        nabla_w[-1] = np.dot(delta, self.activations[-2].T) / m
        
        # Hidden layers
        for l in range(2, len(self.layer_sizes)):
            delta = np.dot(self.weights[-l+1].T, delta)
            # Apply activation derivative for all hidden layers
            delta *= self.activation_derivative(self.zs[-l])
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / m
            nabla_w[-l] = np.dot(delta, self.activations[-l-1].T) / m
        
        return nabla_b, nabla_w

    
    def train(self, X, y, X_val=None, y_val=None, epochs=200, learning_rate=0.01, batch_size=32, lambda_reg=0.01, verbose=True):
        n = X.shape[1]
        adjusted_learning_rate = 0.001 if self.activation == self.relu else learning_rate



        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                
                self.forward(X_batch)
                nabla_b, nabla_w = self.backward(X_batch, y_batch)
                
                # Apply updates
                self.weights = [(1 - adjusted_learning_rate * lambda_reg / n) * w - adjusted_learning_rate * nw 
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - adjusted_learning_rate * nb for b, nb in zip(self.biases, nabla_b)]
            
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