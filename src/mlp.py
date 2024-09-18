import numpy as np

class MLP:
    def __init__(self, input_size, num_classes, hidden_layers=[128, 64]):
        layer_sizes = [input_size] + hidden_layers + [num_classes]
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]

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

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, self.activations[-1]) + b
            self.zs.append(z)
            self.activations.append(self.sigmoid(z))
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.zs[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, self.activations[-2].T)
        
        for l in range(2, len(self.layer_sizes)):
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_derivative(self.zs[-l])
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, self.activations[-l-1].T)
        return nabla_b, nabla_w

    def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01, batch_size=32, lambda_reg=0.01, verbose=True):
        n = X_train.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            X_train_shuffled = X_train[:, permutation]
            y_train_shuffled = y_train[:, permutation]
            
            for i in range(0, n, batch_size):
                X_batch = X_train_shuffled[:, i:i+batch_size]
                y_batch = y_train_shuffled[:, i:i+batch_size]
                
                self.forward(X_batch)
                nabla_b, nabla_w = self.backward(X_batch, y_batch)
                
                #possibly redundant control flow
                # Update with L2 regularization
                self.weights = [(1 - learning_rate * lambda_reg / n) * w - learning_rate * nw 
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_b)]
            
            train_accuracy = self.evaluate(X_train, y_train)
            if verbose and (epoch + 1) % 10 == 0:
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