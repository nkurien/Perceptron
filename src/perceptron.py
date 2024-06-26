import numpy as np
import matplotlib.pyplot as plt



class Perceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
    
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def train(self, X, y, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            # Forward pass
            scores = self.forward(X)
            
            # Compute loss
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            correct_logprobs = -np.log(probs[range(len(X)), y])
            loss = np.sum(correct_logprobs) / len(X)
            
            # Backward pass
            dscores = probs
            dscores[range(len(X)), y] -= 1
            dscores /= len(X)
            
            # Update parameters
            self.weights -= learning_rate * np.dot(X.T, dscores)
            self.bias -= learning_rate * np.sum(dscores, axis=0, keepdims=True)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)
