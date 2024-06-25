import numpy as np
import matplotlib.pyplot as plt
from preprocessing import train_test_split

def load_mnist(path, kind='train'):
    """Load MNIST data from path"""
    labels_path = f"{path}/{kind}-labels-idx1-ubyte/{kind}-labels-idx1-ubyte"
    images_path = f"{path}/{kind}-images-idx3-ubyte/{kind}-images-idx3-ubyte"
    
    print(f"Labels path: {labels_path}")
    print(f"Images path: {images_path}")

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels

# Adjust this path to match your folder structure
data_path = '../data/mnist'

# Load training data
X_train, y_train = load_mnist(data_path, kind='train')

# Load test data
X_test, y_test = load_mnist(data_path, kind='t10k')

# Normalize the data
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, seed=420)


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

# Initialize and train the perceptron
input_size = 784  # 28x28 pixels
num_classes = 10  # digits 0-9
perceptron = Perceptron(input_size, num_classes)
perceptron.train(X_train, y_train)

# Evaluate on test set
predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test accuracy: {accuracy}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {predictions[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()