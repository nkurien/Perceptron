import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np
from mlp import MLP
import sys
import os
from preprocessing import train_test_split, load_mnist

class DrawNumber:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Number")
        
        self.canvas_width = 280
        self.canvas_height = 280
        
        self.canvas = Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(pady=20)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_xy)
        
        self.button_clear = Button(self.master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(pady=10)
        
        self.button_predict = Button(self.master, text="Predict", command=self.predict_number)
        self.button_predict.pack(pady=10)
        
        self.label_result = Label(self.master, text="Draw a number and click Predict", font=("Arial", 16))
        self.label_result.pack(pady=20)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color="black")
        self.draw = ImageDraw.Draw(self.image)
        
        self.old_x = None
        self.old_y = None
        
        # Load the trained model
        self.model = self.load_model()

    def paint(self, event):
        paint_color = "white"
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=20, fill=paint_color, capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                           fill=paint_color, width=20, joint="curve")
        self.old_x = event.x
        self.old_y = event.y

    def reset_xy(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a number and click Predict")

    def predict_number(self):
        # Resize image to 28x28
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        
        # Convert image to numpy array and normalize
        img_array = np.array(img_resized).reshape(784, 1) / 255.0
        
        # Make prediction
        prediction = self.model.predict(img_array)
        
        self.label_result.config(text=f"Predicted number: {prediction}")

    def load_model(self):
        # Initialize the model
        input_size = 784
        hidden_layers = [128, 64]
        num_classes = 10
        model = MLP(input_size=input_size, num_classes=num_classes, hidden_layers=hidden_layers, activation='relu')
        
        # Load the trained weights
        try:
            # First, try to load the primary weights file
            try:
                with np.load('model_weights.npz') as data:
                    model.weights = [data[f'arr_{i}'] for i in range(len(data.files))]
                print("Loaded trained weights successfully from 'model_weights.npz'.")
            except FileNotFoundError:
                # If primary file is not found, try loading the backup file
                with np.load('model_backup.npz') as data:
                    model.weights = [data[f'arr_{i}'] for i in range(len(data.files))]
                print("Loaded trained weights successfully from 'model_backup.npz'.")
        except FileNotFoundError:
            print("Neither 'model_weights.npz' nor 'model_backup.npz' found. Using random weights.")
        except Exception as e:
            print(f"An error occurred while loading weights: {str(e)}. Using random weights.")
        
        return model

def save_model_weights(model, filename='model_weights.npz'):
    np.savez(filename, *model.weights)
    print(f"Model weights saved to {filename}")

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath('__file__'))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)
# Adjust this path to match your folder structure
data_path = os.path.join(current_dir, 'data', 'mnist')

# Load all training data
X_all, y_all = load_mnist(data_path, kind='train')


print(f"All data shape: {X_all.shape}")
print(f"All labels shape: {y_all.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, seed=2108)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, seed=2108)

def one_hot_encode(y, num_classes):
    encoded = np.zeros((num_classes, y.shape[0]))
    encoded[y, np.arange(y.shape[0])] = 1
    return encoded

# Initialize and train the perceptron
input_size = 784  # 28x28 pixels
num_classes = 10  # digits 0-9

# Ensure X_train and X_val are in the correct shape (features, samples) and normalized
X_train = X_train.T.astype(np.float32) / 255
X_val = X_val.T.astype(np.float32) / 255

# One-hot encode the labels
y_train_onehot = one_hot_encode(y_train, num_classes)
y_val_onehot = one_hot_encode(y_val, num_classes)

# Debugging, checking if data is normalised correctly...
print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_train_onehot sample: {y_train_onehot[:, 0]}")
print(f"Corresponding label: {np.argmax(y_train_onehot[:, 0])}")

# Now initialize and train the MLP
mlp = MLP(input_size=input_size, num_classes=num_classes, activation='relu')  # Input layer, two hidden layers, output layer
mlp.train(X_train, y_train_onehot, X_val, y_val_onehot, epochs=50)

save_model_weights(mlp, 'model_weights.npz')

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawNumber(root)
    root.mainloop()