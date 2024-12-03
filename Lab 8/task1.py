# Task 1: Iris Dataset - Modify the Neural Network

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to preprocess dataset (standardization and one-hot encoding)
def preprocess_data(X, y):
    X = StandardScaler().fit_transform(X)
    y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    return X, y_encoded

# Function to build and compile a model
def build_model(input_shape, hidden_layers, output_units):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_units, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to visualize training results
def plot_training_results(history, title):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to generate and plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, labels, dataset_name):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        cmap=plt.cm.Blues, ax=plt.gca()
    )
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.show()

# Load and preprocess Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris, y_iris = preprocess_data(X_iris, y_iris)

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Build and train the original model
original_model = build_model(input_shape=X_train_iris.shape[1], hidden_layers=[8], output_units=3)
history_original = original_model.fit(
    X_train_iris, y_train_iris, validation_split=0.2, epochs=50, batch_size=16, verbose=1
)

# Build and train the modified model
modified_model = build_model(input_shape=X_train_iris.shape[1], hidden_layers=[8, 16], output_units=3)
history_modified = modified_model.fit(
    X_train_iris, y_train_iris, validation_split=0.2, epochs=50, batch_size=16, verbose=1
)

# Visualization for Task 1
plot_training_results(history_original, "Original Model (Iris Dataset)")
plot_training_results(history_modified, "Modified Model (Iris Dataset)")

# Confusion Matrices for Task 1
plot_confusion_matrix(
    original_model, X_test_iris, y_test_iris, labels=iris.target_names, dataset_name="Iris Dataset (Original Model)"
)
plot_confusion_matrix(
    modified_model, X_test_iris, y_test_iris, labels=iris.target_names, dataset_name="Iris Dataset (Modified Model)"
)