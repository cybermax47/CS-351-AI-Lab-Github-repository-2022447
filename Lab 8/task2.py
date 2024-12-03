# Task 2: Wine Dataset - Build and Train a Neural Network

# Import necessary libraries (reuse imports from Task 1)
from sklearn.datasets import load_wine

# Load and preprocess Wine dataset
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
X_wine, y_wine = preprocess_data(X_wine, y_wine)

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42
)

# Build and train the model for the Wine dataset
wine_model = build_model(input_shape=X_train_wine.shape[1], hidden_layers=[8, 16], output_units=3)
history_wine = wine_model.fit(
    X_train_wine, y_train_wine, validation_split=0.2, epochs=50, batch_size=16, verbose=1
)

# Visualization for Task 2
plot_training_results(history_wine, "Model (Wine Dataset)")

# Confusion Matrix for Task 2
plot_confusion_matrix(
    wine_model, X_test_wine, y_test_wine, labels=wine.target_names, dataset_name="Wine Dataset"
)