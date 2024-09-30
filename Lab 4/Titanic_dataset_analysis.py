# Step 1: Download the Dataset
from google.colab import files

# Upload kaggle.json file
files.upload()  # This will prompt you to upload the kaggle.json file


# Step 2: Import Libraries and Load Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
titanic_data = pd.read_csv('train.csv')
print(titanic_data.head())

# Visualizing the distribution of key features
sns.countplot(x='Pclass', data=titanic_data)
plt.title('Passenger Class Distribution')
plt.show()

sns.histplot(titanic_data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

sns.countplot(x='Sex', data=titanic_data)
plt.title('Gender Distribution')
plt.show()

# Checking for missing values
print(titanic_data.isnull().sum())

# Visualizing missing data
sns.heatmap(titanic_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Handling missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data.drop('Cabin', axis=1, inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Encoding categorical variables
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

# Standardizing numerical features
scaler = StandardScaler()
titanic_data[['Age', 'Fare']] = scaler.fit_transform(titanic_data[['Age', 'Fare']])

# Part 2: Implementing k-NN and Decision Trees

# Select features for model training
X = titanic_data[['Age', 'Fare']]  # Use only Age and Fare for training
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementing k-NN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# Implementing Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Evaluating the performance
def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, f1

# k-NN evaluation
knn_metrics = evaluate_model(knn_predictions, y_test)
print("k-NN Performance:")
print(f"Accuracy: {knn_metrics[0]}, Precision: {knn_metrics[1]}, Recall: {knn_metrics[2]}, F1-Score: {knn_metrics[3]}")

# Decision Tree evaluation
dt_metrics = evaluate_model(dt_predictions, y_test)
print("Decision Tree Performance:")
print(f"Accuracy: {dt_metrics[0]}, Precision: {dt_metrics[1]}, Recall: {dt_metrics[2]}, F1-Score: {dt_metrics[3]}")

# Part 3: Visualization

# Decision Boundaries
def plot_decision_boundaries(X, y, model, title):
    x1_min, x1_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    x2_min, x2_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', marker='o')
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.show()

# Use the same features for visualization
X_viz = X  # X already contains only 'Age' and 'Fare'
plot_decision_boundaries(X_viz, y, knn_model, 'k-NN Decision Boundaries')
plot_decision_boundaries(X_viz, y, dt_model, 'Decision Tree Decision Boundaries')

# Performance Visualization
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
knn_scores = list(knn_metrics)
dt_scores = list(dt_metrics)

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, knn_scores, width, label='k-NN')
bars2 = ax.bar(x + width/2, dt_scores, width, label='Decision Tree')

# Adding labels and title
ax.set_xlabel('Metrics')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()
