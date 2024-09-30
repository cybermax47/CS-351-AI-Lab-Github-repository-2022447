# README.txt

## Titanic Dataset Analysis - Lab 4

### Overview

This project focuses on analyzing the Titanic dataset, which is widely used in machine learning for binary classification tasks. The objective is to predict whether a passenger survived the Titanic disaster based on various features, such as age, gender, and fare. We perform exploratory data analysis, preprocess the data, and implement machine learning models (k-Nearest Neighbors and Decision Trees) to predict survival outcomes.

### Steps Involved

1. **Kaggle Setup and Data Acquisition**
   - Upload your `kaggle.json` file to authenticate access to Kaggle's API.
   - Download the Titanic dataset using the Kaggle API and extract the files.

2. **Import Libraries and Load Data**
   - Necessary libraries such as Pandas, NumPy, Seaborn, Matplotlib, and scikit-learn are used for data manipulation, visualization, and machine learning.
   - The dataset is loaded into a Pandas DataFrame for exploration and further processing.

3. **Data Visualization**
   - Visualize key features such as passenger class distribution, age distribution, and gender distribution.
   - Check for missing values and visualize them using a heatmap.

4. **Data Preprocessing**
   - Handle missing values by filling the missing Age values with the median, removing the Cabin column, and imputing missing Embarked values.
   - Encode categorical variables (Sex and Embarked) into numerical formats.
   - Standardize numerical features (Age and Fare) for better model performance.

5. **Machine Learning Models**
   - Implement two machine learning models: k-Nearest Neighbors (k-NN) and Decision Trees.
   - Select Age and Fare as features and the Survived column as the target variable.
   - Split the dataset into training and testing sets for model evaluation.

6. **Model Evaluation**
   - Evaluate both models using performance metrics such as accuracy, precision, recall, and F1-score.
   - Compare the performance of both models.

7. **Visualization of Decision Boundaries**
   - Visualize how both k-NN and Decision Trees create decision boundaries to separate the classes.
   - Compare the two models based on their ability to differentiate between passengers who survived and those who did not.

8. **Performance Comparison**
   - Create bar charts to compare the performance metrics of both k-NN and Decision Tree models.

### Conclusion

This project presents a complete workflow for Titanic dataset analysis, from data preprocessing to machine learning model implementation. By following these steps, you can gain insights into passenger survival patterns and build predictive models to estimate survival chances.

### Requirements

- Python 3.x
- Libraries: 
  - Pandas
  - NumPy
  - Seaborn
  - Matplotlib
  - scikit-learn
  - Kaggle API

### How to Run

1. Clone the repository from GitHub.
2. Install the required libraries mentioned above.
3. Set up Kaggle API by uploading your `kaggle.json` file.
4. Run the notebook to download the Titanic dataset, preprocess the data, and train machine learning models.
5. Evaluate the models' performance and visualize decision boundaries.

### Author
This project was completed as part of a lab task for Introduction to AI course. Instructor: Usama Janjua 