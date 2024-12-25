
## **Lab#07: Feature Selection and PCA**

### **Objective**
To perform feature selection using ANOVA F-Value and apply Principal Component Analysis (PCA) to reduce the dimensionality of the Iris dataset.

### **Steps Included**

1. **Data Setup**:
   - Load the Iris dataset and explore its features.
2. **Feature Selection**:
   - Apply ANOVA F-Value to identify the top 2 most significant features contributing to class separation.
3. **Principal Component Analysis (PCA)**:
   - Reduce the dimensionality of the dataset to 2 components using PCA.
4. **Visualizations**:
   - Plot a 2D scatter plot of the PCA-transformed data to visualize class separations.
5. **Variance Analysis**:
   - Analyze the variance explained by the PCA components and their contribution to the dataset's variability.

### **Key Outputs**
- Top 2 features selected using ANOVA F-Value.
- PCA-transformed dataset visualized as a 2D scatter plot.
- Explained variance by PCA components.

### **How to Run**
- Copy the code into a Jupyter Notebook or Colab.
- Ensure required libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`) are installed.
- Run the notebook step by step.

---

## **General Notes**

- **Dataset**: The Iris dataset is a well-known dataset containing 150 samples of flowers, each with 4 numerical features.
- **Dependencies**: Install all dependencies using pip if not pre-installed.
  ```bash
  pip install numpy pandas matplotlib scikit-learn scipy
  ```
- **Interpretation**: Each task includes visualizations and print statements to aid understanding of the results.
