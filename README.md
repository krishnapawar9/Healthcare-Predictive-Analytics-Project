# Healthcare-Predictive-Analytics-Project

## Overview

This project focuses on building a predictive model to diagnose diabetes using a dataset containing various health metrics. The goal is to develop a reliable machine learning model that can accurately predict the onset of diabetes based on diagnostic measurements. The project follows a complete data science pipeline, including data exploration, preprocessing, model training, evaluation, and saving the final model.

## Repository Contents

This repository contains all the necessary files to understand and reproduce the project:

  * **`Healthcare Predictive Analytics Project.ipynb`**: The main Jupyter Notebook that documents the entire workflow. It includes data loading, exploratory data analysis (EDA), data cleaning, feature engineering, model training with multiple algorithms (Logistic Regression, Random Forest, and Gradient Boosting), hyperparameter tuning, and model evaluation.
  * **`diabetes.xls`**: The dataset used for this project. It contains diagnostic information for several patients, which is used to train and test the predictive models.
  * **`best_diabetes_model.pkl`**: The final, best-performing machine learning model saved in a `.pkl` format. Based on the notebook, this is a **Gradient Boosting Classifier** that was chosen for its superior performance in predicting diabetes.
  * **`scaler.pkl`**: The `StandardScaler` object used to preprocess the numerical features of the dataset. This is essential for ensuring that the model receives scaled data for accurate predictions.
  * **`correlation_heatmap.png`**: A visualization showing the correlation matrix between different features in the dataset. This helps in understanding the relationships between variables.
  * **`feature_boxplots.png`**: Box plots for each feature, useful for identifying data distribution, outliers, and potential issues in the dataset.
  * **`feature_distributions.png`**: Histograms or distribution plots of the features, providing a visual representation of how the data for each variable is spread.
  * **`roc_curve.png`**: The Receiver Operating Characteristic (ROC) curve plot, which is a key metric for evaluating the performance of the classification models, especially for imbalanced datasets.

## Project Steps and Methodology

1.  **Data Loading and Exploration**
2.  **Exploratory Data Analysis (EDA)**
3.  **Data Preprocessing**
4.  **Model Training and Evaluation**
5.  **Best Model Selection**
6.  **Saving the Final Model**

## Requirements

To run the Jupyter Notebook and reproduce the analysis, you will need the following Python libraries:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`
  * `joblib`
  * `ipython`

You can install these libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib ipython
```

## How to Use

1.  Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/HEALTHCARE-PREDICTIVE-ANALYTICS-PROJECT.git
    cd HEALTHCARE-PREDICTIVE-ANALYTICS-PROJECT
    ```
2.  Ensure you have all the required libraries installed.
3.  Open the `Healthcare Predictive Analytics Project.ipynb` notebook in Jupyter or a compatible environment.
4.  Run all the cells in the notebook to see the complete data analysis, model training, and evaluation process.

## Conclusion

This project demonstrates a practical application of predictive analytics in healthcare. The resulting model can be used as a valuable tool to assist in the early diagnosis of diabetes. The saved model and scaler files allow for easy integration into a new system for making real-time predictions on new patient data.

## Author

  * **Krishna Pawar** - https://www.linkedin.com/in/krishna-pawar-842903230/
