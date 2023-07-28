# Gender Classification Model - Machine Learning Project

## Explore the Gender Classification Model!
Click below to open the project in an interactive Google Colab notebook:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Timmapuram-Karthik/Gender-Classification-ML/blob/main/Gender%20Classification.ipynb)

## Overview:
The Gender Classification Model is a machine learning project that utilizes the Logistic Regression algorithm to predict the gender of individuals based on various features such as long hair, forehead width, forehead height, nose width, nose long, lips thin,	distance from nose to lip long. The project is built using Python programming language and scikit-learn library.

## Dataset:

The model is trained on the "gender_classification_v7.csv" dataset, which contains labeled samples of individuals along with their corresponding genders.

## Model Performance:

The trained logistic regression model achieved an impressive accuracy of 97% on the testing data. This indicates that the model is capable of accurately predicting gender based on the provided features.

## Confusion Matrix:

The confusion matrix provides a detailed summary of the model's predictions and is visualized as a heatmap. It allows easy identification of true positives, true negatives, false positives, and false negatives.

|                | Predicted: Male (0) | Predicted: Female (1) |
|----------------|---------------------|-----------------------|
| **Actual: Male (0)** |       727           |          24           |
| **Actual: Female (1)** |        21           |          729          |

## Classification Report:

The classification report presents precision, recall, F1-score, and support for each class (male and female). It provides a comprehensive view of the model's performance for both classes.

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.96    |  0.98  |   0.97   |   751   |
|   1   |   0.98    |  0.96  |   0.97   |   750   |
|-------|-----------|--------|----------|---------|
|  avg/total  |   0.97    |  0.97  |   0.97   |  1501   |

## Getting Started:

To run the Gender Classification Model locally, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have Python and the required libraries (scikit-learn, pandas, matplotlib, seaborn) installed.
3. Run the Jupyter Notebook containing the project code and analysis.

## Conclusion:

The Gender Classification Model is an effective application of machine learning to predict gender based on given features. By following the steps outlined in this project, one can easily build and evaluate a gender classification model. With further improvements and experimentation, the model's accuracy and robustness can be enhanced.
