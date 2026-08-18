# Classifier Comparisons

A Python implementation comparing machine learning classification models to predict categorical outcomes.

## Features

- **Model:** Comparing a Decision Tree model against a Gaussian Naive Bayes model.
- **Error Measure:** Classification scoring through confusion matrices.

## Technologies Used

- **Language:** Python 3.8+
- **Data Manipulation:** `pandas` (file reading)
- **Machine Learning:** `scikit-learn` (feature scaling, data encoding, dataset splits, and model fitting)
- **Data Visualization:** `matplotlib` (graphing confusion matrices)

## Dataset

This exercise uses the **Classification Lab Data** from the `ClassificationLabData.csv` file.

## Technical Notes

### 1. Preprocessing Tools Explained

The classification pipeline cleans and prepares raw columns using statistical transformation methods before passing data to the models:

*   **One-Hot Encoding:** Turns text-based categories into grids of binary ones and zeros. This lets mathematical models understand labels like gender or marital status.
*   **Standard Scaler (Z-Score):** Adjusts numeric columns so they are on the same scale with an average of zero and a spread of one. This makes it so physical units (like turning years of age and hours worked into matching scales / massive numbers) do not accidentally overwhelm smaller numbers during model training.
*   **Confusion Matrix:** A 4x4 grid that tracks exactly where the model guessed correctly and where it made mistakes. It shows the true answers on one side and the model's predictions on the other side.