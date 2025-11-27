# IntroMLCapstone

## Introduction
This is the capstone project for **Intro to Machine Learning**, focusing on predicting housing prices using both classical ML algorithms and literature-based machine learning methods.  
The project uses the Kaggle **House Prices: Advanced Regression Techniques** dataset.  
The goal is to implement classical ML models, reproduce methods from recent research papers, and compare their performance on the housing price prediction task.

---

## Repository Structure

This section includes every file and folder is described in detail.

```bash
IntroMLCapstone/
│
├── data/
│ ├── X_processed.npz
│ ├── X_test_processed.npz
│ ├── data_description.txt
│ ├── knn_val_predictions.npy
│ ├── lr_val_predictions.npy
│ ├── nn_metrics.json
│ ├── nn_val_predictions.npy
│ ├── paper1_xgb_metrics.json
│ ├── paper1_xgb_test_predictions.npy
│ ├── paper1_xgb_val_predictions.npy
│ ├── paper2_rf_val_predictions.npy
│ ├── paper2_xgb_val_predictions.npy
│ ├── test.csv
│ ├── train.csv
│ ├── y.csv
│ └── y_val.npy
│
├── preprocessing/
│   └── preprocessing.ipynb
│       - Loads raw Kaggle dataset
│       - Handles missing values, encodes categorical variables
│       - Performs transformations, normalization, train/validation split
│       - Saves train/val arrays to the data/ folder
│
├── classic_models/
│   ├── linear_regression.ipynb
│   ├── knn_regressor.ipynb
│   └── neural_network_regressor.ipynb
│       - Implements baseline classic ML models
│       - Saves predictions into data/ for unified comparison
│
├── paper_models/
│   ├── paper1_xgboost.ipynb
│   └── paper2_bayesopt_rf_and_xgb.ipynb
│       - Reproduces methodology from 2024 research papers
│       - Includes Bayesian Optimization, XGBoost tuning, feature importance
│
├── results/
│ ├── all_models_metrics.ipynb
│ ├── best_model_actual_vs_pred.png
│ ├── best_model_residuals.png
│ ├── mae_comparison.png
│ ├── metrics_table.png
│ ├── model_comparison_metrics.csv
│ ├── r2_comparison.png
│ └── rmse_comparison.png
│       - Stores consolidated metrics, plots, and visual results
│
├── requirements.txt
│   - All libraries needed to run the project
│
└── README.md
    - Project overview and file descriptions
```

**Folder/File Descriptions:**

- `data/` – Preprocessed datasets, intermediate outputs, saved predictions, and raw CSV files.  
- `preprocessing/` – Notebook for cleaning, encoding, scaling, and preparing features.  
- `classic_models/` – Classic ML models (Linear Regression, KNN, Neural Network).  
- `paper_models/` – Literature-based models (Paper 1 XGBoost, Paper 2 Bayesian-Optimized models).  
- `results/` – Evaluation results, metrics tables, and visualizations for all models.  
- `README.md` – Project documentation (this file).  
- `requirements.txt` – Python package dependencies.

## Project Objective

To compare traditional machine learning algorithms with state-of-the-art literature-based models for the task of housing price prediction, and determine whether advanced models significantly outperform classical techniques.

This includes:

1. Full data preprocessing

2. Implementing 3 classical ML models

3. Reproducing methods from 2 recent research papers

4. Evaluating all models on the same validation split

5. Comparing performance using RMSE, MAE, MSE, R²

6. Visualizing predictions and residuals


## Models Implemented

### Classic ML Models
1. **Linear Regression** – Baseline regression model.  
2. **KNN Regressor** – K-nearest neighbors regression.  
3. **Neural Network Regressor** – MLPRegressor from scikit-learn.

### Literature-Based Models
1. **Paper 1: XGBoost** – From “An Optimal House Price Prediction Algorithm: XGBoost” (2024, arXiv)  
   - Includes preprocessing, hyperparameter tuning, and feature importance.  
2. **Paper 2: Bayesian-Optimized Random Forest** – From “House Price Prediction with Optimistic Machine Learning Methods Using Bayesian Optimization” (2024, SCITEPRESS)  
   - Implements Bayesian Optimization for hyperparameter tuning.  

All models are trained and evaluated using the Kaggle House Prices dataset. Performance metrics include **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**. Visualizations such as residual histograms and predicted vs. actual plots are included in the `results/` folder.

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run preprocessing:
```bash
python preprocessing/preprocessing.ipynb
```

3. Run classic ML models:
```bash
python classic_models/linear_regression.ipynb
python classic_models/knn_regressor.ipynb
python classic_models/neural_network_regressor.ipynb
```

4. Run literature-based models:
```bash
python paper_models/paper1_xgboost.ipynb
python paper_models/paper2_bayesopt_rf.ipynb
```

5. View Results
```bash
results/all_models_metrics.ipynb
results/final_summary_dashboard.ipynb
```
   

## 📚 References

> **Paper 1: XGBoost**
   - Title: *“An Optimal House Price Prediction Algorithm: XGBoost”*  
   - Year: 2024  
   - Source: arXiv  
   - Link: [https://arxiv.org/abs/2402.04082](https://arxiv.org/abs/2402.04082)

> **Paper 2: Bayesian-Optimized Models**
   - Title: *“House Price Prediction with Optimistic Machine Learning Methods Using Bayesian Optimization”*  
   - Year: 2024  
   - Source: SCITEPRESS  
   - Link: [https://www.scitepress.org/Papers/2024/128254/128254.pdf](https://www.scitepress.org/Papers/2024/128254/128254.pdf)

> **Dataset**
   - Kaggle: *House Prices – Advanced Regression Techniques*  
   - Link: [https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)






