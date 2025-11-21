# IntroMLCapstone

## Introduction
This is the capstone project for **Intro to Machine Learning**, focusing on predicting housing prices using both classical ML algorithms and literature-based machine learning methods.  
The project uses the Kaggle **House Prices: Advanced Regression Techniques** dataset.  
The goal is to implement classical ML models, reproduce methods from recent research papers, and compare their performance on the housing price prediction task.

---

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

## References

> Paper 1: “An Optimal House Price Prediction Algorithm: XGBoost” (2024, arXiv)

> Paper 2: “House Price Prediction with Optimistic Machine Learning Methods Using Bayesian Optimization” (2024, SCITEPRESS)

> Kaggle Dataset: House Prices: Advanced Regression Techniques


