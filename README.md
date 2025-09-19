***

# Car Price Prediction with a Decision Tree Regressor

### Project Overview
This project applies machine learning techniques to predict the price of used cars based on various features such as mileage, horsepower, and age. The goal is to build a robust predictive model and demonstrate a complete machine learning workflow, from data analysis to model evaluation. This project was developed as part of a prediction competition for my ECON 424 - Machine Learning class at the University of Waterloo.

A key focus of this project was to achieve the highest possible accuracy using a single, interpretable model, such as a Decision Tree Regressor. This project also explores other models, including penalized regression, to showcase an understanding of model trade-offs and is a valued skill in fields like fintech.

### Methodology
The project follows a standard machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):** The dataset was analyzed to understand the distribution of key features. Histograms and scatter plots were used to identify data characteristics such as skewness in price and mileage, and multimodality in horsepower.

2.  **Data Preprocessing and Feature Engineering:**
    * **Data Cleaning:** Handled missing values and removed duplicates to ensure data quality.
    * **Feature Creation:** Engineered new features to provide better predictive signals to the model, including car age (`2025 - year`), `mileage_per_year`, and `avg_fuel_efficiency`.
    * **Data Transformation:** A logarithmic transformation was applied to the highly skewed `price` and `mileage` features to normalize their distribution, which helps the model learn more effectively.
    * **Feature Binning & Encoding:** Continuous features like `horsepower` and `mileage` were transformed into distinct categories using `pandas.cut()`, and then converted into a numerical format using `OneHotEncoder`. This method provides the Decision Tree with clearer signals for making splits.

3.  **Model Building & Evaluation:**
    * **Baseline Model:** Started wutg a sunoke Kubear Regressuib nidek ti establish a performance benchmark, This is a crucial step in econometrics to understand the base level of a model's explanatory power.
    * **Advanced Models:** Ran penaluzed regression models (LassoCV, RidgeCV) and a Decision Tree Regressor to see if they could improve upon the baseline.
    * **Hyper Parameter Tuning:** For the Decision Tree, **Randomized Search Cross-Validation(`RandomizedSearchCV`)** was used to efficiently sample the parameter space and find the optimal combination of parameteres for the best possible performance for a single tree.  

### Results
The final model was evaluated on unseen test data to assess its performance. The results are based on the best model found by the grid search.

* **Optimal Hyperparameters:**
    * `max_depth`: 14
    * `min_samples_split`: 11
    * `min_samples_leaf`: 13

* **Model Performance Metrics:**
    * **R-squared ($R^2$) Score:** 57%
    * **Mean Absolute Error (MAE):** $1380.93

### Visual Model Evaluation
The following plot shows the model's performance by comparing its predicted prices against the actual prices. Points closer to the red line indicate more accurate predictions. This visual is an intuitive way to understand the model's error distribution.

### Technologies Used
* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib


***