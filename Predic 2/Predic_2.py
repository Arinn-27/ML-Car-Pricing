import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from scipy.stats import randint
import numpy as np

df_large = pd.read_csv('/Predic 2/424_F2025_PC2_large_train_data_v1.csv')
df_small = pd.read_csv('/Predic 2/424_F2025_PC2_small_train_data_v1.csv')
df_medium = pd.read_csv('/Predic 2/424_F2025_PC2_medium_train_data_v1.csv')

df_test = pd.read_csv('/Predic 2/424_F2025_PC2_test_without_response_variable_data_v1.csv')

df_large = df_large[df_large['mileage']<=275000]

df_combined = pd.concat([df_small, df_large,df_medium])
df_combined= df_combined[df_combined['mileage'] !=df_combined['mileage'].max()]

df_combined['price'] = np.log(df_combined['price'])

df_combined= df_combined[df_combined['mileage'] !=df_combined['mileage'].max()]
x_vals = df_combined.drop('price',axis=1)
y_vals = df_combined['price']

x_train, x_test, y_train, y_test = \
    train_test_split(x_vals,y_vals,test_size=0.25)

# models 
    
param_distributions = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}
model_tree = RandomizedSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=10,
    scoring='r2',
    random_state=42
)
model_tree.fit(x_train,y_train)
predic_tree = model_tree.predict(x_test)
r2_tree = r2_score(y_test,predic_tree)
print('Model Score is: {0} \
    \n R2 of the model is: {1}'.format(model_tree.best_score_,r2_tree))
