import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import randint
import numpy as np

df_small = pd.read_csv(
    '/workspaces/ECON-424/Predic_2/424_F2025_PC2_small_train_data_v1.csv'
    )

for column in df_small.drop('price', axis=1).columns:
    plt.figure()
    plt.scatter(df_small['price'], df_small[column])
    plt.xlabel('price')
    plt.ylabel(column)
    plt.title('Scatter plot of Price vs {}'.format(column))

    plt.savefig("scatter_small_{}.png".format(column))

print("Scatter Plots Created")
print("\n\n")

def make_hist(df):
    for column in df.columns:
        plt.figure()
        plt.hist(df[column], bins=30, edgecolor='k')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title('Histogram of {}'.format(column))

        plt.savefig("hist_small_{}.png".format(column))

make_hist(df_small)
print("Histograms Created")
print("\n\n")

def preprocess_data(df):
    df['price'] = np.log(df['price'])
    df['age'] = 2025 - df['year']
    df['is_old'] = (df['year'] < 2010).astype(int)
    df['is_high_mileage'] = (df['mileage'] > 100000).astype(int)
    df['avg_fuel_efficiency'] = (df['city_fuel_economy'] + 
                                 df['highway_fuel_economy']) / 2
    df['mileage/year'] = df['mileage'] / df['age']
    df['engine/hp'] = df['engine_displacement'] / df['horsepower']
    return df
    


def create_binned_and_encoded_features(df, col_name, bins, labels):
    '''
    Creates a binned feature from a continuous column and then one-hot encodes it.
    The function returns a new DataFrame with the original columns plus
    the newly created one-hot encoded features

    create_binned_and_encoded_features:\
     DataFrame Str List (ListofStr) -> DataFrame
    
    '''
    df_copy = df.copy()
    
    binned_col_name = '{}_binned'.format(col_name)
    df_copy[binned_col_name] = pd.cut(df_copy[col_name],
                                      bins=bins,
                                      labels=labels)
    
    col_to_encode = df_copy[[binned_col_name]]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = pd.DataFrame(encoder.fit_transform(col_to_encode),
                                    columns=encoder.get_feature_names_out(
                                        [binned_col_name]))
    df_final = pd.concat([df_copy.reset_index(drop=True),
                          encoded_features.reset_index(drop=True)], axis=1)

    return df_final

df_small = preprocess_data(df_small)

df_final = create_binned_and_encoded_features(df_small,
                                   'horsepower',
                                   bins=[0,100, 150, 200, 300,400,float('inf')],
                                   labels=['Very Low', 'Low', 'Medium', 'High',
                                           'Very High','Super high'])

df_final = create_binned_and_encoded_features(df_final,
                                              'mileage',
                                                bins=[0,50000, 100000, 150000, float('inf')],
                                                labels=['Low', 'Medium', 'High', 'Very High'])
df_final = df_final.drop(['horsepower_binned',
                          'mileage_binned'], axis=1)
x_vals = df_final.drop(['price',
                        'city_fuel_economy',
                        'highway_fuel_economy'],axis=1)
y_vals = df_final['price']
x_train, x_test, y_train, y_test = \
    train_test_split(x_vals,y_vals,test_size=0.20)


## Models

##  base line
## linear regression
model_lin = LinearRegression()
model_lin.fit(x_train,y_train)
y_pred_lin = model_lin.predict(x_test)
r2_lin = r2_score(y_test,y_pred_lin)
print("R2 for linear model:", r2_lin)
print("\n")

## Lasso CV
model_lasso = LassoCV()
model_lasso.fit(x_train,y_train)
y_pred_lasso = model_lasso.predict(x_test)
r2_lasso = r2_score(y_test,y_pred_lasso)
print("R2 for Lasso model:", r2_lasso)
print("\n")

## Ridge CV
model_ridge = RidgeCV()
model_ridge.fit(x_train,y_train)
y_pred_ridge = model_ridge.predict(x_test)
r2_ridge = r2_score(y_test,y_pred_ridge)
print("R2 for Ridge model:", r2_ridge)
print("\n") 

## Decision Tree with Randomized Search CV
param_dist = {
    'max_depth': randint(1, 35),
    'min_samples_split': randint(1, 35),
    'min_samples_leaf': randint(2, 35)
}
model_tree = RandomizedSearchCV(
    estimator=DecisionTreeRegressor(random_state=37),
    param_distributions=param_dist,
    n_iter=100,
    cv=10,
    random_state=37,
    scoring = 'r2',
    n_jobs=-1
)
model_tree.fit(x_train,y_train)
y_pred_tree = model_tree.predict(x_test)
r2_tree = r2_score(y_test,y_pred_tree)
mae_tree = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_tree))
print("R2 for Decision Tree model:", r2_tree)
print("Best parameters for Decision Tree model:", model_tree.best_params_)
print('MAE for Decision Tree model:', mae_tree)
