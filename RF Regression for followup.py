"""

Title: RF Regressor for FollowUp Property Prediction
- Created: 2020.05.06
- Updated: 2020.10.22
- Author: Kyung Min, Lee

Learned from
- "Chapter 2 of Hands-on Machine Learning Book"
- Sckit-Learn documents

"""

# import packages
import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sklearn
warnings.filterwarnings(action='ignore')
np.random.seed(777)
RND_SEED = 777


# Extract Datasets
# df = pd.read_excel("C:/Users/kmlee/Documents/Handong/Labatory/Nondestructive test/followup/datasets/gather_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
# df_ri = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
# df_ap = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='amp,phs', skiprows=0)
df_riap = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag,amp,phs', skiprows=0)

# dataset for real,imag dataset
#X = df_ri.iloc[:, 8:]; X = np.array(X);
#arr1 = np.linspace(2,7,7-2+1); arr2 = np.linspace(14,39,39-14+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,13,13-2+1); arr2 = np.linspace(20,39,39-20+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,19,19-2+1); arr2 = np.linspace(26,39,39-26+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,25,25-2+1); arr2 = np.linspace(32,39,39-32+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X);
#X = df_ri.iloc[:, 2:34]; X = np.array(X);

# dataset for amp,phs dataset
#X = df_ap.iloc[:, 8:]; X = np.array(X);
#arr1 = np.linspace(2,7,7-2+1); arr2 = np.linspace(14,39,39-14+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,13,13-2+1); arr2 = np.linspace(20,39,39-20+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,19,19-2+1); arr2 = np.linspace(26,39,39-26+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,25,25-2+1); arr2 = np.linspace(32,39,39-32+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X);
#X = df_ap.iloc[:, 2:34]; X = np.array(X)

# dataset for real,imag,amp,phs dataset
#X = df_riap.iloc[:, 14:]; X = np.array(X);
#arr1 = np.linspace(2,13,13-2+1); arr2 = np.linspace(26,77,77-26+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,25,25-2+1); arr2 = np.linspace(38,77,77-38+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,37,37-2+1); arr2 = np.linspace(50,77,77-50+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X);
#arr1 = np.linspace(2,49,49-2+1); arr2 = np.linspace(62,77,77-62+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X);
#X = df_riap.iloc[:, 2:66]; X = np.array(X);

X = df_riap.iloc[:, 2:];
X = np.array(X)

# train, test split(8:2)
from sklearn.model_selection import train_test_split
X = df_riap.iloc[:, 2:]
ys = df_riap.iloc[:,0]
el = df_riap.iloc[:,1]

X_train, X_test, ys_train, ys_test = train_test_split(X, ys, test_size=0.2, random_state=1)
X_train, X_test, el_train, el_test = train_test_split(X, elong, test_size=0.2, random_state=1)

rs_ys_rmse = []; rs_ys_nrmse = []; rs_ys_r2 = []; rs_ys_ar2 = [];
rs_el_rmse = []; rs_el_nrmse = []; rs_el_r2 = []; rs_el_ar2 = [];

K = 10
kf = sklearn.KFold(n_splits=K, shuffle=True, random_state=RND_SEED)

k = 1
# implication with 10 kfold cross validation
for tr_idx, ts_idx in kf.split(X):

    # train, test split
    X_tr, X_ts = X[tr_idx], X[ts_idx]
    ys_tr, ys_ts = ys[tr_idx], ys[ts_idx]
    el_tr, el_ts = el[tr_idx], el[ts_idx]

    mu_ys_ts = np.mean(ys_ts)
    mu_el_ts = np.mean(el_ts)

    normalizer = sklearn.StandardScaler()
    X_tr = normalizer.fit_transform(X_tr)
    X_ts = normalizer.transform(X_ts)

    print("\nYIELD STRESS\n")

    # RF regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    # Hyper Parameters Tuning: GridSearch CV

    param_grid = [
        {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
         'max_features': ['auto', 'sqrt', 'log2'],
         'min_samples_leaf': [1, 3, 5, 7, 9]
         }
    ]

    rf_reg = RandomForestRegressor(random_state=RND_SEED)

    grid_search = GridSearchCV(rf_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(X_tr, ys_tr)

    print("GridSearch Best Estimator:")
    print(grid_search.best_estimator_)
    print('\n')

    print("GridSearch Best Parameters:")
    print(grid_search.best_params_)
    print('\n')

    # Predictions
    ys_rf_final_model = grid_search.best_estimator_

# training section
# for tr_idx, ts_idx in kf.split(X):
#     ys_rf_train_predictions = ys_rf_final_model.predict(X_train_std) #* np.std(ys_train) + np.mean(ys_train)
#     ys_rf_train_mse = mean_squared_error(ys_train, ys_rf_train_predictions)
#     ys_rf_train_rmse = np.sqrt(ys_rf_train_mse)
#     ys_rf_train_nrmse = ys_rf_train_rmse / np.mean(ys_train)
#     print("yield stress rf train prediction rmse: ", ys_rf_train_rmse)
#     print("yield stress rf train prediction nrmse: ", ys_rf_train_nrmse)
#
# # test prediction
# ys_rf_test_predictions = ys_rf_final_model.predict(X_test_std) #* np.std(ys_train) + np.mean(ys_train)
# ys_rf_test_mse = mean_squared_error(ys_test,ys_rf_test_predictions)
# ys_rf_test_rmse = np.sqrt(ys_rf_test_mse)
# ys_rf_test_nrmse = ys_rf_test_rmse/np.mean(ys_test)
# print("yield stress rf test prediction rmse: ", ys_rf_test_rmse)
# print("yield stress rf test prediction nrmse: ", ys_rf_test_nrmse)

# Calculate adjusted R^2 scores
from sklearn.metrics import r2_score

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj

ys_rf_r2 = r2_score(y_true=ys_test, y_pred=ys_rf_test_predictions)
ys_rf_r2_adj = r2_adj_score(y_pred=ys_rf_test_predictions, y_true=ys_test, X=X_train)

# print("yield stress rf_r2__score: ", round(ys_rf_r2,4))
# print("yield stress rf_r2_adj_score: ", round(ys_rf_r2_adj,4))   # independent variable수가 많아서 분모가 -가 됨


# Elongation
print("\nELONGATION\n")

# SVR regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf_reg = RandomForestRegressor(random_state=1)
rf_reg.fit(X_train_std, el_train)
el_predictions_rf = rf_reg.predict(X_train_std) #* np.std(el_train) + np.mean(el_train)
rf_mse = mean_squared_error(el_train, el_predictions_rf)
rf_rmse = np.sqrt(rf_mse)
rf_nrmse = rf_rmse/np.mean(ys_train)
print("test rf_rmse: ", rf_rmse)
print('test rf_nrmse: ', rf_nrmse)

# Hyper Parameters Tuning: GridSearch CV
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'max_features': ['auto', 'sqrt', 'log2'],
     'min_samples_leaf': [1, 3, 5, 7, 9]
     }
]

rf_reg = RandomForestRegressor(random_state=1)

grid_search = GridSearchCV(rf_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train_std, el_train)

print("GridSearch Best Estimator:")
print(grid_search.best_estimator_)
print('\n')

print("GridSearch Best Parameters:")
print(grid_search.best_params_)
print('\n')

# Predictions
el_rf_final_model = grid_search.best_estimator_

# train prediction
# el_rf_train_predictions = el_rf_final_model.predict(X_train_std) #* np.std(el_train) + np.mean(el_train)
# el_rf_train_mse = mean_squared_error(el_train, el_rf_train_predictions)
# el_rf_train_rmse = np.sqrt(el_rf_train_mse)
# el_rf_train_nrmse = el_rf_train_rmse/np.mean(el_train)
# print("elongation rf train prediction rmse: ", el_rf_train_rmse)
# print("elongation rf train prediction nrmse: ", el_rf_train_nrmse)
#
# # test prediction
# el_rf_test_predictions = el_rf_final_model.predict(X_test_std) #* np.std(el_train) + np.mean(el_train)
# el_rf_test_mse = mean_squared_error(el_test,el_rf_test_predictions)
# el_rf_test_rmse = np.sqrt(el_rf_test_mse)
# el_rf_test_nrmse = el_rf_test_rmse/np.mean(el_test)
# print("elonagtion rf test prediction rmse: ", el_rf_test_rmse)
# print("elonagtion rf test prediction nrmse: ", el_rf_test_nrmse)

# Calculate adjusted R^2 scores
from sklearn.metrics import r2_score

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj

el_rf_r2 = r2_score(y_true=el_test, y_pred=el_rf_test_predictions)
el_rf_r2_adj = r2_adj_score(y_pred=el_rf_test_predictions, y_true=el_test, X=X_train)

# print("elongation rf_r2__score: ", round(el_rf_r2,4))
# print("elongation rf_r2_adj_score: ", round(el_rf_r2_adj,4))   # independent variable수가 많아서 분모가 -가 됨

