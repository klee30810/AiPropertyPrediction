"""

Title: XGB Regressor for FollowUp Property Prediction
- Created: 2020.05.05
- Updated: 2020.10.22
- Author: Kyung Min, Lee

Learned from
- "Chapter 2 of Hands-on Machine Learning Book"
- Scikit-Learn documents

"""

# import packages
import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import xlrd
import datetime
warnings.filterwarnings(action='ignore')
np.random.seed(1)


# Extract Datasets
#df = pd.read_excel("C:/Users/kmlee/Documents/Handong/Labatory/Nondestructive test/followup/datasets/gather_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
df = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)

# train, test split(8:2)
from sklearn.model_selection import train_test_split
# df = pd.read_excel("C:/Users/kmlee/Documents/Handong/Labatory/Nondestructive test/followup/datasets/gather_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
# df_ri = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
# df_ap = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='amp,phs', skiprows=0)
df_riap = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag,amp,phs', skiprows=0)

# X = df.iloc[:, [3,4,5,6,7,8,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]; X = np.array(X)
# X = df.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]; X = np.array(X)
# X = df.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,27,28,29,30,31,32,33,34,35,36,37,38,39]]; X = np.array(X)
# X = df.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33,34,35,36,37,38,39]]; X = np.array(X)
# X = df.iloc[:, 3:33]; X = np.array(X)
X = df_riap.iloc[:, 2:];
X = np.array(X)
#X = df.iloc[:, 2:]
ys = df_riap.iloc[:,0]
elong = df_riap.iloc[:,1]

X_train, X_test, ys_train, ys_test = train_test_split(X, ys, test_size=0.2, random_state=1)
X_train, X_test, el_train, el_test = train_test_split(X, elong, test_size=0.2, random_state=1)


# correlation heatmap
# df_corr = df.corr()
# fig, ax = plt.subplots( figsize=(7,7) )
# mask = np.zeros_like(df_corr, dtype=np.bool)
# sns.heatmap(df_corr,cmap = 'RdYlBu_r',
#             annot = True,   # 실제 값을 표시한다
#             mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
#             linewidths=.5,  # 경계면 실선으로 구분하기
#             cbar_kws={"shrink": .5}, # 컬러바 크기 절반으로 줄이기
#             vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
#            )
# plt.show()

# Feature Scaling => Standardization
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)
# ys_train = np.array(ys_train)
# ys_train = ys_train.reshape(-1,1)
# ys_train_std = data_pipeline.fit_transform(ys_train)
# ys_test = np.array(ys_test)
# ys_test = ys_test.reshape(-1,1)
# ys_test_std = data_pipeline.fit_transform(ys_test)
# el_train = np.array(el_train)
# el_train = el_train.reshape(-1,1)
# el_train_std = data_pipeline.fit_transform(el_train)
# el_test = np.array(el_test)
# el_test = el_test.reshape(-1,1)
# el_test_std = data_pipeline.fit_transform(el_test)

# Yield Stress
print("\nYIELD STRESS\n")

# SVR regression
from sklearn.metrics import mean_squared_error

xgb_reg = xgb.sklearn.XGBRegressor(random_state=1)
xgb_reg.fit(X_train_std, ys_train)
ys_predictions_xgb = xgb_reg.predict(X_train_std) #* np.std(ys_train) + np.mean(ys_train)
xgb_mse = mean_squared_error(ys_train, ys_predictions_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_nrmse = xgb_rmse/np.mean(ys_train)
print("test xgb_rmse: ", xgb_rmse)
print('test xgb_nrmse: ', xgb_nrmse)

# Hyper Parameters Tuning: GridSearch CV
from sklearn.model_selection import GridSearchCV

# range_gamma = [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
# range_alpha = [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
# range_lambda = [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
# range_estimator = [10**1, 10**2,10**3,10**4,10**5,10**6,10**7]
# range_depth = [2,3,4,5,6]
# range_lr = [10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]

# param_grid = [
#     {'booster': ['gbtree'], 'learning_rate': range_lr, 'gamma': range_gamma, 'max_depth': range_depth,
#      'n_estimators': range_estimator, 'reg_lambda': range_lambda, 'reg_alpha': range_alpha
#
#      }
# ]
param_grid = [
    {'gamma': [0.1, 0, 1], 'learning_rate': [0.001, 0.01, 0.1],
     'max_depth': [3, 5], 'n_estimators': [100, 500, 1000],
     'alpha': [0.001, 0.01, 0.1], 'lambda': [0.001, 0.01, 0.1]
     }
    ]

xgb_reg = xgb.sklearn.XGBRegressor(random_state=1)

grid_search = GridSearchCV(xgb_reg, param_grid, cv=3,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train_std, ys_train)

print("GridSearch Best Estimator:")
print(grid_search.best_estimator_)
print('\n')

print("GridSearch Best Parameters:")
print(grid_search.best_params_)
print('\n')

# Predictions
ys_xgb_final_model = grid_search.best_estimator_

# train prediction
ys_xgb_train_predictions = ys_xgb_final_model.predict(X_train_std) #* np.std(ys_train) + np.mean(ys_train)
ys_xgb_train_mse = mean_squared_error(ys_train, ys_xgb_train_predictions)
ys_xgb_train_rmse = np.sqrt(ys_xgb_train_mse)
ys_xgb_train_nrmse = ys_xgb_train_rmse/np.mean(ys_train)
print("yield stress xgb train prediction rmse: ", ys_xgb_train_rmse)
print("yield stress xgb train prediction nrmse: ", ys_xgb_train_nrmse)

# test prediction
ys_xgb_test_predictions = ys_xgb_final_model.predict(X_test_std) #* np.std(ys_train) + np.mean(ys_train)
ys_xgb_test_mse = mean_squared_error(ys_test,ys_xgb_test_predictions)
ys_xgb_test_rmse = np.sqrt(ys_xgb_test_mse)
ys_xgb_test_nrmse = ys_xgb_test_rmse/np.mean(ys_test)
print("yield stress xgb test prediction rmse: ", ys_xgb_test_rmse)
print("yield stress xgb test prediction nrmse: ", ys_xgb_test_nrmse)

# Calculate adjusted R^2 scores
from sklearn.metrics import r2_score

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj

ys_xgb_r2 = r2_score(y_true=ys_test, y_pred=ys_xgb_test_predictions)
ys_xgb_r2_adj = r2_adj_score(y_pred=ys_xgb_test_predictions, y_true=ys_test, X=X_train)

print("yield stress xgb_r2__score: ", round(ys_xgb_r2,4))
print("yield stress xgb_r2_adj_score: ", round(ys_xgb_r2_adj,4))   # independent variable수가 많아서 분모가 -가 됨


# Elongation
print("\nELONGATION\n")

# SVR regression
from sklearn.metrics import mean_squared_error

xgb_reg = xgb.sklearn.XGBRegressor(random_state=1)
xgb_reg.fit(X_train_std, el_train)
el_predictions_xgb = xgb_reg.predict(X_train_std) #* np.std(el_train) + np.mean(el_train)
xgb_mse = mean_squared_error(el_train, el_predictions_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_nrmse = xgb_rmse/np.mean(ys_train)
print("test xgb_rmse: ", xgb_rmse)
print("test xgb_nrmse: ", xgb_nrmse)

# Hyper Parameters Tuning: GridSearch CV
from sklearn.model_selection import GridSearchCV

# range_gamma = [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
# range_alpha = [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
# range_lambda = [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
# range_estimator = [10**1, 10**2,10**3,10**4,10**5,10**6,10**7]
# range_depth = [2,3,4,5,6]
# range_lr = [10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
#
#
# param_grid = [
#     {'booster': ['gbtree'], 'learning_rate': range_lr, 'gamma': range_gamma, 'max_depth': range_depth,
#      'n_estimators': range_estimator, 'reg_lambda': range_lambda, 'reg_alpha': range_alpha
#
#      }
# ]
param_grid = [
    {'gamma': [0.1, 0, 1], 'learning_rate': [0.001, 0.01, 0.1],
     'max_depth': [3, 5], 'n_estimators': [100, 500, 1000],
     'alpha': [0.001, 0.01, 0.1], 'lambda': [0.001, 0.01, 0.1]
      }]

xgb_reg = xgb.sklearn.XGBRegressor(random_state=1)

grid_search = GridSearchCV(xgb_reg, param_grid, cv=3,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train_std, el_train.ravel())

print("GridSearch Best Estimator:")
print(grid_search.best_estimator_)
print('\n')

print("GridSearch Best Parameters:")
print(grid_search.best_params_)
print('\n')

# Predictions
el_xgb_final_model = grid_search.best_estimator_

# train prediction
el_xgb_train_predictions = el_xgb_final_model.predict(X_train_std) #* np.std(el_train) + np.mean(el_train)
el_xgb_train_mse = mean_squared_error(el_train, el_xgb_train_predictions)
el_xgb_train_rmse = np.sqrt(el_xgb_train_mse)
el_xgb_train_nrmse = el_xgb_train_rmse/np.mean(el_train)
print("elongation xgb train prediction rmse: ", el_xgb_train_rmse)
print("elongation xgb train prediction nrmse: ", el_xgb_train_nrmse)

# test prediction
el_xgb_test_predictions = el_xgb_final_model.predict(X_test_std) #* np.std(el_train) + np.mean(el_train)
el_xgb_test_mse = mean_squared_error(el_test,el_xgb_test_predictions)
el_xgb_test_rmse = np.sqrt(el_xgb_test_mse)
el_xgb_test_nrmse = el_xgb_test_rmse / np.mean(el_test)
print("elonagtion xgb test prediction rmse: ", el_xgb_test_rmse)
print("elonagtion xgb test prediction nrmse: ", el_xgb_test_nrmse)

# Calculate adjusted R^2 scores
from sklearn.metrics import r2_score

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj

el_xgb_r2 = r2_score(y_true=el_test, y_pred=el_xgb_test_predictions)
el_xgb_r2_adj = r2_adj_score(y_pred=el_xgb_test_predictions, y_true=el_test, X=X_train)

print("elongation xgb_r2__score: ", round(el_xgb_r2,4))
print("elongation xgb_r2_adj_score: ", round(el_xgb_r2_adj,4))   # independent variable수가 많아서 분모가 -가 됨

rs_ys_rmse.append(rmse1)
rs_ys_nrmse.append(rmse1/mu_ys_ts)
rs_ys_r2.append(rsq1)
rs_ys_ar2.append(adjrsq1)

rs_el_rmse.append(rmse2)
rs_el_nrmse.append(rmse2/mu_el_ts)
rs_el_r2.append(rsq2)
rs_el_ar2.append(adjrsq2)

# save the result
result_filename = 'results/' + 'XGB_result.csv'
print(result_filename)
with open(result_filename, 'a') as f:
    f.write(str(datetime.datetime.now()))
    f.write(f"\nXGB Yield Stress GridSearch Best Estimator:\n")
    f.write(f'ys:rmse   mean={np.average(rs_ys_rmse)} std={np.std(rs_ys_rmse)}\n')
    f.write(f'ys:nrmse  mean={np.average(rs_ys_nrmse)} std={np.std(rs_ys_nrmse)}\n')
    f.write(f'ys:r2     mean={np.average(rs_ys_r2)} std={np.std(rs_ys_r2)}\n')
    f.write(f'ys:adj-r2 mean={np.average(rs_ys_ar2)} std={np.std(rs_ys_ar2)}\n')
    f.write(f'\n--------------------\n')
    f.write(f"\nElongation GridSearch Best Estimator:\n")
    f.write(f'el:rmse   mean={np.average(rs_el_rmse)} std={np.std(rs_el_rmse)}\n')
    f.write(f'el:nrmse  mean={np.average(rs_el_nrmse)} std={np.std(rs_el_nrmse)}\n')
    f.write(f'el:r2     mean={np.average(rs_el_r2)} std={np.std(rs_el_r2)}\n')
    f.write(f'el:adj-r2 mean={np.average(rs_el_ar2)} std={np.std(rs_el_ar2)}\n')
    f.write(f'\n\n')

# Save the predictions
pred_filename = 'pred/bagging_' + BASE_MODEL + '_preds.csv'
print(pred_filename)
with open(pred_filename, 'a') as f:
    f.write('\n')
    f.write(str(datetime.datetime.now()))
    f.write(f'\nBase model spec\n')
    f.write(
        f'\nalpha: {ALPHA}, l1_ratio: {L1_RATIO}, sgd_penalty: {SGD_PENALTY}, sgd_power_t: {SGD_POWER_T}, sgd_eta0: {SGD_ETA0}, en_max_iter: {EN_MAX_ITER}\n')
    f.write(f"\nYield Stress GridSearch Best Estimator:\n")
    f.write(str(ys_bag_final_model))
    f.write(f"\nElongation GridSearch Best Estimator:\n")
    f.write(str(el_bag_final_model))
    y_pred = pd.DataFrame(y_pred)
    f.write(str(y_pred))
    print('\n\n')
#
# Drawing the True vs predicted values figures
for yi in range(2):
    y_target = y_true[yi]

    mn = min(np.min(y_target), np.min(y_pred[:, yi]))
    mx = max(np.max(y_target), np.max(y_pred[:, yi]))
    span = mx - mn
    margin = (span * 0.05)

    fig = plt.figure
    x = np.linspace(mn - margin, mx + margin, len(y_target))

    plt.plot(x, x, c='k')
    plt.scatter(y_target, y_pred[:, yi], c='b', label='test predictions')
    # plt.scatter(ys_test, ys_rf_test_predictions, c='r', label='test predictions')
    plt.grid()
    plt.legend()
    plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Valued")

    plt.savefig('plots/' + base_model_name + '_G1_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

# Drawing the sample IDs vs values figures
for yi in range(2):
    y_target = y_true[yi]

    mn = min(np.min(y_target), np.min(y_pred[:, yi]))
    mx = max(np.max(y_target), np.max(y_pred[:, yi]))
    span = mx - mn
    margin = (span * 0.05)

    fig = plt.figure
    x = np.linspace(mn - margin, mx + margin, len(y_target))
    xx = np.linspace(1, len(y_target), len(y_target))
    plt.plot(xx, y_target, "b-", xx, y_pred[:, yi], "r--")
    plt.grid()
    plt.legend(['Y_true', 'Y_predicted'])
    plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
    plt.xlabel("Instance ID")
    plt.ylabel("Values")

    plt.savefig('plots/' + base_model_name + '_G2_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

# Drawing prediction differences figures
for yi in range(2):
    y_target = y_true[yi]

    mn = min(np.min(y_target), np.min(y_pred[:, yi]))
    mx = max(np.max(y_target), np.max(y_pred[:, yi]))
    span = mx - mn
    margin = (span * 0.05)

    fig = plt.figure
    x = np.linspace(mn - margin, mx + margin, len(y_target))
    xx = np.linspace(1, len(y_target), len(y_target))
    zeros = np.zeros(len(y_target))
    differences = y_target - y_pred[:, yi]
    plt.plot(xx, zeros, "b-", xx, differences, "r--")
    plt.grid()
    plt.legend(['Perfect', 'Differences'])
    plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
    plt.xlabel("Instance ID")
    plt.ylabel("Difference")

    plt.savefig('plots/' + base_model_name + '_G3_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()