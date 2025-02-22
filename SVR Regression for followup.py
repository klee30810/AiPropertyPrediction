"""

Title: SVR Regressor for FollowUp Property Prediction
- Created: 2020.05.06
- Updated: 2020.05.29
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
warnings.filterwarnings(action='ignore')
np.random.seed(1)


# Extract Datasets
#df = pd.read_excel("C:/Users/kmlee/Documents/Handong/Labatory/Nondestructive test/followup/datasets/gather_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
df = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)

# train, test split(8:2)
from sklearn.model_selection import train_test_split
X = df.iloc[:, 2:]
ys = df.iloc[:,0]
elong = df.iloc[:,1]

X_train, X_test, ys_train, ys_test = train_test_split(X, ys, test_size=0.2, random_state=1)
X_train, X_test, el_train, el_test = train_test_split(X, elong, test_size=0.2, random_state=1)


# correlation heatmap -> community edition이라서 안보이므로 업그레이드 필요
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

# Yield Stress
print("\nYIELD STRESS\n")

# SVR regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_std, ys_train)
ys_predictions_svr = svr_reg.predict(X_train_std) #* np.std(ys_train) + np.mean(ys_train)
svr_mse = mean_squared_error(ys_train, ys_predictions_svr)
svr_rmse = np.sqrt(svr_mse)
svr_nrmse = svr_rmse / np.mean(ys_train)
print("test svr_rmse: ", svr_rmse)
print("test rlr_nrmse: ", svr_nrmse)

# Hyper Parameters Tuning: GridSearch CV
from sklearn.model_selection import GridSearchCV

C_range = [390,400,525,550,575,600,700,800,900,1000,2000,4000]
gamma_range = [0.008,0.0085,0.009,0.0095,0.01,0.013,0.015,0.017,0.02]
epsilon_range =[0.8,0.85,0.9,0.91,0.92,0.93,0.95,0.97,0.98]

param_grid = [
    {'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon': epsilon_range}
]

svr_reg = SVR()

grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
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
ys_svr_final_model = grid_search.best_estimator_

# train prediction
ys_svr_train_predictions = ys_svr_final_model.predict(X_train_std) #* np.std(ys_train) + np.mean(ys_train)
ys_svr_train_mse = mean_squared_error(ys_train, ys_svr_train_predictions)
ys_svr_train_rmse = np.sqrt(ys_svr_train_mse)
ys_svr_train_nrmse = ys_svr_train_rmse/np.mean(ys_train)
print("yield stress svr train prediction rmse: ", ys_svr_train_rmse)
print("yield stress svr train prediction nrmse: ", ys_svr_train_nrmse)


# test prediction
ys_svr_test_predictions = ys_svr_final_model.predict(X_test_std) #* np.std(ys_train) + np.mean(ys_train)
ys_svr_test_mse = mean_squared_error(ys_test,ys_svr_test_predictions)
ys_svr_test_rmse = np.sqrt(ys_svr_test_mse)
ys_svr_test_nrmse = ys_svr_test_rmse/np.mean(ys_test)
print("yield stress svr test prediction rmse: ", ys_svr_test_rmse)
print("yield stress svr test prediction nrmse: ", ys_svr_test_nrmse)

# Calculate adjusted R^2 scores
from sklearn.metrics import r2_score

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj

svr_r2 = r2_score(y_true=ys_test, y_pred=ys_svr_test_predictions)
svr_r2_adj = r2_adj_score(y_pred=ys_svr_test_predictions, y_true=ys_test, X=X_train)

print("yield stress svr_r2__score: ", round(svr_r2,4))
print("yield stress svr_r2_adj_score: ", round(svr_r2_adj,4))   # independent variable수가 많아서 분모가 -가 됨


# Elongation
print("\nELONGATION\n")

# SVR regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_std, el_train)
el_predictions_svr = svr_reg.predict(X_train_std) #* np.std(el_train) + np.mean(el_train)
svr_mse = mean_squared_error(el_train, el_predictions_svr)
svr_rmse = np.sqrt(svr_mse)
svr_nrmse = svr_rmse/np.mean(ys_train)
print("test svr_rmse: ", svr_rmse)
print("test svr_nrmse: ", svr_nrmse)

# Hyper Parameters Tuning: GridSearch CV
from sklearn.model_selection import GridSearchCV

C_range = [70,73,75,77,80,83,85,88,90,95,100]
gamma_range = [0.001,0.0013,0.0015,0.0017,0.00019,0.002]
epsilon_range = [0.00001,0.00002,0.00003,0.00004,0.00005,0.000055,0.00006,0.000065,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.003]

param_grid = [
    {'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon': epsilon_range}
]

svr_reg = SVR()

grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
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
el_svr_final_model = grid_search.best_estimator_

# train prediction
el_svr_train_predictions = el_svr_final_model.predict(X_train_std) #* np.std(el_train) + np.mean(el_train)
el_svr_train_mse = mean_squared_error(el_train, el_svr_train_predictions)
el_svr_train_rmse = np.sqrt(el_svr_train_mse)
el_svr_train_nrmse = el_svr_train_rmse / np.mean(el_train)
print("elongation svr train prediction rmse: ", el_svr_train_rmse)
print("elongation svr train prediction nrmse: ", el_svr_train_nrmse)

# test prediction
el_svr_test_predictions = el_svr_final_model.predict(X_test_std) #* np.std(el_train) + np.mean(el_train)
el_svr_test_mse = mean_squared_error(el_test,el_svr_test_predictions)
el_svr_test_rmse = np.sqrt(el_svr_test_mse)
el_svr_test_nrmse = el_svr_test_rmse/np.mean(el_test)
print("elonagtion svr test prediction rmse: ", el_svr_test_rmse)
print("elongation svr test prediction nrmse: ", el_svr_test_nrmse)

# Calculate adjusted R^2 scores
from sklearn.metrics import r2_score

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj

svr_r2 = r2_score(y_true=el_test, y_pred=el_svr_test_predictions)
svr_r2_adj = r2_adj_score(y_pred=el_svr_test_predictions, y_true=el_test, X=X_train)

print("elongation svr_r2__score: ", round(svr_r2,4))
print("elongation svr_r2_adj_score: ", round(svr_r2_adj,4))   # independent variable수가 많아서 분모가 -가 됨

# save the result
# result_filename = 'results/bagging_' + BASE_MODEL + '_result.csv'
# print(result_filename)
# with open(result_filename, 'a') as f:
#     f.write(str(datetime.datetime.now()))
#     f.write(f'\nBase model spec')
#     f.write(
#         f'\nalpha: {ALPHA}, l1_ratio: {L1_RATIO}, sgd_penalty: {SGD_PENALTY}, sgd_power_t: {SGD_POWER_T}, sgd_eta0: {SGD_ETA0}, en_max_iter: {EN_MAX_ITER}\n')
#     f.write('\nOverall Results\n')
#     f.write(f"\nYield Stress GridSearch Best Estimator:\n")
#     f.write(str(ys_bag_final_model))
#     f.write('\n\n')
#     f.write(f'ys:rmse   mean={np.average(rs_ys_rmse)} std={np.std(rs_ys_rmse)}\n')
#     f.write(f'ys:nrmse  mean={np.average(rs_ys_nrmse)} std={np.std(rs_ys_nrmse)}\n')
#     f.write(f'ys:r2     mean={np.average(rs_ys_r2)} std={np.std(rs_ys_r2)}\n')
#     f.write(f'ys:adj-r2 mean={np.average(rs_ys_ar2)} std={np.std(rs_ys_ar2)}\n')
#     f.write(f'\n--------------------\n')
#     f.write(f"\nElongation GridSearch Best Estimator:\n")
#     f.write(str(el_bag_final_model))
#     f.write('\n\n')
#     f.write(f'el:rmse   mean={np.average(rs_el_rmse)} std={np.std(rs_el_rmse)}\n')
#     f.write(f'el:nrmse  mean={np.average(rs_el_nrmse)} std={np.std(rs_el_nrmse)}\n')
#     f.write(f'el:r2     mean={np.average(rs_el_r2)} std={np.std(rs_el_r2)}\n')
#     f.write(f'el:adj-r2 mean={np.average(rs_el_ar2)} std={np.std(rs_el_ar2)}\n')
#     f.write(f'\n\n')
#
# # Save the predictions
# pred_filename = 'pred/bagging_' + BASE_MODEL + '_preds.csv'
# print(pred_filename)
# with open(pred_filename, 'a') as f:
#     f.write('\n')
#     f.write(str(datetime.datetime.now()))
#     f.write(f'\nBase model spec\n')
#     f.write(
#         f'\nalpha: {ALPHA}, l1_ratio: {L1_RATIO}, sgd_penalty: {SGD_PENALTY}, sgd_power_t: {SGD_POWER_T}, sgd_eta0: {SGD_ETA0}, en_max_iter: {EN_MAX_ITER}\n')
#     f.write(f"\nYield Stress GridSearch Best Estimator:\n")
#     f.write(str(ys_bag_final_model))
#     f.write(f"\nElongation GridSearch Best Estimator:\n")
#     f.write(str(el_bag_final_model))
#     y_pred = pd.DataFrame(y_pred)
#     f.write(str(y_pred))
#     print('\n\n')
#
# # Drawing the True vs predicted values figures
# for yi in range(2):
#     y_target = y_true[yi]
#
#     mn = min(np.min(y_target), np.min(y_pred[:, yi]))
#     mx = max(np.max(y_target), np.max(y_pred[:, yi]))
#     span = mx - mn
#     margin = (span * 0.05)
#
#     fig = plt.figure
#     x = np.linspace(mn - margin, mx + margin, len(y_target))
#
#     plt.plot(x, x, c='k')
#     plt.scatter(y_target, y_pred[:, yi], c='b', label='test predictions')
#     # plt.scatter(ys_test, ys_rf_test_predictions, c='r', label='test predictions')
#     plt.grid()
#     plt.legend()
#     plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
#     plt.xlabel("True Values")
#     plt.ylabel("Predicted Valued")
#
#     plt.savefig('plots/' + base_model_name + '_G1_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
# # Drawing the sample IDs vs values figures
# for yi in range(2):
#     y_target = y_true[yi]
#
#     mn = min(np.min(y_target), np.min(y_pred[:, yi]))
#     mx = max(np.max(y_target), np.max(y_pred[:, yi]))
#     span = mx - mn
#     margin = (span * 0.05)
#
#     fig = plt.figure
#     x = np.linspace(mn - margin, mx + margin, len(y_target))
#     xx = np.linspace(1, len(y_target), len(y_target))
#     plt.plot(xx, y_target, "b-", xx, y_pred[:, yi], "r--")
#     plt.grid()
#     plt.legend(['Y_true', 'Y_predicted'])
#     plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
#     plt.xlabel("Instance ID")
#     plt.ylabel("Values")
#
#     plt.savefig('plots/' + base_model_name + '_G2_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
# # Drawing prediction differences figures
# for yi in range(2):
#     y_target = y_true[yi]
#
#     mn = min(np.min(y_target), np.min(y_pred[:, yi]))
#     mx = max(np.max(y_target), np.max(y_pred[:, yi]))
#     span = mx - mn
#     margin = (span * 0.05)
#
#     fig = plt.figure
#     x = np.linspace(mn - margin, mx + margin, len(y_target))
#     xx = np.linspace(1, len(y_target), len(y_target))
#     zeros = np.zeros(len(y_target))
#     differences = y_target - y_pred[:, yi]
#     plt.plot(xx, zeros, "b-", xx, differences, "r--")
#     plt.grid()
#     plt.legend(['Perfect', 'Differences'])
#     plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
#     plt.xlabel("Instance ID")
#     plt.ylabel("Difference")
#
#     plt.savefig('plots/' + base_model_name + '_G3_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
#     # plt.show()
#     plt.close()