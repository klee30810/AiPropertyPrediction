"""

Title: Bagging Regressor for FollowUp Property Prediction
- Created: 2020.06.04
- Updated: 2020.06.11
- Author: Kyung Min, Lee

Learned from
- "Chapter 2 of Hands-on Machine Learning Book"
- PyTorch documents

"""

# import packages
import random
import datetime
import sys
import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet


import warnings
warnings.filterwarnings(action='ignore')

# Seed Setting
RND_SEED = 777
np.random.seed(RND_SEED)
random.seed(RND_SEED)
Y_LABEL = ['Yield Stress', 'Elongation']

def r2_adj_score(y_pred, y_true, X):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    r2_adj = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

    return r2_adj


def bagging_batch_run(BASE_MODEL, ALPHA, L1_RATIO, SGD_PENALTY, SGD_POWER_T, SGD_ETA0, EN_MAX_ITER):
    # Extract Datasets
    # df = pd.read_excel("C:/Users/kmlee/Documents/Handong/Labatory/Nondestructive test/followup/datasets/gather_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
    df = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)

    X = df.iloc[:, 2:]; X = np.array(X)
    Y = df.iloc[:, [0, 1]];
    ys = Y.iloc[:,0]; ys = np.array(ys)
    el = Y.iloc[:,1]; el = np.array(el);
    Y = np.array(Y)

    y_true = [ys, el]
    y_pred = np.zeros((df.shape[0],2))

    rs_ys_rmse = []; rs_ys_nrmse = []; rs_ys_r2 = []; rs_ys_ar2 = []
    rs_el_rmse = []; rs_el_nrmse = []; rs_el_r2 = []; rs_el_ar2 = []

    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=RND_SEED)

    base_model_name = 'Bagging_' + BASE_MODEL + '-alpha=' + str(ALPHA) + '-l1_ratio=' + str(L1_RATIO) + '-sgd_penalty=' + str(SGD_PENALTY)
    print(base_model_name)

    k = 1
    # implication with 10 kfold cross validation
    for tr_idx, ts_idx in kf.split(X):
        sys.stdout.write('.')
        sys.stdout.flush()
        
        # train, test split
        X_tr, X_ts = X[tr_idx], X[ts_idx]
        ys_tr, ys_ts = ys[tr_idx], ys[ts_idx]
        el_tr, el_ts = el[tr_idx], el[ts_idx]

        mu_ys_ts = np.mean(ys_ts)
        mu_el_ts = np.mean(el_ts)

        normalizer = StandardScaler()
        X_tr = normalizer.fit_transform(X_tr)
        X_ts = normalizer.transform(X_ts)

        N = X_tr.shape[0]
        d = X_tr.shape[1]

        # Model Choice
        # Stochastic Gradient Descent
        if BASE_MODEL == 'sgd':
            base_reg = SGDRegressor(random_state=RND_SEED, alpha=ALPHA, penalty=SGD_PENALTY, l1_ratio=L1_RATIO, power_t=SGD_POWER_T, eta0=SGD_ETA0)
            ys_estimators_range = [500, 1000, 3000, 5000, 8000, 10000, 50000]
            ys_lr_range = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.1]
            ys_loss_range = ['linear', 'square', 'exponential']
            el_estimators_range = [500, 1000, 3000, 5000, 8000, 10000, 50000]
            el_lr_range = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.1]
            el_loss_range = ['linear', 'square', 'exponential']

        # Elastic Net
        elif BASE_MODEL == 'elastic':
            base_reg = ElasticNet(random_state=RND_SEED, alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=EN_MAX_ITER)
            SGD_PENALTY = 'NONE'

        # GridSearchCV - yield stress
        ys_param_grid = [
            {
                'base_estimator': [base_reg],
                'n_estimators': [100, 1000, 10000],
                'max_samples': [0.8, 0.9, 1],

            }
        ]

        # GridSearchCV - elongation
        el_param_grid = [
            {
                'base_estimator': [base_reg],
                'n_estimators': [100, 1000, 10000],
                'max_samples': [0.8, 0.9, 1],

            }
        ]

        # Train
        # Yield Stress
        print(f'\nStart {k}th CV YIELD STRESS Grid Search Cross Validation!\n')

        bag_reg = BaggingRegressor(random_state=777)
        ys_grid_search = GridSearchCV(bag_reg, ys_param_grid, cv=2,
                                      scoring='neg_mean_squared_error',
                                      return_train_score=True)
        ys_grid_search.fit(X_tr, ys_tr)

        print("\nYield Stress GridSearch Best Estimator:\n")
        print(ys_grid_search.best_estimator_)
        print('\n')

        ys_bag_final_model = ys_grid_search.best_estimator_

        # train predictions
        ys_bag_train_predictions = ys_bag_final_model.predict(X_tr)  # * np.std(ys_train) + np.mean(ys_train)
        ys_bag_train_mse = mean_squared_error(ys_tr, ys_bag_train_predictions)
        ys_bag_train_rmse = np.sqrt(ys_bag_train_mse)
        ys_bag_train_nrmse = ys_bag_train_rmse / np.mean(ys_tr)
        ys_bag_r2 = r2_score(y_true=ys_tr, y_pred=ys_bag_train_predictions)
        ys_bag_r2_adj = r2_adj_score(y_pred=ys_bag_train_predictions, y_true=ys_tr, X=X_tr)

        print("yield stress bag train prediction rmse: ", ys_bag_train_rmse)
        print("yield stress bag train prediction nrmse: ", ys_bag_train_nrmse)
        print("yield stress bag_r2__score: ", round(ys_bag_r2, 4))
        print("yield stress bag_r2_adj_score: ", round(ys_bag_r2_adj, 4))  # independent variable수가 많아서 분모가 -가 됨

        # test predictions
        ys_bag_test_predictions = ys_bag_final_model.predict(X_ts)  # * np.std(ys_test) + np.mean(ys_test)
        ys_bag_test_mse = mean_squared_error(ys_ts, ys_bag_test_predictions)
        ys_bag_test_rmse = np.sqrt(ys_bag_test_mse)
        ys_bag_test_nrmse = ys_bag_test_rmse / np.mean(ys_ts)
        ys_bag_r2 = r2_score(y_true=ys_ts, y_pred=ys_bag_test_predictions)
        ys_bag_r2_adj = r2_adj_score(y_pred=ys_bag_test_predictions, y_true=ys_ts, X=X_ts)

        print("yield stress bag test prediction rmse: ", ys_bag_test_rmse)
        print("yield stress bag test prediction nrmse: ", ys_bag_test_nrmse)
        print("yield stress bag_r2__score: ", round(ys_bag_r2, 4))
        print("yield stress bag_r2_adj_score: ", round(ys_bag_r2_adj, 4))  # independent variable수가 많아서 분모가 -가 됨

        rs_ys_rmse.append(ys_bag_test_rmse)
        rs_ys_nrmse.append(ys_bag_test_nrmse)
        rs_ys_r2.append(ys_bag_r2)
        rs_ys_ar2.append(ys_bag_r2_adj)

        # Elongation
        print(f'\nStart {k}th CV ELONGATION Grid Search Cross Validation!\n')

        el_grid_search = GridSearchCV(bag_reg, el_param_grid, cv=2,
                                      scoring='neg_mean_squared_error',
                                      return_train_score=True)
        el_grid_search.fit(X_tr, el_tr)

        print("\nElongation GridSearch Best Estimator:\n")
        print(el_grid_search.best_estimator_)
        print('\n')

        el_bag_final_model = el_grid_search.best_estimator_

        # train predictions
        el_bag_train_predictions = el_bag_final_model.predict(X_tr)  # * np.std(el_train) + np.mean(el_train)
        el_bag_train_mse = mean_squared_error(el_tr, el_bag_train_predictions)
        el_bag_train_rmse = np.sqrt(el_bag_train_mse)
        el_bag_train_nrmse = el_bag_train_rmse / np.mean(el_tr)
        el_bag_r2 = r2_score(y_true=el_tr, y_pred=el_bag_train_predictions)
        el_bag_r2_adj = r2_adj_score(y_pred=el_bag_train_predictions, y_true=el_tr, X=X_tr)

        print("elongation bag train prediction rmse: ", el_bag_train_rmse)
        print("elongation bag train prediction nrmse: ", el_bag_train_nrmse)
        print("elongation bag_r2__score: ", round(el_bag_r2, 4))
        print("elongation bag_r2_adj_score: ", round(el_bag_r2_adj, 4))  # independent variable수가 많아서 분모가 -가 됨

        # test predictions
        el_bag_test_predictions = el_bag_final_model.predict(X_ts)  # * np.std(el_test) + np.mean(el_test)
        el_bag_test_mse = mean_squared_error(el_ts, el_bag_test_predictions)
        el_bag_test_rmse = np.sqrt(el_bag_test_mse)
        el_bag_test_nrmse = el_bag_test_rmse / np.mean(el_ts)
        el_bag_r2 = r2_score(y_true=el_ts, y_pred=el_bag_test_predictions)
        el_bag_r2_adj = r2_adj_score(y_pred=el_bag_test_predictions, y_true=el_ts, X=X_ts)

        print("elongation bag test prediction rmse: ", el_bag_test_rmse)
        print("elongation bag test prediction nrmse: ", el_bag_test_nrmse)
        print("elongation bag_r2__score: ", round(el_bag_r2, 4))
        print("elongation bag_r2_adj_score: ", round(el_bag_r2_adj, 4))  # independent variable수가 많아서 분모가 -가 됨

        rs_el_rmse.append(el_bag_test_rmse)
        rs_el_nrmse.append(el_bag_test_nrmse)
        rs_el_r2.append(el_bag_r2)
        rs_el_ar2.append(el_bag_r2_adj)

        y_pred[ts_idx, 0] = ys_bag_test_predictions.reshape(-1)
        y_pred[ts_idx, 1] = el_bag_test_predictions.reshape(-1)

        k += 1

    print('\ndone\n')

    # Print results
    print('Overall Results\n')
    print("\nYield Stress GridSearch Best Estimator:\n")
    print(ys_bag_final_model)
    print('\n\n')
    print('ys:rmse   ', ' mean=%.4f std=%.4f' % (np.average(rs_ys_rmse), np.std(rs_ys_rmse)))
    print('ys:nrmse   ', ' mean=%.4f std=%.4f' % (np.average(rs_ys_nrmse), np.std(rs_ys_nrmse)))
    print('ys:r2    ', ' mean=%.4f std=%.4f' % (np.average(rs_ys_r2), np.std(rs_ys_r2)))
    print('ys:adj-r2', ' mean=%.4f std=%.4f' % (np.average(rs_ys_ar2), np.std(rs_ys_ar2)))
    print('\n----------------\n')
    print("\nElongation GridSearch Best Estimator:\n")
    print(el_bag_final_model)
    print('\n\n')
    print('el:rmse  ', ' mean=%.4f std=%.4f' % (np.average(rs_el_rmse), np.std(rs_el_rmse)))
    print('el:nrmse  ', ' mean=%.4f std=%.4f' % (np.average(rs_el_nrmse), np.std(rs_el_nrmse)))
    print('el:r2    ', ' mean=%.4f std=%.4f' % (np.average(rs_el_r2), np.std(rs_el_r2)))
    print('el:adj-r2', ' mean=%.4f std=%.4f' % (np.average(rs_el_ar2), np.std(rs_el_ar2)))
    print('\n\n')

    # save the result
    result_filename = 'results/bagging_' + BASE_MODEL + '_result.csv'
    print(result_filename)
    with open(result_filename, 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write(f'\nBase model spec')
        f.write(f'\nalpha: {ALPHA}, l1_ratio: {L1_RATIO}, sgd_penalty: {SGD_PENALTY}, sgd_power_t: {SGD_POWER_T}, sgd_eta0: {SGD_ETA0}, en_max_iter: {EN_MAX_ITER}\n')
        f.write('\nOverall Results\n')
        f.write(f"\nYield Stress GridSearch Best Estimator:\n")
        f.write(str(ys_bag_final_model))
        f.write('\n\n')
        f.write(f'ys:rmse   mean={np.average(rs_ys_rmse)} std={np.std(rs_ys_rmse)}\n')
        f.write(f'ys:nrmse  mean={np.average(rs_ys_nrmse)} std={np.std(rs_ys_nrmse)}\n')
        f.write(f'ys:r2     mean={np.average(rs_ys_r2)} std={np.std(rs_ys_r2)}\n')
        f.write(f'ys:adj-r2 mean={np.average(rs_ys_ar2)} std={np.std(rs_ys_ar2)}\n')
        f.write(f'\n--------------------\n')
        f.write(f"\nElongation GridSearch Best Estimator:\n")
        f.write(str(el_bag_final_model))
        f.write('\n\n')
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
        f.write(f'\nalpha: {ALPHA}, l1_ratio: {L1_RATIO}, sgd_penalty: {SGD_PENALTY}, sgd_power_t: {SGD_POWER_T}, sgd_eta0: {SGD_ETA0}, en_max_iter: {EN_MAX_ITER}\n')
        f.write(f"\nYield Stress GridSearch Best Estimator:\n")
        f.write(str(ys_bag_final_model))
        f.write(f"\nElongation GridSearch Best Estimator:\n")
        f.write(str(el_bag_final_model))
        y_pred = pd.DataFrame(y_pred)
        f.write(str(y_pred))
        print('\n\n')

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
        plt.stem(xx, differences, linefmt='r--.', markerfmt='bx', basefmt='k-')
        plt.grid()
        plt.legend(['Differences'])
        plt.title(BASE_MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
        plt.xlabel("Instance ID")
        plt.ylabel("Difference")

        plt.savefig('plots/' + base_model_name + '_G3_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
        # plt.show()
        plt.close()


# Execution
print("Choose the base model(sgd, elastic): ")
BASE = input()

if BASE == 'sgd':
    for alpha in [0.001,  0.003, 0.005]:
        for l1_ratio in [0.2, 0.3, 0.4]:
            for sgd_penalty in ['l2', 'l1', 'elasticnet']:
                for power_t in [0.01, 0.02, 0.25, 0.03]:
                    for eta0 in [0.001, 0.01, 0.1]:
                        bagging_batch_run(BASE_MODEL = BASE, ALPHA=alpha, L1_RATIO=l1_ratio, SGD_PENALTY=sgd_penalty, SGD_ETA0=eta0, SGD_POWER_T=power_t, EN_MAX_ITER=0)

elif BASE == 'elastic':
    for alpha in [0.001, 0.005, 0.01]:
        for l1_ratio in [0.2, 0.3, 0.4]:
            for max_iter in [1000,10000]:
                bagging_batch_run(BASE_MODEL = BASE, ALPHA=alpha, L1_RATIO=l1_ratio, SGD_PENALTY=0, SGD_ETA0=0, SGD_POWER_T=0, EN_MAX_ITER=max_iter)

# trial
# if BASE == 'sgd':
#     for alpha in [0.001]:
#         for l1_ratio in [0.3]:
#             for sgd_penalty in ['l2', 'l1', 'elasticnet']:
#                 bagging_batch_run(BASE_MODEL = BASE, ALPHA=alpha, L1_RATIO=l1_ratio, SGD_PENALTY=sgd_penalty)
#
# elif BASE == 'elastic':
#     for alpha in [0.001]:
#         for l1_ratio in [0.7]:
#            bagging_batch_run(BASE_MODEL = BASE, ALPHA=alpha, L1_RATIO=l1_ratio, SGD_PENALTY=0)
#
