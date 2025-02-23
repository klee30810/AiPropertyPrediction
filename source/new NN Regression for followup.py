"""

Title: NN Regressor for FollowUp Property Prediction
- Created: 2020.05.06
- Updated: 2020.06.03
- Author: Kyung Min, Lee

Learned from
- "Chapter 2 of Hands-on Machine Learning Book"
- PyTorch documents

"""

# import packages
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import sys
import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time

# import NN modules created by km
from km_nn_modules import *

warnings.filterwarnings(action='ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Seed Setting
RND_SEED = 777
np.random.seed(RND_SEED)
random.seed(RND_SEED)
torch.manual_seed(RND_SEED)



def batch_run(LR, WD, ITERATION, LR_ADJUST_START, MODEL):
    # Extract Datasets
    # df = pd.read_excel("C:/Users/kmlee/Documents/Handong/Labatory/Nondestructive test/followup/datasets/gather_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
    df = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)

    X = df.iloc[:, 2:]; X = np.array(X)
    Y = df.iloc[:, [0, 1]];
    ys = Y.iloc[:,0]; ys = np.array(ys)
    el = Y.iloc[:,1]; el = np.array(el);
    Y = np.array(Y)

    y_true = [ys, el]
    y_pred = np.zeros((27,2))

    rs_ys_rmse = []; rs_ys_nrmse = []; rs_ys_r2 = []; rs_ys_ar2 = []
    rs_el_rmse = []; rs_el_nrmse = []; rs_el_r2 = []; rs_el_ar2 = []

    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=RND_SEED)


    model_class_name = MODEL + '-lr=' + str(LR) + '-weight_decay=' + str(WD)
    print(model_class_name)

    k = 1
    # implication with 10 kfold cross validation
    for tr_idx, ts_idx in kf.split(X):

        net1 = eval(MODEL)().to(device)
        net2 = eval(MODEL)().to(device)
        
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

        # convert numpy array to tensor in shape of input size
        X_tr = torch.from_numpy(X_tr).float()
        ys_tr = torch.from_numpy(ys_tr.reshape(-1,1)).float()
        el_tr = torch.from_numpy(el_tr.reshape(-1,1)).float()

        X_ts = torch.from_numpy(X_ts).float()
        ys_ts = torch.from_numpy(ys_ts.reshape(-1,1)).float()
        el_ts = torch.from_numpy(el_ts.reshape(-1,1)).float()

        # Define Optimizer and Loss Function
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=LR, weight_decay=WD)
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=LR, weight_decay=WD)
        loss_func = torch.nn.MSELoss()
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', patience=500, factor=0.95, verbose=False)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', patience=500, factor=0.95, verbose=False)

        # Train
        # Yield Stress
        print(f'\nStart {k}th YIELD STRESS Training!\n')

        inputs = Variable(X_tr).to(device)
        outputs = Variable(ys_tr).to(device)

        for i in range(ITERATION):
            prediction = net1(inputs)
            loss1 = loss_func(prediction, outputs)
            optimizer1.zero_grad()
            loss1.backward()
            if i > LR_ADJUST_START:
                scheduler1.step(loss1)
            else:
                optimizer1.step()

            if i % 5000 == 0:
                print('Process: {}/{}, Loss : {}'.format(i, ITERATION, loss1))


        # Elongation
        print(f'\nStart {k}th  ELONGATION Training!\n')
        inputs = Variable(X_tr).to(device)
        outputs = Variable(el_tr).to(device)

        for i in range(ITERATION):
            prediction = net2(inputs)
            loss2 = loss_func(prediction, outputs)
            optimizer2.zero_grad()
            loss2.backward()
            if i > LR_ADJUST_START:
                scheduler2.step(loss2)
            else:
                optimizer2.step()

            if i % 5000 == 0:
                print('Process: {}/{}, Loss : {}'.format(i, ITERATION, loss2))

        #    if i % 10 == 0:
        #        # plot and show learning process
        #        plt.cla()
        #        plt.scatter(x.data.numpy(), y.data.numpy())
        #        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
        #        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
        #        plt.pause(0.1)
        #
        # plt.show()

        # Test
        print(f'\nStart {k}th Test\n')
        with torch.no_grad():
            ys_ts_pred = net1(Variable(X_ts).to(device))
            el_ts_pred = net2(Variable(X_ts).to(device))

        y_pred[ts_idx, 0] = ys_ts_pred.cpu().numpy().reshape(-1)
        y_pred[ts_idx, 1] = el_ts_pred.cpu().numpy().reshape(-1)

        rmse1 = np.sqrt(mean_squared_error(ys_ts, ys_ts_pred.cpu()))
        rsq1 = r2_score(ys_ts, ys_ts_pred.cpu())
        adjrsq1 = 1 - (1-rsq1)*(N-1)/(N-d-1)

        rmse2 = np.sqrt(mean_squared_error(el_ts, el_ts_pred.cpu()))
        rsq2 = r2_score(el_ts, el_ts_pred.cpu())
        adjrsq2 = 1 - (1-rsq2)*(N-1)/(N-d-1)

        rs_ys_rmse.append(rmse1)
        rs_ys_nrmse.append(rmse1/mu_ys_ts)
        rs_ys_r2.append(rsq1)
        rs_ys_ar2.append(adjrsq1)

        rs_el_rmse.append(rmse2)
        rs_el_nrmse.append(rmse2/mu_el_ts)
        rs_el_r2.append(rsq2)
        rs_el_ar2.append(adjrsq2)

        k += 1

    print('\ndone\n')

    # Print results
    print('Overall Results\n')
    print('ys:rmse   ', ' mean=%.4f std=%.4f' % (np.average(rs_ys_rmse), np.std(rs_ys_rmse)))
    print('ys:nrmse   ', ' mean=%.4f std=%.4f' % (np.average(rs_ys_nrmse), np.std(rs_ys_nrmse)))
    print('ys:r2    ', ' mean=%.4f std=%.4f' % (np.average(rs_ys_r2), np.std(rs_ys_r2)))
    print('ys:adj-r2', ' mean=%.4f std=%.4f' % (np.average(rs_ys_ar2), np.std(rs_ys_ar2)))
    print('----------------\n')
    print('el:rmse  ', ' mean=%.4f std=%.4f' % (np.average(rs_el_rmse), np.std(rs_el_rmse)))
    print('el:nrmse  ', ' mean=%.4f std=%.4f' % (np.average(rs_el_nrmse), np.std(rs_el_nrmse)))
    print('el:r2    ', ' mean=%.4f std=%.4f' % (np.average(rs_el_r2), np.std(rs_el_r2)))
    print('el:adj-r2', ' mean=%.4f std=%.4f' % (np.average(rs_el_ar2), np.std(rs_el_ar2)))
    print('\n')

    # save the result
    result_filename = 'results/' + MODEL + '_result.csv'
    print(result_filename)
    with open(result_filename, 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write(f'\nLearning rate: {LR}, Weight decay: {WD}, LR adjust start: {LR_ADJUST_START}')
        f.write('\nOverall Results\n')
        f.write(f'ys:rmse   mean={np.average(rs_ys_rmse)} std={np.std(rs_ys_rmse)}\n')
        f.write(f'ys:nrmse  mean={np.average(rs_ys_nrmse)} std={np.std(rs_ys_nrmse)}\n')
        f.write(f'ys:r2     mean={np.average(rs_ys_r2)} std={np.std(rs_ys_r2)}\n')
        f.write(f'ys:adj-r2 mean={np.average(rs_ys_ar2)} std={np.std(rs_ys_ar2)}\n')
        f.write(f'--------------------\n')
        f.write(f'el:rmse   mean={np.average(rs_el_rmse)} std={np.std(rs_el_rmse)}\n')
        f.write(f'el:nrmse  mean={np.average(rs_el_nrmse)} std={np.std(rs_el_nrmse)}\n')
        f.write(f'el:r2     mean={np.average(rs_el_r2)} std={np.std(rs_el_r2)}\n')
        f.write(f'el:adj-r2 mean={np.average(rs_el_ar2)} std={np.std(rs_el_ar2)}\n')
        f.write(f'\n\n')

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
    #     plt.title(MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
    #     plt.xlabel("True Values")
    #     plt.ylabel("Predicted Valued")
    #
    #     plt.savefig('output/' + MODEL + '_G1_plot_y' + str(yi), bbox_inches='tight')
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
    #     plt.title(MODEL + ' on ' + Y_LABEL[yi], fontsize=20)
    #     plt.xlabel("Instance ID")
    #     plt.ylabel("Values")
    #
    #     plt.savefig('output/' + MODEL + '_G2_plot_y' + str(yi), bbox_inches='tight')
    #     # plt.show()
    #     plt.close()
    #
    # Save the predictions
    pred_filename = 'pred/' + MODEL + '_y_pred.csv'
    print(pred_filename)
    with open(pred_filename, 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write(f'\nLearning rate: {LR}, Weight decay: {WD}, LR adjust start: {LR_ADJUST_START}')
        f.write(str(y_pred))
        print('\n')

# Execution
print("Choose the model name: ")
model = input()

for wd in [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]:
    for lr in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]:
        for lr_adjust_start in [20000,30000,40000]:
            batch_run(LR=lr, WD=wd, ITERATION=50000, LR_ADJUST_START=lr_adjust_start, MODEL=model)
