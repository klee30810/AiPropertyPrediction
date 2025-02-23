"""

Title: NN Regressor for FollowUp Property Prediction
- Created: 2020.05.06
- Updated: 2020.10.28
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
# from torch.nn.parallel import DataParallelModel, DataParallelCriterion
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
import os

# import NN modules created by km
from km_nn_modules import *

warnings.filterwarnings(action='ignore')
# Check GPU
# GPU 할당 변경하기 => need to correct according to device
GPU_NUM = 0  # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
#print(device)
print('Current cuda device ', torch.cuda.current_device())  # check

# Seed Setting
RND_SEED = 777
np.random.seed(RND_SEED)
random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
Y_LABEL = ['Yield Stress', 'Plastic Strain']


def batch_run(LR, WD, ITERATION, LR_ADJUST_START, MODEL, DO, FEATURE):
    if FEATURE == 'ys': feature = 1;
    elif FEATURE == 'ps': feature = 2;

    # Extract Datasets
    df = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0)
    df_ri = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag', skiprows=0); feasec = 'all';
    #df_ap = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='amp,phs', skiprows=0); feasec = 'all';
    #df_riap = pd.read_excel("induced_voltage_corrected_10-1000Hz.xlsx", sheet_name='real,imag,amp,phs', skiprows=0); feasec = 'all'

    # 1/6
    # ri sub-dataset for real,imag dataset 1/6
    #X = df_ri.iloc[:, 8:]; X = np.array(X); feasec = 1;
    #arr1 = np.linspace(2,7,7-2+1); arr2 = np.linspace(14,39,39-14+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X); feasec = 2;
    #arr1 = np.linspace(2,13,13-2+1); arr2 = np.linspace(20,39,39-20+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X); feasec = 3;
    #arr1 = np.linspace(2,19,19-2+1); arr2 = np.linspace(26,39,39-26+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X); feasec = 4;
    #arr1 = np.linspace(2,25,25-2+1); arr2 = np.linspace(32,39,39-32+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X); feasec = 5;
    #X = df_ri.iloc[:, 2:34]; X = np.array(X); feasec = 6

    # ap sub-dataset for amp,phs dataset 1/6
    #X = df_ap.iloc[:, 8:]; X = np.array(X); feasec = 1;
    #arr1 = np.linspace(2,7,7-2+1); arr2 = np.linspace(14,39,39-14+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X); feasec = 2;
    #arr1 = np.linspace(2,13,13-2+1); arr2 = np.linspace(20,39,39-20+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X); feasec = 3;
    #arr1 = np.linspace(2,19,19-2+1); arr2 = np.linspace(26,39,39-26+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X); feasec = 4;
    #arr1 = np.linspace(2,25,25-2+1); arr2 = np.linspace(32,39,39-32+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X); feasec = 5;
    #X = df_ap.iloc[:, 2:34]; X = np.array(X) feasec = 6;

    # riap sub-dataset for real,imag,amp,phs dataset 1/6
    #X = df_riap.iloc[:, 14:]; X = np.array(X); feasec = 1;
    #arr1 = np.linspace(2,13,13-2+1); arr2 = np.linspace(26,77,77-26+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X); feasec = 2;
    #arr1 = np.linspace(2,25,25-2+1); arr2 = np.linspace(38,77,77-38+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X); feasec = 3;
    #arr1 = np.linspace(2,37,37-2+1); arr2 = np.linspace(50,77,77-50+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X); feasec = 4;
    #arr1 = np.linspace(2,49,49-2+1); arr2 = np.linspace(62,77,77-62+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X); feasec = 5;
    #X = df_riap.iloc[:, 2:66]; X = np.array(X); feasec = 6;

    # 1/3
    # ri sub-dataset for real,imag dataset 1/3
    #X = df_ri.iloc[:, 24:]; X = np.array(X); feasec = 0.3;
    #arr1 = np.linspace(2,23,23-2+1); arr2 = np.linspace(34,39,39-34+1); arr = np.concatenate((arr1,arr2)); X = df_ri.iloc[:, arr]; X = np.array(X); feasec = 0.6;
    #X = df_ri.iloc[:, 2:34]; X = np.array(X); feasec = 0.9

    # ap sub-dataset for amp,phs dataset 1/3
    #X = df_ap.iloc[:, 24:]; X = np.array(X); feasec = 0.3;
    #arr1 = np.linspace(2,23,23-2+1); arr2 = np.linspace(34,39,39-34+1); arr = np.concatenate((arr1,arr2)); X = df_ap.iloc[:, arr]; X = np.array(X); feasec = 0.6;
    #X = df_ap.iloc[:, 2:34]; X = np.array(X) feasec = 0.9;

    # riap sub-dataset for real,imag,amp,phs dataset 1/3
    #X = df_riap.iloc[:, 46:]; X = np.array(X); feasec = 0.3;
    #arr1 = np.linspace(2,45,45-2+1); arr2 = np.linspace(66,77,77-67+1); arr = np.concatenate((arr1,arr2)); X = df_riap.iloc[:, arr]; X = np.array(X); feasec = 0.6;
    #X = df_riap.iloc[:, 2:66]; X = np.array(X); feasec = 0.9;

    # full-datasets
    X = df_ri.iloc[:,2:]; X = np.array(X);
    #X = df_ap.iloc[:,2:]; X = np.array(X);
    #X = df_riap.iloc[:,2:]; X = np.array(X);

    Y = df.iloc[:, [0, 1]];
    ys = Y.iloc[:,0]; ys = np.array(ys);
    el = Y.iloc[:,1]; el = np.array(el);
    Y = np.array(Y)
    y_true = [ys, el]
    y_pred = np.zeros((27,2))

    rs_ys_rmse = []; rs_ys_nrmse = []; rs_ys_r2 = []; rs_ys_ar2 = []
    rs_el_rmse = []; rs_el_nrmse = []; rs_el_r2 = []; rs_el_ar2 = []

    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=RND_SEED)

    #savepth needed
    model_class_name: object = MODEL + '-lr=' + str(LR) + '-wd=' + str(WD) + '-lrST=' + str(LR_ADJUST_START) + '-DO=' + str(DO) + '-fea=' + str(FEATURE) + '-feasec=' + str(feasec);
    print(model_class_name)

    k = 1
    # implication with 10 kfold cross validation
    for tr_idx, ts_idx in kf.split(X):

        net1 = eval(MODEL)(DO).to(device)
        net2 = eval(MODEL)(DO).to(device)
        
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

        if feature == 1:
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


        # Plastic Strain
        print(f'\nStart {k}th  ELONGATION Training!\n')
        inputs = Variable(X_tr).to(device)
        outputs = Variable(el_tr).to(device)

        if feature == 2:
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

    # save the model and state_dict
    MODELPATH = 'models/'

    torch.save(net1, MODELPATH + model_class_name + '_ys.pt')  # 전체 모델 저장
    torch.save(net1.state_dict(), MODELPATH + model_class_name + '_ys_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save(net2, MODELPATH + model_class_name + '_ps.pt')  # 전체 모델 저장
    torch.save(net2.state_dict(), MODELPATH + model_class_name + '_ps_state_dict.pt')  # 모델 객체의 state_dict 저장
    print('\nModel saved!\n')

    # save the result
    result_filename = 'results/' + MODEL + '_feature=' + str(FEATURE) + '_result.csv'
    print(result_filename)
    with open(result_filename, 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write(str(model_class_name))
        f.write(f'\nLearning rate: {LR}, Weight decay: {WD}, LR adjust start: {LR_ADJUST_START}, Dropout: {DO}')
        f.write('\nOverall Results\n')
        if feature == 1:
          f.write(f'ys:rmse   mean={np.average(rs_ys_rmse)} std={np.std(rs_ys_rmse)}\n')
          f.write(f'ys:nrmse  mean={np.average(rs_ys_nrmse)} std={np.std(rs_ys_nrmse)}\n')
          f.write(f'ys:r2     mean={np.average(rs_ys_r2)} std={np.std(rs_ys_r2)}\n')
          f.write(f'ys:adj-r2 mean={np.average(rs_ys_ar2)} std={np.std(rs_ys_ar2)}\n')
        f.write(f'--------------------\n')

        if feature == 2:
          f.write(f'el:rmse   mean={np.average(rs_el_rmse)} std={np.std(rs_el_rmse)}\n')
          f.write(f'el:nrmse  mean={np.average(rs_el_nrmse)} std={np.std(rs_el_nrmse)}\n')
          f.write(f'el:r2     mean={np.average(rs_el_r2)} std={np.std(rs_el_r2)}\n')
          f.write(f'el:adj-r2 mean={np.average(rs_el_ar2)} std={np.std(rs_el_ar2)}\n')
        f.write(f'\n\n')

    print('\nResults saved! \n')

    # Save the predictions
    pred_filename = 'pred/' + MODEL + '_feature=' + str(FEATURE)+ '_y_pred.csv'
    print(pred_filename)
    with open(pred_filename, 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write(str(model_class_name))
        f.write(f'\nLearning rate: {LR}, Weight decay: {WD}, LR adjust start: {LR_ADJUST_START}, Dropout: {DO}')
        f.write(str(y_pred))
        print('\n')
    print('\nPredictions saved! \n')

    # Drawing the True vs predicted values figures
    #    for yi in range(2):
    if feature == 1:
        yi = 0;
    elif feature == 2:
        yi = 1;

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
    plt.title(model_class_name + ' on ' + Y_LABEL[yi], fontsize=20)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Valued")

    plt.savefig('plots/' + model_class_name + '_G1_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    print('\nG1 plot saved! \n')

    # Drawing the sample IDs vs values figures
#    for yi in range(2):
    if feature == 1: yi = 0;
    elif feature == 2: yi = 1;

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
    plt.title(model_class_name + ' on ' + Y_LABEL[yi], fontsize=20)
    plt.xlabel("Instance ID")
    plt.ylabel("Values")

    plt.savefig('plots/' + model_class_name + '_G2_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    print('\nG2 plot saved! \n')

    # Drawing prediction differences figures
#    for yi in range(2):

    if feature == 1:
        yi = 0;
    elif feature == 2:
        yi = 1;
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
    plt.title(model_class_name + ' on ' + Y_LABEL[yi], fontsize=20)
    plt.xlabel("Instance ID")
    plt.ylabel("Difference")

    plt.savefig('plots/' + model_class_name + '_G3_plot_' + Y_LABEL[yi] + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    print('\nG3 plot saved! \n')



# Execution
print("Choose the model name for yield stress: ")
model_ys = input()
print("Choose the model name for plastic strain: ")
model_ps = input()

# Execution

# Yield stress
for wd in [1e-06, 1e-07, 1e-08, 1e-05, 1e-04, 1e-03]:
    for lr in [0.0005]:
        for lr_start in [30000]:
            for do in [0]:
                for feature in ['ys']:
                    batch_run(LR=lr, WD=wd, ITERATION=50000, LR_ADJUST_START=lr_start, MODEL=model_ys, DO=do, FEATURE=feature)


# Plastic Strain
for wd in [1e-06, 1e-07, 1e-08, 1e-05, 1e-04, 1e-03]:
    for lr in [5e-06]:
        for lr_start in [30000]:
            for do in [0]:
                for feature in ['ps']:
                    batch_run(LR=lr, WD=wd, ITERATION=50000, LR_ADJUST_START=lr_start, MODEL=model_ps, DO=do, FEATURE=feature)

# # Dropout setting
# for wd in [ 0.0001,0.005,0.001,0.01]:
#     for lr in [0.0001, 0.0005, 0.001,0.01]:
#         for lr_start in [30000]:
#             for do in [0.1, 0.5, 0.8, 0.9]:
#                 for feature in ['ps']:
#                     batch_run(LR=lr, WD=wd, ITERATION=50000, LR_ADJUST_START=lr_start, MODEL=model_ps, DO=do, FEATURE=feature)

# trial
# for wd in [0.00001]:
#     for lr in [0.007]:
#         for lr_adjust_start in [40000]:
#            for do in [0.9]:
#                for feature in ['ys']:
#                    batch_run(LR=lr, WD=wd, ITERATION=10, LR_ADJUST_START=lr_adjust_start, MODEL=model_ys, DO=0.9, FEATURE=feature)
#
# for wd in [0.00001]:
#     for lr in [0.007]:
#         for lr_adjust_start in [40000]:
#             for do in [0.9]:
#                 for feature in ['ps']:
#                     batch_run(LR=lr, WD=wd, ITERATION=10, LR_ADJUST_START=lr_adjust_start, MODEL=model_ps, DO=0.9, FEATURE=feature)
