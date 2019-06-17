# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:48:24 2019

@author: Andy
"""

import pandas as pd
from trafficmgmt import preprocessing, split, dense, fulltraining, save_model, load_model, predict, rmse

if __name__=='__main__':
    print('Preprocessing...')
    X,y = preprocessing('demand.csv')
    X_train, X_test, y_train, y_test = split(X,y)
    
    parameters = {
        'dropout' : 0.5,
        'epoch' : 10,
        'batch_size' : 512}
    
    print('Loading model...')
    model = dense(X_train, parameters)
    model = fulltraining(model, X_train, X_test, y_train, y_test, parameters)
    save_model(model)
    
    print('Predicting...')
    pred = predict(model, X_test)
    pred.index = y_test.index
    y_test = y_test.sort_index()
    pred = pred.sort_index()
    pred['geohash6'],pred['day'],pred['timestamp'] = zip(*pred.index)
    y_test['geohash6'],y_test['day'],y_test['timestamp'] = zip(*y_test.index)
    
    print('Calculating RMSE...')
    rmset1 = rmse(pred['T1'],y_test['T1'])
    rmset2 = rmse(pred['T2'],y_test['T2'])
    rmset3 = rmse(pred['T3'],y_test['T3'])
    rmset4 = rmse(pred['T4'],y_test['T4'])
    rmset5 = rmse(pred['T5'],y_test['T5'])
    predall = pd.Series()
    predall = predall.append([pred['T1'],pred['T2'],pred['T3'],pred['T4'],pred['T5']])
    yall = pd.Series()
    yall = yall.append([y_test['T1'],y_test['T2'],y_test['T3'],y_test['T4'],y_test['T5']])
    rmsetall = rmse(predall,yall)
    
    print('RMSE at T1: ', rmset1)
    print('RMSE at T2: ', rmset2)
    print('RMSE at T3: ', rmset3)
    print('RMSE at T4: ', rmset4)
    print('RMSE at T5: ', rmset5)
    print('RMSE at T1 to T5: ', rmsetall)