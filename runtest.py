# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:48:24 2019

@author: Andy
"""

import pandas as pd
from trafficmgmt import preprocessing, dense, load_model, predict, rmse

if __name__=='__main__':
    print('Preprocessing...')
    X,y = preprocessing('demand.csv')
    
    parameters = {
        'dropout' : 0.5,
        'epoch' : 10,
        'batch_size' : 512}
    
    print('Loading model...')
    model = dense(X, parameters)
    load_model(model)
    
    print('Predicting...')
    pred = predict(model, X)
    pred.index = y.index
    y = y.sort_index()
    pred = pred.sort_index()
    pred['geohash6'],pred['day'],pred['timestamp'] = zip(*pred.index)
    y['geohash6'],y['day'],y['timestamp'] = zip(*y.index)
    
    print('Calculating RMSE...')
    rmset1 = rmse(pred['T1'],y['T1'])
    rmset2 = rmse(pred['T2'],y['T2'])
    rmset3 = rmse(pred['T3'],y['T3'])
    rmset4 = rmse(pred['T4'],y['T4'])
    rmset5 = rmse(pred['T5'],y['T5'])
    predall = pd.Series()
    predall = predall.append([pred['T1'],pred['T2'],pred['T3'],pred['T4'],pred['T5']])
    yall = pd.Series()
    yall = yall.append([y['T1'],y['T2'],y['T3'],y['T4'],y['T5']])
    rmsetall = rmse(predall,yall)
    
    print('RMSE at T1: ', rmset1)
    print('RMSE at T2: ', rmset2)
    print('RMSE at T3: ', rmset3)
    print('RMSE at T4: ', rmset4)
    print('RMSE at T5: ', rmset5)
    print('RMSE at T1 to T5: ', rmsetall)