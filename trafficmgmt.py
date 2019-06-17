# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:58:50 2019

@author: Andy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

import math
from math import log10
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__decodemap = { }
for i in range(len(__base32)):
    __decodemap[__base32[i]] = i
del i

def decode_exactly(geohash):
    """
    Decode the geohash to its exact values, including the error
    margins of the result.  Returns four float values: latitude,
    longitude, the plus/minus error for latitude (as a positive
    number) and the plus/minus error for longitude (as a positive
    number).
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even: # adds longitude info
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:      # adds latitude info
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon, lat_err, lon_err

def decode(geohash):
    """
    Decode geohash, returning two strings with latitude and longitude
    containing only relevant digits and with trailing zeroes removed.
    """
    lat, lon, lat_err, lon_err = decode_exactly(geohash)
    # Format to the number of decimals that are known
    lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
    lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)
    if '.' in lats: lats = lats.rstrip('0')
    if '.' in lons: lons = lons.rstrip('0')
    return lats, lons

def distance(originlats, originlons, destlats, destlons):
    
    radius = 6371 # km
    
    dlat = (destlats - originlats).map(math.radians)
    dlon = (destlons - originlons).map(math.radians)
    
    a = (dlat/2).map(math.sin) * (dlat/2).map(math.sin) + originlats.map(math.radians).map(math.cos) * destlats.map(math.radians).map(math.cos) * (dlon/2).map(math.sin) * (dlon/2).map(math.sin)
    c = pd.Series(2 * np.vectorize(math.atan2)(a.map(math.sqrt), (1-a).map(math.sqrt)))
    d = radius * c

    return d

def preprocessing(filename):

    raw = pd.read_csv(filename)
    timekeys = pd.Series(raw.timestamp.unique())
    timevalues = pd.to_datetime(timekeys, format="%H:%M").dt.time
    timedict = dict(zip(timekeys, timevalues))
    raw['timestamp'] = raw['timestamp'].map(timedict)
    
    allindex =  product(np.sort(raw.geohash6.unique()),np.sort(raw.day.unique()),np.sort(raw.timestamp.unique()))
    raw.index = list(zip(raw.geohash6,raw.day,raw.timestamp))
    df = pd.DataFrame(index=allindex).sort_index()
    df['geohash6'],df['day'],df['timestamp'] = zip(*df.index)
    df['day'] = df['day'].astype(np.int8)
    df['demand'] = raw['demand'].astype(np.float32)
    df['demand'].fillna(0,inplace=True)
    del raw
    
    daykeys = df.day.unique()
    dayvalues = daykeys % 7
    daydict = dict(zip(daykeys, dayvalues))
    df['day_'] = df['day'].map(daydict)
    
    meanday = dict(df.groupby('day_')['demand'].mean())
    meantime = dict(df.groupby('timestamp')['demand'].mean())
    meangeo = dict(df.groupby('geohash6')['demand'].mean())
    
    df['meanday'] = df.day_.map(meanday)
    df['meantime'] = df.timestamp.map(meantime)
    df['meangeo'] = df.geohash6.map(meangeo)
    
    geokeys = np.sort(df.geohash6.unique())
    decoder = lambda x: decode(x)
    geovalues = list(map(decoder,geokeys))
    
    geo_ = pd.DataFrame(list(product(geovalues,geovalues)),columns=['origin','destination'])
    temp = pd.DataFrame(geo_.origin.tolist(), columns=['originlats','originlons'], index=geo_.index, dtype=float)
    geo_ = geo_.merge(temp,left_index=True,right_index=True)
    temp = pd.DataFrame(geo_.destination.tolist(), columns=['destlats','destlons'], index=geo_.index, dtype=float)
    geo_ = geo_.merge(temp,left_index=True,right_index=True)
    geo_ = geo_.drop(columns=['origin','destination'])
    
    geo_['distance'] = distance(geo_.originlats,geo_.originlons,geo_.destlats,geo_.destlons)
    geomatrix = pd.DataFrame(geo_['distance'].values.reshape(len(geovalues),len(geovalues)), index=geokeys, columns=geokeys)
    geomatrix = geomatrix.applymap(lambda x: x**np.pi) # one of hyperparameters that can be tuned
    geomatrix = geomatrix.applymap(np.reciprocal)
    geomatrix = geomatrix.replace(np.inf,1)
    np.fill_diagonal(geomatrix.values, 0)
    geomatrix_ = geomatrix.values
    
    days = np.sort(df.day.unique())
    times = np.sort(df.timestamp.unique())
    
    demandmatrix = df.demand.values.reshape(len(geokeys),len(days)*len(times)).T
    demandmatrix_ = demandmatrix @ geomatrix_
    df['neighbordemand'] = demandmatrix_.T.flatten().astype(np.float32)
    
    neighborscaler = MinMaxScaler()
    neighborscaler = neighborscaler.fit(df['neighbordemand'].values.reshape(-1, 1))
    df['neighbordemand'] = neighborscaler.transform(df['neighbordemand'].values.reshape(-1, 1))
    
    df.rename(columns={'demand':'d_d-0_t-0','neighbordemand':'nd_d-0_t-0'},inplace=True)
    
    for i in ['d','nd']:
        for j in list(range(1,6)):
            columnname = i + '_d-0_t' + str(-j)
            df[columnname] = np.where(df.geohash6==df.geohash6.shift(j),df[i + '_d-0_t-0'].shift(j),np.nan)
        for j in list(range(-5,6)):
            columnname = i + '_d-1_t' + str(-j)
            df[columnname] = np.where(df.geohash6==df.geohash6.shift(j+96),df[i + '_d-0_t-0'].shift(j+96),np.nan)
        for j in list(range(-5,6)):
            columnname = i + '_d-7_t' + str(-j)
            df[columnname] = np.where(df.geohash6==df.geohash6.shift(j+(96*7)),df[i + '_d-0_t-0'].shift(j+(96*7)),np.nan)
    
    for j in list(range(-5,0)):
        columnname = 'T' + str(-j)
        df[columnname] = np.where(df.geohash6==df.geohash6.shift(j),df['d_d-0_t-0'].shift(j),np.nan)
    
    df = df.dropna()
    df = df.drop(columns=['geohash6','day','timestamp','day_'])
    
    #df = df[~((df['T1']==0)&(df['T2']==0)&(df['T3']==0)&(df['T4']==0)&(df['T5']==0))]
    
    y = pd.DataFrame()
    y['T1'],y['T2'],y['T3'],y['T4'],y['T5'] = df['T1'],df['T2'],df['T3'],df['T4'],df['T5']
    X = df.drop(columns=['T1','T2','T3','T4','T5'])
    del df
    
    return X, y

def split(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
    
    return X_train, X_test, y_train, y_test



def dense(X, parameters):
    main_input = keras.Input(shape=(X.shape[1],), name='main_input')
    
    normx1 = keras.layers.normalization.BatchNormalization() (main_input)
    encoder1 = keras.layers.Dense(64,kernel_initializer='glorot_normal', activation='selu')(normx1)
    dropencoder1 = keras.layers.core.Dropout(rate = parameters['dropout'])(encoder1)
    encoder2 = keras.layers.Dense(64,kernel_initializer='glorot_normal', activation='selu')(dropencoder1)
    dropencoder2 = keras.layers.core.Dropout(rate = parameters['dropout'])(encoder2)
    normdecoder = keras.layers.normalization.BatchNormalization() (dropencoder2)
    
    main_output = keras.layers.Dense(5, activation='linear', name='main_output')(normdecoder)
    
    model = keras.models.Model(inputs=main_input, outputs=main_output)
    
    return model

def fulltraining(model, X_train, X_test, y_train, y_test, parameters):
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1)
    
    model.compile(optimizer='nadam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    # fit network
    history = model.fit({'main_input': X_train}, y_train, epochs=parameters['epoch'], batch_size=parameters['batch_size'], validation_data=({'main_input': X_test}, y_test), verbose=1, callbacks=[reduce_lr], shuffle=False)
    # plot history
    #plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    return model

def save_model(model):
    model.save_weights('newgrab.h5')

def load_model(model):
    model.load_weights('grab.h5')

    
def predict(model,X):
    # make a prediction
    predraw = model.predict({'main_input': X})
    pred = pd.DataFrame(predraw, columns=['T1','T2','T3','T4','T5'])
    return pred

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())