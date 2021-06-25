# import pandas as pd
# from numpy import loadtxt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
# from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import geopy.distance as gd
import re

import osmnx as ox
import importlib as im
import utils
# import random
# import time
# import itertools
import pickle
from os import listdir
from os.path import isfile, join

im.reload(utils)

# if torch.cuda.is_available():
#   device = torch.device("cuda:0")
#   print("Cuda Device Available")
#   print("Name of the Cuda Device: ", torch.cuda.get_device_name())
#   print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
city = 'dtchicago'
G = utils.getGraphWithSetting(city)

# with open('CU_graph.pkl', 'rb') as f:
#     G = pickle.load(f)

#########################################################################
## DONT RUN THIS BLOCK IF ALL YOU WANT TO DO IS LEARN OVER THE DATASET ##
nroutes = 100
npairs = 1000
stages = 200
Edetours = 1
pool_size = 50

print(Edetours)
print('-------------------- STARTING POOL PROCESSING --------------------')
print('Pools of', pool_size, 'tasked with processing', npairs * stages, 'jobs, each with batch size', nroutes, 'in',
      stages, 'stages')
directory = 'dataset_'+city+'_max5risk/'
filename = 'res'

with open(directory+city+'_graph.pkl', 'wb') as f:
    pickle.dump(G, f)

file = directory + filename
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
badfiles = []
if len(onlyfiles) > 0:
    idxs = [re.findall("\d+", onlyfiles[i])[0] for i in range(len(onlyfiles)) if onlyfiles[i][0] is 'r']
    if len(idxs) > 0:
        idxs = list(map(int, idxs))
        shiftval = np.max(idxs)
    else:
        shiftval = 0
else:
    shiftval = 0

for i in range(stages):
    file_idx = i + shiftval + 1
    print('------------------ Stage', i + 1, '------------------')
    results = utils.fastBigData(G, city, nroutes, npairs, Edetours, pool_size)
    localfile = file + str(file_idx) + '.pkl'
    print('Saving to', localfile)
    with open(localfile, 'wb') as f:
        pickle.dump(results, f)

    print('Saving successful, opening file', localfile)
    with open(localfile, 'rb') as f:
        res_pkl = pickle.load(f)

    print('File open successful, testing unpack')
    try:
        ro, ri, dd = zip(*res_pkl)
    except Exception as ex:
        print(ex)
        print(localfile + ' encoutered a problem unpacking INVESTIGATE')
        badfiles.append(localfile)
    print('EOF, moving to next stage')

#########################################################################
#########################################################################

## LEARNING BLOCK ##

# directory = 'datasets/'
# filename = 'res'
# file = directory + filename
# onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
#
# routes = []
# risks = []
#
# for file in onlyfiles:
#     localfile = directory+file
#     print('Opening file', localfile)
#     with open(directory+file, 'rb') as f:
#         res_pkl = pickle.load(f)
#     ro, ri = zip(*res_pkl)
#     routes.extend(ro)
#     risks.extend(ri)
#
#
# risks = [risks[i] for i in range(len(risks)) if len(routes[i])>0]
# routes = [routes[i] for i in range(len(routes)) if len(routes[i])>0]
#
# data_size = len(risks)
# split = 0.7
# train_data_size = int(0.7*data_size)
# test_data_size = data_size - train_data_size
#
# oxp = []
# oyp = []
# dxp = []
# dyp = []
# rrv = []
# rrd = []
#
# rp = []
# rxp = []
# ryp = []
#
# risks_treated = [risks[i] for i in range(data_size) if risks[i] < 1e9]
# threshold = np.mean(risks_treated)
# threshold = 1e6
# d_threshold = 500
#
#
#
# for i in range(data_size):
#     o = routes[i][0].orig
#     d = routes[i][0].dest
#     r = routes[i][0].optNode
#     dist = gd.distance((G.nodes[r]['y'], G.nodes[r]['x']), utils.depot).m
#     if len(routes[i]) > 0 and 1e7 / 2 > risks[i] > 0.0 and dist > d_threshold:
#         oxp.append(G.nodes[o]['x'])
#         oyp.append(G.nodes[o]['y'])
#         dxp.append(G.nodes[d]['x'])
#         dyp.append(G.nodes[d]['y'])
#
#         rp.append(r)
#         rxp.append(G.nodes[r]['x'])
#         ryp.append(G.nodes[r]['y'])
#
#         rrv.append(risks[i])
#         rrd.append(risks[i])
#
# ur = 1.0
# lr = 0.0
#
# ubx = np.maximum(np.max(oxp),np.max(dxp))
# lbx = np.minimum(np.min(oxp),np.min(dxp))
# uby = np.maximum(np.max(oyp),np.max(dyp))
# lby = np.minimum(np.min(oyp),np.min(dyp))
# ubr = np.maximum(np.max(rrv),np.max(rrv))
# lbr = np.minimum(np.min(rrv),np.min(rrv))
# oxp = [utils.mapRange(x,lbx,ubx,lr,ur) for x in oxp]
# oyp = [utils.mapRange(x,lby,uby,lr,ur) for x in oyp]
# dxp = [utils.mapRange(x,lbx,ubx,lr,ur) for x in dxp]
# dyp = [utils.mapRange(x,lby,uby,lr,ur) for x in dyp]
# rrv = [utils.mapRange(x,lbr,ubr,lr,ur) for x in rrv]
# rxp = [utils.mapRange(x,lbx,ubx,lr,ur) for x in rxp]
# ryp = [utils.mapRange(x,lby,uby,lr,ur) for x in ryp]
#
# Xtrain = np.array([[oxp[i], oyp[i], dxp[i], dyp[i]] for i in range(len(rrv))])
# # Xtrain = np.abs(Xtrain)
# # Xtrain = Xtrain/np.max(Xtrain)
# Ytrain = np.array(rrv)
# # Ytrain = Ytrain/np.max(Ytrain)
# # Ytrain = np.array(rrd)
#
#
# model = Sequential()
# model.add(Dense(32, input_dim=4, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(1, activation='relu'))
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mean_squared_logarithmic_error")
#
# history = model.fit(Xtrain, Ytrain, epochs=10, batch_size=200, validation_split=0.2)
# plt.figure(figsize=(5, 5), dpi=100)
# plt.plot(history.history["loss"], label="Training Loss", linewidth = 3)
# plt.plot(history.history["val_loss"], label="Validation Loss", linewidth = 3)
# plt.legend()
# plt.show()
#
# test_size = 10
# test = [Xtrain[i] for i in range(test_size)]
#
# predictions = model.predict(Xtrain)
#
#
# for i in range(test_size):
#     pred = predictions[i]*np.max(rrv)
#     print(pred, rrv[i], rrv[i]>threshold, pred/rrv[i])
#
#
# idxs = [i for i in range(len(rrv)) if rrv[i] > 0]
# rrvp = [rrv[i] for i in idxs]
# rxpp = [rxp[i] for i in idxs]
# rypp = [ryp[i] for i in idxs]
#
# preds = [np.abs(predictions[i][0]) for i in range(len(rrvp))]
# perf = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]
# perf_pct = [predictions[i][0]/rrvp[i] for i in range(len(rrvp))]
#
# fig = plt.figure(figsize=(15, 15), dpi=100)
# ax = Axes3D(fig)
# # ax.scatter(rxpp, rypp, rrvp, marker='o', s=10, c=rrvp, alpha=0.1)
# # ax.view_init(elev=90., azim=270)
# ax.scatter(rxpp, rypp, preds, marker='o', s=10, c=perf, alpha=0.03)
# ax.view_init(elev=00., azim=270)
# ax.set_xlabel('$X$')
# ax.set_ylabel('$Y$')
# ax.set_zlabel('$Z$')
# plt.show()