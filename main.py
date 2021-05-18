# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
import pandas as pd
from numpy import loadtxt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import geopy.distance as gd
import re

import osmnx as ox
import networkx as nx
import importlib as im
import utils
import random
import time
import itertools
import pickle
from os import listdir
from os.path import isfile, join

im.reload(utils)

# if torch.cuda.is_available():
#   device = torch.device("cuda:0")
#   print("Cuda Device Available")
#   print("Name of the Cuda Device: ", torch.cuda.get_device_name())
#   print("GPU Computational Capablity: ", torch.cuda.get_device_capability())

places = ['Champaign, Illinois, USA', 'Urbana, Illinois, USA']
# places = 'Chicago, Illinois, USA'       # orig[0]
#                                         # Out[177]: 5891694350
#                                         # dest[0]
#                                         # Out[178]: 2844260418
G = ox.graph_from_place(places, network_type="drive", simplify=False)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
G = ox.bearing.add_edge_bearings(G)
G = ox.utils_graph.get_largest_component(G, strongly=True)

nroutes = 1
npairs = 1000
stages = 100
Edetours = 0.0
pool_size = 63

print('-------------------- STARTING POOL PROCESSING --------------------')
print('Pools of', pool_size, 'tasked with processing', npairs * stages, 'jobs, each with batch size', nroutes, 'in',
      stages, 'stages')
directory = 'datasets/'
filename = 'res'
file = directory + filename
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
if len(onlyfiles) > 0:
    idxs = [re.findall("\d+", onlyfiles[i])[0] for i in range(len(onlyfiles))]
    idxs = list(map(int, idxs))
    shiftval = np.max(idxs)
else:
    shiftval = 0

for i in range(stages):
    file_idx = i + shiftval + 1
    print('------------------ Stage', i + 1, '------------------')
    results = utils.fastBigData(G, nroutes, npairs, Edetours, pool_size)
    localfile = file + str(file_idx) + '.pkl'
    print('Saving to', localfile)
    with open(localfile, 'wb') as f:
        pickle.dump(results, f)

    print('Saving successful, opening file', localfile)
    with open(localfile, 'rb') as f:
        res_pkl = pickle.load(f)

    print('File open successful, EOF')


# with open('D1.pkl', 'rb') as f:
#     res_pkl = pickle.load(f)
# routes, risks = zip(*res_pkl)
#
directory = 'datasets/'
filename = 'res'
file = directory + filename
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

routes = []
risks = []

for file in onlyfiles:
    localfile = directory+file
    print('Opening file', localfile)
    with open(directory+file, 'rb') as f:
        res_pkl = pickle.load(f)
    ro, ri = zip(*res_pkl)
    routes.extend(ro)
    risks.extend(ri)


data_size = len(risks)
split = 0.7
train_data_size = int(0.7*data_size)
test_data_size = data_size - train_data_size

oxp = []
oyp = []
dxp = []
dyp = []
rrv = []
rrc = []

risks_treated = [risks[i] for i in range(data_size) if risks[i] < 1e9]
threshold = np.mean(risks_treated)
threshold = 1e6

for i in range(data_size):
    if len(routes[i]) > 0 and risks[i] < 1e7/2:
        o = routes[i][0].orig
        d = routes[i][0].dest
        oxp.append(G.nodes[o]['x'])
        oyp.append(G.nodes[o]['y'])
        dxp.append(G.nodes[d]['x'])
        dyp.append(G.nodes[d]['y'])
        rrv.append(risks[i])
        if risks[i] > threshold:
            rrc.append(1.0)
        else:
            rrc.append(0.0)

ur = 1.0
lr = 0.0

ubx = np.maximum(np.max(oxp),np.max(dxp))
lbx = np.minimum(np.min(oxp),np.min(dxp))
uby = np.maximum(np.max(oyp),np.max(dyp))
lby = np.minimum(np.min(oyp),np.min(dyp))
ubr = np.maximum(np.max(rrv),np.max(rrv))
lbr = np.minimum(np.min(rrv),np.min(rrv))
oxp = [utils.mapRange(x,lbx,ubx,lr,ur) for x in oxp]
oyp = [utils.mapRange(x,lby,uby,lr,ur) for x in oyp]
dxp = [utils.mapRange(x,lbx,ubx,lr,ur) for x in dxp]
dyp = [utils.mapRange(x,lby,uby,lr,ur) for x in dyp]
rrv = [utils.mapRange(x,lbr,ubr,lr,ur) for x in rrv]

Xtrain = np.array([[oxp[i], oyp[i], dxp[i], dyp[i]] for i in range(len(rrv))])
# Xtrain = np.abs(Xtrain)
# Xtrain = Xtrain/np.max(Xtrain)
Ytrain = np.array(rrv)
# Ytrain = Ytrain/np.max(Ytrain)
# Ytrain = np.array(rrc)


model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(32, input_dim=4, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512, input_dim=4, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mean_absolute_error", metrics=["accuracy"])

history = model.fit(Xtrain, Ytrain, epochs=50, batch_size=200, validation_split=0.1)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(history.history["loss"], label="Training Loss", linewidth = 3)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth = 3)
plt.legend()
plt.show()

predictions = model.predict(Xtrain)

for i in range(10):
    pred = predictions[i]*np.max(rrv)
    print(pred, rrv[i], rrv[i]>threshold, pred/rrv[i])

