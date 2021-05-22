import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
import pandas as pd

im.reload(utils)

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)

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

with open('CU_graph.pkl', 'wb') as f:
    pickle.dump(G, f)

directory = '/data/risk_datasets/mixed_data_old_no_dev/'
directory = 'datasets_no_detours/'
filename = 'res'
file = directory + filename
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

routes = []
risks = []

for file in onlyfiles:
    localfile = directory + file
    print('Opening file', localfile)
    with open(directory + file, 'rb') as f:
        res_pkl = pickle.load(f)
    ro, ri = zip(*res_pkl)
    routes.extend(ro)
    risks.extend(ri)

risks = [risks[i] for i in range(len(risks)) if len(routes[i]) > 0]
routes = [routes[i] for i in range(len(routes)) if len(routes[i]) > 0]

data_size = len(risks)
split = 0.7
train_data_size = int(0.7 * data_size)
test_data_size = data_size - train_data_size

oxp = []
oyp = []
dxp = []
dyp = []
rrv = []
rrc = []

rp = []
rxp = []
ryp = []

risks_treated = [risks[i] for i in range(data_size) if risks[i] < 1e9]
threshold = np.mean(risks_treated)
r_threshold = 1e7 /2
d_threshold = 500

for i in range(data_size):
    o = routes[i][0].orig
    d = routes[i][0].dest
    r = routes[i][0].optNode
    dist = gd.distance((G.nodes[r]['y'], G.nodes[r]['x']), utils.depot).m
    if len(routes[i]) > 0 and r_threshold > risks[i] > 0.0 and dist > d_threshold:
        oxp.append(G.nodes[o]['x'])
        oyp.append(G.nodes[o]['y'])
        dxp.append(G.nodes[d]['x'])
        dyp.append(G.nodes[d]['y'])

        rp.append(r)
        rxp.append(G.nodes[r]['x'])
        ryp.append(G.nodes[r]['y'])

        rrv.append(risks[i])
        rrc.append(risks[i])


ur = 1.0
lr = 0.0

ubx = np.maximum(np.max(oxp), np.max(dxp))
lbx = np.minimum(np.min(oxp), np.min(dxp))
uby = np.maximum(np.max(oyp), np.max(dyp))
lby = np.minimum(np.min(oyp), np.min(dyp))
ubr = np.maximum(np.max(rrv), np.max(rrv))
lbr = np.minimum(np.min(rrv), np.min(rrv))
oxp = [utils.mapRange(x, lbx, ubx, lr, ur) for x in oxp]
oyp = [utils.mapRange(x, lby, uby, lr, ur) for x in oyp]
dxp = [utils.mapRange(x, lbx, ubx, lr, ur) for x in dxp]
dyp = [utils.mapRange(x, lby, uby, lr, ur) for x in dyp]
rrv = [utils.mapRange(x, lbr, ubr, lr, ur) for x in rrv]
rxp = [utils.mapRange(x, lbx, ubx, lr, ur) for x in rxp]
ryp = [utils.mapRange(x, lby, uby, lr, ur) for x in ryp]

df = pd.DataFrame(list(zip(oxp, oyp, dxp, dyp, rrv)),
                  columns=['X_origin', 'Y_origin', 'X_destination', 'Y_destination', 'risk'])
Inputs = ['X_origin', 'Y_origin', 'X_destination', 'Y_destination']
Outputs = ['risk']

xo = df['X_origin'].values
yo = df['Y_origin'].values
xd = df['X_destination'].values
yd = df['Y_destination'].values
r = df['risk'].values

inp_stack = torch.tensor(df[Inputs].values, dtype=torch.float64)
out_stack = torch.tensor(df[Outputs].values, dtype=torch.float32)

total_data_entries = len(r)
test_data = int(total_data_entries * .2)

train_inputs = inp_stack[:total_data_entries-test_data].to(device=device)
train_outputs = out_stack[:total_data_entries-test_data].to(device=device)

test_inputs = inp_stack[total_data_entries-test_data:total_data_entries].to(device=device)
test_outputs = out_stack[total_data_entries-test_data:total_data_entries].to(device=device)

rxpt = rxp[total_data_entries-test_data:total_data_entries]
rypt = ryp[total_data_entries-test_data:total_data_entries]

class Model(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

layers = []
for i in range(5):
    layers.append(512)

model = Model(4, 1, layers, p=0.4)
model = nn.DataParallel(model)
model.to(device)
epochs = 5000
num_stints = 100
aggregated_losses = []

loss_function = nn.MSELoss().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for k in range(num_stints):
    print('Stint', k)
    for i in range(epochs):
        i += 1
        y_pred = model(train_inputs.float())
        single_loss = loss_function(y_pred, train_outputs)
        aggregated_losses.append(single_loss)

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()

    print('saving model')
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': single_loss,
    }, 'checkpoints/mdl.pth')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fig = plt.figure(figsize=(10, 10), dpi=100)
plt.plot(aggregated_losses, label="Training Loss", linewidth = 3)
plt.show()

with torch.no_grad():
    model.eval()
    predictions = model(test_inputs.float())
    single_loss = loss_function(predictions, test_outputs)

predictions = predictions.cpu().numpy()
reals = test_outputs.cpu().numpy()

test_size = 10

for i in range(test_size):
    pred = predictions[i]
    print(pred, reals[i], np.abs(pred-reals[i]))


idxs = [i for i in range(len(reals)) if reals[i] > 0]
rrvp = [reals[i] for i in idxs]
rxpp = [rxpt[i] for i in idxs]
rypp = [rypt[i] for i in idxs]

preds = [np.abs(predictions[i][0]) for i in range(len(rrvp))]
perf = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]
perf_pct = [predictions[i][0]/rrvp[i] for i in range(len(rrvp))]
perf_mse = [(predictions[i][0] - rrvp[i])**2 for i in range(len(rrvp))]

pu = np.max(perf)
pl = np.min(perf)

perfp = [utils.mapRange(x, pl, pu, lr, ur) for x in perf]
perf_mse = np.mean(perf_mse)

# fig = plt.figure(figsize=(15, 15), dpi=100)
# ax = Axes3D(fig)
# ax.scatter(rxpp, rypp, rrvp, marker='o', s=10, c='red', alpha=0.01)
# ax.scatter(rxpp, rypp, preds, marker='o', s=10, c=preds, alpha=0.03)
# ax.view_init(elev=00., azim=270)
# ax.set_xlabel('$X$')
# ax.set_ylabel('$Y$')
# ax.set_zlabel('$Z$')
# plt.show()
#
print(np.median(perf))
print(perf_mse)
print(np.max(np.abs(perf)))