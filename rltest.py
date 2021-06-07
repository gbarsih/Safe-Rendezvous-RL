import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torchviz import make_dot
# from torch.utils.data import DataLoader
import numpy as np
# from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import matplotlib as mpl
import geopy.distance as gd
# import re
import seaborn as sns;

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")
sns.color_palette("tab10")
sns.color_palette("crest", as_cmap=True)

import osmnx as ox
import networkx as nx
import importlib as im
import geopandas as gpd
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, Polygon
import utils
# import random
import time
# import itertools
import pickle
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
# from multiprocessing import Pool as ThreadPool
# import multiprocessing as mp
im.reload(utils)

figsize = (25,15)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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

city = 'champaign'
G = utils.getGraphWithSetting(city)

# with open('CU_graph.pkl', 'wb') as f:
#     pickle.dump(G, f)
#
# with open('CU_graph.pkl', 'rb') as f:
#     G = pickle.load(f)

directory = '/data/risk_datasets/mixed_data_old_no_dev/'
directory = 'dataset_'+city+'_max5risk/'
filename = 'res'
file = directory + filename
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) and f[0] is 'r']

routes = []
risks = []
deltas = []
for file in onlyfiles:
    localfile = directory + file
    print('Opening file', localfile)
    with open(directory + file, 'rb') as f:
        res_pkl = pickle.load(f)


    ro, ri, dd = zip(*res_pkl)
    deltas.extend(dd)
    routes.extend(ro)
    risks.extend(ri)


# risks = [risks[i] for i in range(len(risks)) if len(routes[i]) > 0]
# routes = [routes[i] for i in range(len(routes)) if len(routes[i]) > 0]
# deltas = [deltas[i] for i in range(len(deltas)) if len(routes[i]) > 0]

data_size = len(risks)
split = 0.7
train_data_size = int(0.7 * data_size)
test_data_size = data_size - train_data_size

oxp = []
oyp = []
dxp = []
dyp = []
rrv = []
rrd = []

rp = []
rxp = []
ryp = []

risks_treated = [risks[i] for i in range(data_size) if risks[i] < 2.5e9]
threshold = np.mean(risks_treated)
r_threshold = 2.5e9
d_threshold = 500
deltas_threshold = np.percentile(deltas, 99.9)

depot_local = utils.getDepotLocation(city)
depot_node = ox.get_nearest_nodes(G,[depot_local[1]],[depot_local[0]])

for i in range(data_size):
    o = routes[i].orig
    d = routes[i].dest
    r = routes[i].optNode
    dist = gd.distance((G.nodes[r]['y'], G.nodes[r]['x']), depot_local).m
    if len(routes[i].nodes) > 0 and deltas[i] < deltas_threshold and dist > d_threshold:
        oxp.append(G.nodes[o]['x'])
        oyp.append(G.nodes[o]['y'])
        dxp.append(G.nodes[d]['x'])
        dyp.append(G.nodes[d]['y'])

        rp.append(r)
        rxp.append(G.nodes[r]['x'])
        ryp.append(G.nodes[r]['y'])

        rrv.append(risks[i])
        rrd.append(deltas[i])

ur = 1.0
lr = -0.0

ubx = np.maximum(np.max(oxp), np.max(dxp))
lbx = np.minimum(np.min(oxp), np.min(dxp))
uby = np.maximum(np.max(oyp), np.max(dyp))
lby = np.minimum(np.min(oyp), np.min(dyp))
ubr = np.maximum(np.max(rrv), np.max(rrv))
lbr = np.minimum(np.min(rrv), np.min(rrv))
ubd = np.maximum(np.max(rrv), np.max(rrv))
lbd = np.minimum(np.min(rrd), np.min(rrd))
oxp = [utils.mapRange(x, lbx, ubx, lr, ur) for x in oxp]
oyp = [utils.mapRange(x, lby, uby, lr, ur) for x in oyp]
dxp = [utils.mapRange(x, lbx, ubx, lr, ur) for x in dxp]
dyp = [utils.mapRange(x, lby, uby, lr, ur) for x in dyp]
rrv = [utils.mapRange(x, lbr, ubr, lr, ur) for x in rrv]
rxp = [utils.mapRange(x, lbx, ubx, lr, ur) for x in rxp]
ryp = [utils.mapRange(x, lby, uby, lr, ur) for x in ryp]
ddv = [utils.mapRange(x, lbd, ubd, lr, ur) for x in rrd]

coord_bounds = {'ubx': ubx, 'lbx': lbx, 'uby': uby, 'lby': lby, 'ur': ur, 'lr': lr, 'lbd': lbd, 'ubd': ubd}

# df = pd.DataFrame(list(zip(oxp, oyp, dxp, dyp, rrv)),
#                   columns=['X_origin', 'Y_origin', 'X_destination', 'Y_destination', 'risk'])
# Inputs = ['X_origin', 'Y_origin', 'X_destination', 'Y_destination']
# Outputs = ['risk']
df = pd.DataFrame(list(zip(oxp, oyp, dxp, dyp, rrv)),
                  columns=['X_origin', 'Y_origin', 'X_destination', 'Y_destination', 'deltas'])
Inputs = ['X_origin', 'Y_origin', 'X_destination', 'Y_destination']
Outputs = ['deltas']

xo = df['X_origin'].values
yo = df['Y_origin'].values
xd = df['X_destination'].values
yd = df['Y_destination'].values
r = df['deltas'].values

inp_stack = torch.tensor(df[Inputs].values, dtype=torch.float64)
out_stack = torch.tensor(df[Outputs].values, dtype=torch.float32)

total_data_entries = len(r)
test_data = int(total_data_entries * .1)

train_inputs = inp_stack[:total_data_entries - test_data].to(device=device)
train_outputs = out_stack[:total_data_entries - test_data].to(device=device)

test_inputs = inp_stack[total_data_entries - test_data:total_data_entries].to(device=device)
test_outputs = out_stack[total_data_entries - test_data:total_data_entries].to(device=device)

rxpt = rxp[total_data_entries - test_data:total_data_entries]
rypt = ryp[total_data_entries - test_data:total_data_entries]



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
for i in range(2):
    layers.append(128)

# layers.append(32)

model = Model(4, 1, layers, p=0.7)
model = nn.DataParallel(model)
model.to(device)
epochs = 5000
num_stints = 1
aggregated_losses = []

loss_function = nn.MSELoss().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# checkpoint = torch.load('checkpoints/mdl_10_512_max5risk.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss_function = checkpoint['loss']
# model.to(device)
# model.eval()

model.train()
for k in range(num_stints):
    print('Stint', k)
    start = time.time()
    for i in range(epochs):
        i += 1
        y_pred = model(train_inputs.float())
        single_loss = loss_function(y_pred, train_outputs)
        aggregated_losses.append(single_loss)

        if i % 25 == 1:
            end = time.time()
            elapsed = end - start
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f} time: {elapsed:10.3f}')
            start = time.time()

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

fig = plt.figure(figsize=figsize, dpi=100)
plt.plot(aggregated_losses, label="Training Loss", linewidth=3)
# plt.ylim((0, np.percentile(aggregated_losses,90).cpu().detach().numpy()))
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title(directory)
plt.show()

with torch.no_grad():
    model.eval()
    predictions = model(test_inputs.float())
    # single_loss = loss_function(predictions, test_outputs)

predictions = predictions.cpu().numpy()
reals = test_outputs.cpu().numpy()

test_size = len(reals)

for i in range(10):
    pred = predictions[i]
    print(pred, reals[i], np.abs(pred - reals[i]))

idxs = [i for i in range(len(reals)) if reals[i] > 0]
rrvp = [reals[i] for i in idxs]
rxpp = [rxpt[i] for i in idxs]
rypp = [rypt[i] for i in idxs]

# preds = [np.abs(predictions[i][0]) for i in range(len(rrvp))]
# perf = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]
# perf_pct = [predictions[i][0] / rrvp[i] for i in range(len(rrvp))]
# perf_mse = [(predictions[i][0] - rrvp[i]) ** 2 for i in range(len(rrvp))]

preds = [np.abs(predictions[i][0]) for i in range(len(rrvp))]
perf = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]
perf_pct = [predictions[i][0] / rrvp[i] for i in range(len(rrvp))]
perf_mse = [(predictions[i][0] - rrvp[i]) ** 2 for i in range(len(rrvp))]

pu = np.max(perf)
pl = np.min(perf)

perfp = [utils.mapRange(x, pl, pu, lr, ur) for x in perf]
perf_mse = np.mean(perf_mse)

fig = plt.figure(figsize=figsize, dpi=100)
ax = Axes3D(fig)
ax.scatter(rxpp, rypp, rrvp, marker='o', s=10, c='blue', alpha=0.1)
ax.scatter(rxpp, rypp, preds, marker='o', s=10, c='red', alpha=0.1)
ax.view_init(elev=90., azim=270)
# ax.set_zlim(lr, 0.1)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Risk$')
ax.set_title(directory)
plt.show()

print('median', np.median(perf))
print('mse', perf_mse)
print('max e', np.max(np.abs(perf)))

## plot a heatmap of risks
im.reload(utils)
fig = plt.figure(figsize=figsize, dpi=100)
suptitle_font = {'fontsize': 40, 'fontweight': 'normal', 'y': 0.98}
tts = 'Risk Assessment for Edge Destinations'+directory
fig.suptitle('Risk Assessment for Edge Destinations', **suptitle_font)
ax221 = fig.add_subplot(221)
ax222 = fig.add_subplot(222)
ax223 = fig.add_subplot(223)
ax224 = fig.add_subplot(224)
c = None
rp = ur
rm = lr
vmin = None
vmax = 0.1
predictions = utils.TwoDPredictions(model, rm, rp, device)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax221, cbar=True,
            center=c, cbar_kws={'label': 'Risk'}, vmin=vmin)
ax221.set_xlabel('Longitude')
ax221.set_ylabel('Latitude')
predictions = utils.TwoDPredictions(model, rp, rp, device)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax222, cbar=True,
            center=c, cbar_kws={'label': 'Risk'}, vmin=vmin)
ax222.set_xlabel('Longitude')
ax222.set_ylabel('Latitude')
predictions = utils.TwoDPredictions(model, rm, rm, device)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax223, cbar=True,
            center=c, cbar_kws={'label': 'Risk'}, vmin=vmin)
ax223.set_xlabel('Longitude')
ax223.set_ylabel('Latitude')
predictions = utils.TwoDPredictions(model, rp, rm, device)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax224, cbar=True,
            center=c, cbar_kws={'label': 'Risk'}, vmin=vmin)
ax224.set_xlabel('Longitude')
ax224.set_ylabel('Latitude')
plt.show()

#now plot on top of graph
im.reload(utils)
Gc = utils.getGraphWithSetting(city)

nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
fig = plt.figure(figsize=(15, 15), dpi=100)
ax221 = fig.add_subplot(221)
ax222 = fig.add_subplot(222)
ax223 = fig.add_subplot(223)
ax224 = fig.add_subplot(224)

ns = 20

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, 0.0, 1.0, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r')
ox.plot_graph(G, ax = ax221, node_color=nc, node_size=ns, edge_linewidth=0.5, figsize=(15,15),
                        bgcolor='white', show=False, close=False)
ax221 = utils.scatterGraph(depot_node, G, ax221, color='blue')

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, 1.0, 1.0, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r')
ox.plot_graph(G, ax = ax222, node_color=nc, node_size=ns, edge_linewidth=0.5, figsize=(15,15),
                        bgcolor='white', show=False, close=False)
ax222 = utils.scatterGraph(depot_node, G, ax222, color='blue')

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, 0.0, 0.0, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r')
ox.plot_graph(G, ax = ax223, node_color=nc, node_size=ns, edge_linewidth=0.5, figsize=(15,15),
                        bgcolor='white', show=False, close=False)
ax223 = utils.scatterGraph(depot_node, G, ax223, color='blue')

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, 1.0, 0.0, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r')
ox.plot_graph(G, ax = ax224, node_color=nc, node_size=ns, edge_linewidth=0.5, figsize=(15,15),
                        bgcolor='white', show=False, close=False)
ax224 = utils.scatterGraph(depot_node, G, ax224, color='blue')


plt.show()

#plot the isochrones
network_type = 'drive'
trip_times = [5, 10, 15, 20] #in minutes
travel_speed = 15 #driving speed in km/hour
G = utils.getGraphWithSetting(city)
depot_local = utils.getDepotLocation(city)
gdf_nodes = ox.graph_to_gdfs(G, edges=False)
center_node = ox.get_nearest_node(G, depot_local)
G = ox.project_graph(G)

meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
for u, v, k, data in G.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

iso_colors = ox.plot.get_colors(n=len(trip_times), cmap='plasma', start=0, return_hex=True)

isochrone_polys = []
for trip_time in sorted(trip_times, reverse=True):
    subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance='time')
    node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
    bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
    isochrone_polys.append(bounding_poly)

# #plot isochrone and heatmap side by side

# im.reload(utils)
# fig = plt.figure(figsize=figsize, dpi=100)
# suptitle_font = {'fontsize': 40, 'fontweight': 'normal', 'y': 0.98}
# tts = 'Risk Assessment for Edge Destinations'+directory
# fig.suptitle('Risk Assessment for Edge Destinations', **suptitle_font)
# ax211 = fig.add_subplot(121)
# ax212 = fig.add_subplot(122)
# predictions = utils.TwoDPredictions(model, rm, rp, device)
# sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax211, vmin=vmin, cbar=False,
#             center=c)
#
# fig2, ax = ox.plot_graph(G, ax=ax212, show=False, close=False, edge_color="k", edge_alpha=0.2,
#                         node_size=0, bgcolor="#ffffff")
# for polygon, fc in zip(isochrone_polys, iso_colors):
#     patch = PolygonPatch(polygon, fc=fc, ec='none', alpha=0.4, zorder=-1)
#     ax.add_patch(patch)
# plt.show()


