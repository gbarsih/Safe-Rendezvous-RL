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
import seaborn as sns
import random

# sns.set_theme()
# sns.set_style("whitegrid")
sns.set_context("paper")
# sns.color_palette("tab10")
# sns.color_palette("crest", as_cmap=True)

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

im.reload(utils)

figsize = utils.figsize
dpi = utils.dpi
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

city = 'champaign'  # chicago, dtchicago, champaign
G = utils.getGraphWithSetting(city)

risks, deltas, routes = utils.getDatasetFromCity(city)

oxp, oyp, dxp, dyp, rrv, rxp, ryp, ddv, coord_bounds, depot_local, depot_node = utils.unPackDataset(city, G, risks,
                                                                                                    deltas, routes)

ur = coord_bounds['ur']

df = pd.DataFrame(list(zip(oxp, oyp, dxp, dyp, rrv, ddv)),
                  columns=['X_origin', 'Y_origin', 'X_destination', 'Y_destination', 'risks', 'deltas'])
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

n_l = 3
n_n = 128
d_p = 0.2
lr = 0.001
act = "SELU" #dtchicago soa: 3-32-0.2-GELU
epochs = 1000

class ModelSELU(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.SELU())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

class ModelGELU(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.GELU())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

class ModelReLU(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

class ModelSigmoid(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.Sigmoid())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

layers = []
for i in range(n_l):
    layers.append(n_n)  # 32-128 range with low dropout (0.1-0.4) seems to work the best

# layers.append(32)

if act == 'SELU':
    model = ModelSELU(4, 1, layers, p=d_p)
elif act == 'ReLU':
    model = ModelReLU(4, 1, layers, p=d_p)
elif act == 'Sigmoid':
    model = ModelSigmoid(4, 1, layers, p=d_p)
elif act == 'GELU':
    model = ModelGELU(4, 1, layers, p=d_p)

model = nn.DataParallel(model)
model.to(device)
num_stints = 1
aggregated_losses = []

loss_function = nn.MSELoss().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        if i % 100 == 1:
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

# fig = plt.figure(figsize=figsize, dpi=100)
# plt.plot(aggregated_losses, label="Training Loss", linewidth=3)
# # plt.ylim((0, np.percentile(aggregated_losses,90).cpu().detach().numpy()))
# plt.xlabel('Epoch')
# plt.ylabel('Training Loss')
# plt.title(directory)
# plt.show()

with torch.no_grad():
    model.eval()
    predictions = model(test_inputs.float())
    # single_loss = loss_function(predictions, test_outputs)

predictions = predictions.cpu().numpy()
reals = test_outputs.cpu().numpy()

bias, optv = utils.findOptSamp(reals,predictions,0.001,0.01,99.99)

diffs = [np.abs(np.abs(reals[i] - predictions[i])-bias)  for i in range(len(reals))]
mx = np.argmin(diffs)
# mx = 0

pcts = []
acc = 0
tot = 0
diffs = []
rths = np.percentile(reals, 99.99)
for i in range(len(reals)):
    if reals[i] < rths:
        pcts.append(np.abs(reals[i] - predictions[i]) / predictions[i])
        tot += 1
        if (reals[mx]>=reals[i] and predictions[mx]>=predictions[i]) or (reals[mx]<reals[i] and predictions[mx]<predictions[i]):
            acc += 1
        else:
            d1 = np.abs(predictions[i] - predictions[mx]) / predictions[i]
            d2 = np.abs(predictions[i] - predictions[mx]) / predictions[mx]
            diffs.append(np.minimum(d1, d2))

pcts = np.stack(pcts)
print(np.mean(diffs))
pct_err = acc/tot
print(pct_err)

acc = 0
tot = 0
cases = 10000
diffs_ran = []
rths = np.percentile(reals, 99.99)
for i in range(cases):
    i1 = random.randint(0, len(reals) - 1)
    i2 = random.randint(0, len(reals) - 1)
    if reals[i1] < rths and reals[i2] < rths:
        tot += 1
        if (reals[i1] > reals[i2] and predictions[i1] > predictions[i2]) or (
                reals[i1] < reals[i2] and predictions[i1] < predictions[i2]):
            acc += 1
        else:
            d1 = np.abs(predictions[i1]-predictions[i2])/predictions[i1]
            d2 = np.abs(predictions[i1]-predictions[i2])/predictions[i2]
            diffs_ran.append(np.minimum(d1,d2))

perf_ran = acc/tot
print(np.mean(diffs_ran))
print(perf_ran)
pct_err = np.maximum(perf_ran, pct_err)
mean_rank_err = np.minimum(np.mean(diffs_ran), np.mean(diffs))

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
perf_mae = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]

pu = np.max(perf)
pl = np.min(perf)

perfp = [utils.mapRange(x, pl, pu, lr, ur) for x in perf]
perf_mse = np.mean(perf_mse)

# fig = plt.figure(figsize=figsize, dpi=100)
# ax = Axes3D(fig)
# ax.scatter(rxpp, rypp, rrvp, marker='o', s=10, c='blue', alpha=0.1)
# ax.scatter(rxpp, rypp, preds, marker='o', s=10, c='red', alpha=0.1)
# ax.view_init(elev=00., azim=270)
# # ax.set_zlim(lr, 0.5)
# ax.set_xlabel('$X$')
# ax.set_ylabel('$Y$')
# ax.set_zlabel('$Risk$')
# ax.set_title(directory)
# plt.show()


## plot a heatmap of risks
im.reload(utils)
fig = plt.figure(figsize=utils.figsize, dpi=utils.dpi)
suptitle_font = {'fontsize': 14, 'fontweight': 'normal', 'y': 0.98}
fig.suptitle(utils.getCityName(city) + ', $n_l$: {}, $n_n$: {}, $p_d$: {}, Perf: {:4.2f}%'.format(n_l, n_n, d_p, pct_err*100),
             **suptitle_font)
ax221 = fig.add_subplot(221)
ax222 = fig.add_subplot(222)
ax223 = fig.add_subplot(223)
ax224 = fig.add_subplot(224)
c = None
rp = 0.75
rm = 0.25
vmin = None
vmax = 0.1
numticks = 100
depot_x, depot_y = utils.LatLonToUnit(depot_local[0], depot_local[1], coord_bounds)
predictions = utils.TwoDPredictions(model, rm, rp, device, numticks)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax221, cbar=True,
            center=c, cbar_kws={'label': 'Risk', 'location': 'top', 'use_gridspec': False}, vmin=vmin)
ax221.scatter(rm * numticks, (ur - rp) * numticks, s=100, color=utils.dest_color)
ax221.scatter(depot_x * numticks, (ur - depot_y) * numticks, s=100, color=utils.depot_color)
ax221.set_xlabel('Longitude')
ax221.set_ylabel('Latitude')

predictions = utils.TwoDPredictions(model, rp, rp, device, numticks)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax222, cbar=True,
            center=c, cbar_kws={'label': 'Risk', 'location': 'top', 'use_gridspec': False}, vmin=vmin)
ax222.scatter(rp * numticks, (ur - rp) * numticks, s=100, color=utils.dest_color)
ax222.scatter(depot_x * numticks, (ur - depot_y) * numticks, s=100, color=utils.depot_color)
ax222.set_xlabel('Longitude')
ax222.set_ylabel('Latitude')

predictions = utils.TwoDPredictions(model, rm, rm, device, numticks)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax223, cbar=True,
            center=c, cbar_kws={'label': 'Risk', 'location': 'top', 'use_gridspec': False}, vmin=vmin)
ax223.scatter(rm * numticks, (ur - rm) * numticks, s=100, color=utils.dest_color)
ax223.scatter(depot_x * numticks, (ur - depot_y) * numticks, s=100, color=utils.depot_color)
ax223.set_xlabel('Longitude')
ax223.set_ylabel('Latitude')

predictions = utils.TwoDPredictions(model, rp, rm, device, numticks)
sns.heatmap(predictions, square=True, xticklabels=False, yticklabels=False, ax=ax224, cbar=True,
            center=c, cbar_kws={'label': 'Risk', 'location': 'top', 'use_gridspec': False}, vmin=vmin)
ax224.scatter(rp * numticks, (ur - rm) * numticks, s=100, color=utils.dest_color)
ax224.scatter(depot_x * numticks, (ur - depot_y) * numticks, s=100, color=utils.depot_color)
ax224.set_xlabel('Longitude')
ax224.set_ylabel('Latitude')

filepath = 'images/' + city + 'heat.png'
plt.savefig(filepath)

plt.show()

# now plot on top of graph
im.reload(utils)
Gc = utils.getGraphWithSetting(city)

nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
fig = plt.figure(figsize=utils.figsize, dpi=utils.dpi)
suptitle_font = {'fontsize': 14, 'fontweight': 'normal', 'y': 0.98}
ax221 = fig.add_subplot(221)
ax222 = fig.add_subplot(222)
ax223 = fig.add_subplot(223)
ax224 = fig.add_subplot(224)

ns = utils.ns

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, rm, rp, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r', cmap=sns.color_palette("rocket", as_cmap=True))
ox.plot_graph(G, ax=ax221, node_color=nc, node_size=ns, edge_linewidth=utils.edge_lw, figsize=utils.figsize,
              bgcolor='white', show=False, close=False)
ax221 = utils.scatterGraph(depot_node, G, ax221, color=utils.depot_color)
lon_d, lat_d = utils.UnitToLatLon(rm, rp, coord_bounds)
dest_node = ox.get_nearest_nodes(G, [lon_d], [lat_d])
ax221 = utils.scatterGraph(dest_node, G, ax221, color=utils.dest_color)

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, rp, rp, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r', cmap=sns.color_palette("rocket", as_cmap=True))
ox.plot_graph(G, ax=ax222, node_color=nc, node_size=ns, edge_linewidth=utils.edge_lw, figsize=utils.figsize,
              bgcolor='white', show=False, close=False)
ax222 = utils.scatterGraph(depot_node, G, ax222, color=utils.depot_color)
lon_d, lat_d = utils.UnitToLatLon(rp, rp, coord_bounds)
dest_node = ox.get_nearest_nodes(G, [lon_d], [lat_d])
ax222 = utils.scatterGraph(dest_node, G, ax222, color=utils.dest_color)

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, rm, rm, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r', cmap=sns.color_palette("rocket", as_cmap=True))
ox.plot_graph(G, ax=ax223, node_color=nc, node_size=ns, edge_linewidth=utils.edge_lw, figsize=utils.figsize,
              bgcolor='white', show=False, close=False)
ax223 = utils.scatterGraph(depot_node, G, ax223, color=utils.depot_color)
lon_d, lat_d = utils.UnitToLatLon(rm, rm, coord_bounds)
dest_node = ox.get_nearest_nodes(G, [lon_d], [lat_d])
ax223 = utils.scatterGraph(dest_node, G, ax223, color=utils.dest_color)

G = Gc

predictions, node_list = utils.InferNodeRiskMultiple(G, nodes.index, rp, rm, model, device, coord_bounds)
nx.set_node_attributes(G, {nodes.index[i]: predictions[i][0] for i in range(len(predictions))}, name='r')
nc = ox.plot.get_node_colors_by_attr(G, attr='r', cmap=sns.color_palette("rocket", as_cmap=True))
ox.plot_graph(G, ax=ax224, node_color=nc, node_size=ns, edge_linewidth=utils.edge_lw, figsize=utils.figsize,
              bgcolor='white', show=False, close=False)
ax224 = utils.scatterGraph(depot_node, G, ax224, color=utils.depot_color)
lon_d, lat_d = utils.UnitToLatLon(rp, rm, coord_bounds)
dest_node = ox.get_nearest_nodes(G, [lon_d], [lat_d])
ax224 = utils.scatterGraph(dest_node, G, ax224, color=utils.dest_color)

filepath = 'images/' + city + 'map_heat_untitled.png'
plt.savefig(filepath)

fig.suptitle(utils.getCityName(city) + ', $n_l$: {}, $n_n$: {}, $p_d$: {}, Perf: {:4.2f}%'.format(n_l, n_n, d_p, pct_err*100),
             **suptitle_font)

filepath = 'images/' + city + 'map_heat.png'
plt.savefig(filepath)

plt.show()

# #plot the isochrones
# network_type = 'drive'
# trip_times = [5, 10, 15, 20] #in minutes
# travel_speed = 15 #driving speed in km/hour
# G = utils.getGraphWithSetting(city)
# depot_local = utils.getDepotLocation(city)
# gdf_nodes = ox.graph_to_gdfs(G, edges=False)
# center_node = ox.get_nearest_node(G, depot_local)
# G = ox.project_graph(G)
#
# meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
# for u, v, k, data in G.edges(data=True, keys=True):
#     data['time'] = data['length'] / meters_per_minute
#
# iso_colors = ox.plot.get_colors(n=len(trip_times), cmap='plasma', start=0, return_hex=True)
#
# isochrone_polys = []
# for trip_time in sorted(trip_times, reverse=True):
#     subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance='time')
#     node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
#     bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
#     isochrone_polys.append(bounding_poly)

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

print('Median', np.median(perf))
print('MSE', perf_mse)
print('MAE', np.mean(perf_mae))
print('Max Err', np.max(np.abs(perf)))
print('Pct Err', np.mean(pcts))
print('Pct Err adjusted', pct_err)
print('Rank Err', mean_rank_err)

import csv

fields = [city, n_l, n_n, act, d_p, epochs, perf_mse, np.mean(perf_mae)]
print(fields)

with open(r'log_hyper.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    # writer.writerow(["city", "n_layers", "n_neurons", "activation", "dropout", "epochs", "mse", "mae"])
