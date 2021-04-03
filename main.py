import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import multiprocessing as mp
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

from animate import Animator
import matplotlib
# matplotlib.use('Qt4Agg')
from cars import Cars, TrafficLights
# import convergent_learner as cl
from matplotlib import animation
import osmnx as ox
import simulation as sim

cpus = mp.cpu_count() - 1

ox.config(use_cache=True, log_console=True)


def getUcMap(_filepath=None, _gdf_filepath=None):
    ox.config(use_cache=True, log_console=True)
    _gdf1 = ox.geometries_from_place("Urbana, Illinois, USA", {"building": True})
    _gdf_parks1 = ox.geometries_from_place("Urbana, Illinois, USA", {"leisure": 'park'})
    _gdf_water1 = ox.geometries_from_place("Urbana, Illinois, USA", {"water": True})
    _gdf2 = ox.geometries_from_place("Champaign, Illinois, USA", {"building": True})
    _gdf_parks2 = ox.geometries_from_place("Champaign, Illinois, USA", {"leisure": 'park'})
    _gdf_water2 = ox.geometries_from_place("Champaign, Illinois, USA", {"water": True})
    _gdf = _gdf1.append(_gdf2)
    _gdf_water = _gdf_water1.append(_gdf_water2)
    _gdf_parks = _gdf_parks1.append(_gdf_parks2)
    if _filepath is None:
        _G1 = ox.graph_from_place("Urbana, Illinois, USA", network_type="drive")
        _G2 = ox.graph_from_place("Champaign, Illinois, USA", network_type="drive")
        _G = nx.compose(_G1, _G2)
        print("Saving to file!!")
        ox.save_graphml(_G, filepathuc)
        return _G, _gdf, _gdf_water, _gdf_parks

    else:
        print("Loading graph from file!!")
        _filepath = "./data/data.graphml"
        _G = ox.load_graphml(filepath)

        return _G, _gdf, _gdf_water, _gdf_parks


def getChampCountyMap(_filepath=None):
    ox.config(use_cache=True, log_console=True)
    _gdf = ox.geometries_from_place("Champaign County, Illinois, USA", {"building": True})
    _gdf_parks = ox.geometries_from_place("Champaign County, Illinois, USA", {"leisure": 'park'})
    _gdf_water = ox.geometries_from_place("Champaign County, Illinois, USA", {"water": True})
    if _filepath is None:
        _G = ox.graph_from_place("Champaign County, Illinois, USA", network_type="drive")
        print("Saving to file!!")
        ox.save_graphml(_G, filepathcc)

        _G = ox.add_edge_speeds(_G)
        _G = ox.add_edge_travel_times(_G)

        return _G, _gdf, _gdf_water, _gdf_parks

    else:
        print("Loading graph from file!!")
        _G = ox.load_graphml(_filepath)

        # edges = ox.graph_to_gdfs(_G, nodes=False)
        # edges["highway"] = edges["highway"].astype(str)
        # edges.groupby("highway")[["length", "speed_kph", "travel_time"]].mean().round(1)
        # hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
        _G = ox.add_edge_speeds(_G)
        _G = ox.add_edge_travel_times(_G)

        return _G, _gdf, _gdf_water, _gdf_parks


def shortest_path(G, orig, dest):
    try:
        return ox.shortest_path(G, orig, dest, weight="travel_time")
    except Exception:
        # for unsolvable routes (due to directed graph perimeter effects)
        return None


def getRouteData(G, route):
    x = []
    y = []
    speed = []
    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
        if "geometry" in data:
            # if geometry attribute exists, add all its coords to list
            xs, ys = data["geometry"].xy
            x.extend(xs)
            y.extend(ys)
            speed.extend([a for a in zip(data.values())][-2][0]*np.ones(len(xs)))
        else:
            # otherwise, the edge is a straight line from node to node
            x.extend((G.nodes[u]["x"], G.nodes[v]["x"]))
            y.extend((G.nodes[u]["y"], G.nodes[v]["y"]))
            speed.extend([a for a in zip(data.values())][-2][0]*np.ones(1))
    return x, y, speed
 # TODO: fix size mismatch between x,y and speed

# filepathuc = "./data/data.graphml"
# G, gdf, gdf_water, gdf_parks = getUcMap()

filepathcc = "./data/data_cc.graphml"
G, gdf, gdf_water, gdf_parks = getChampCountyMap(filepathcc)

bbox = (40.1213, 40.0996, -88.1981, -88.2455)

routex = np.random.uniform(low=bbox[0], high=bbox[1], size=2)
routey = np.random.uniform(low=bbox[2], high=bbox[3], size=2)

orig = ox.get_nearest_node(G, (routex[0], routey[0]))
dest = ox.get_nearest_node(G, (routex[1], routey[1]))

route1 = ox.shortest_path(G, orig, dest, weight="length")
route2 = ox.shortest_path(G, orig, dest, weight="travel_time")

fig, ax = ox.plot_footprints(gdf, alpha=0.5, show=False, close=False, bgcolor="#ffffff", bbox=bbox)
fig, ax = ox.plot_footprints(gdf_parks, ax=ax, alpha=0.5, color="green", show=False, bgcolor="#ffffff", bbox=bbox)
fig, ax = ox.plot_footprints(gdf_water, ax=ax, alpha=0.5, color="blue", show=False, bgcolor="#ffffff")
fig, ax = ox.plot_graph(G, ax=ax, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                        bgcolor="#ffffff", show=False, bbox=bbox)

fig, ax = ox.plot_graph_routes(
    G, ax=ax, routes=[route1, route2], route_colors=["r", "y"], route_linewidth=6, node_size=0, save=True, dpi=3000,
    bbox=bbox, close=False, show=False
)

plt.show()

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Hyper-parameters
# # input_size = 784 # 28x28
# num_classes = 10
# num_epochs = 10
# batch_size = 100
# learning_rate = 0.001
#
# input_size = 28
# sequence_length = 28
# hidden_size = 256
# num_layers = 2
#
# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='./data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='./data',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
#
#
# # Fully connected neural network with one hidden layer
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes,p=0.3):
#         super(RNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         # -> x needs to be: (batch_size, seq, input_size)
#
#         # or:
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         self.drop = nn.Dropout(p=p)
#
#     def forward(self, x):
#         # Set initial hidden states (and cell states for LSTM)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#
#         # x: (n, 28, 28), h0: (2, n, 128)
#
#         # Forward propagate RNN
#         out, _ = self.gru(x, h0)
#         out = self.drop(out)
#         # or:
#         # out, _ = self.lstm(x, (h0,c0))
#
#         # out: tensor of shape (batch_size, seq_length, hidden_size)
#         # out: (n, 28, 128)
#
#         # Decode the hidden state of the last time step
#         out = out[:, -1, :]
#         # out: (n, 128)
#
#         out = self.fc(out)
#         # out: (n, 10)
#         return out
#
#
# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
#
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Train the model
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # origin shape: [N, 1, 28, 28]
#         # resized: [N, 28, 28]
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
#
# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')
