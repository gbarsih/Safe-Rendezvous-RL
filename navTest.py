import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import geopy.distance as gd

import osmnx as ox
import networkx as nx
import importlib as im
import utils

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

print(gd.distance(coords_1, coords_2).km)

cpus = mp.cpu_count() - 1
ox.config(use_cache=True, log_console=True)
sim_env_dt = 1 #envinronment updates every 1s
EARTH_RADIUS_M = 6_371_009


# filepathuc = "./data/data.graphml"
# G, gdf, gdf_water, gdf_parks = getUcMap()

filepathcc = "./data/data_cc.graphml"
G, gdf, gdf_water, gdf_parks = utils.getChampCountyMap(filepathcc)

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

im.reload(utils)

x, y, s, v, t, n, d = utils.computeIntervals(G,route2,0.0);
droute2 = utils.iRoute(x,y,s,v,t,n,d,(0,0))
print(droute2.n)

cm = droute2.n
cm.append(0)

plt.figure(figsize=(10, 10), dpi=80)
plt.scatter(droute2.x,droute2.y,c=cm,s=100)
#plt.gca().set_aspect('equal', adjustable='box');
#plt.savefig('images/raw_route.png')
plt.gray()
plt.show()

plt.plot([x / max(droute2.d) for x in droute2.d])
plt.plot([x / max(droute2.t) for x in droute2.t])
plt.show()

tr = np.ones(len(droute2.t))

for i in range(len(tr)):
    tr[i] = droute2.t[i] + np.random.randn(1)*np.sqrt(droute2.t[i])/5


plt.plot(droute2.t)
plt.plot(tr)
plt.show()