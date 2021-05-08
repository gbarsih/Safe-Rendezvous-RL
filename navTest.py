import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import geopy.distance as gd

import osmnx as ox
import networkx as nx
import importlib as im
import utils
import random

im.reload(utils)

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

print(gd.distance(coords_1, coords_2).km)

cpus = mp.cpu_count() - 1
ox.config(use_cache=True, log_console=True)
sim_env_dt = 1  # envinronment updates every 1s
EARTH_RADIUS_M = 6_371_009

# filepathuc = "./data/data.graphml"
# G, gdf, gdf_water, gdf_parks = utils.getUCMap()
places = ['Champaign, Illinois, USA', 'Urbana, Illinois, USA']
# places = 'Chicago, Illinois, USA'
G = ox.graph_from_place(places, network_type="drive",simplify=False)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
G = ox.bearing.add_edge_bearings(G)
# filepathcc = "./data/data_cc.graphml"
# G, gdf, gdf_water, gdf_parks = utils.getChampCountyMap(filepathcc)

bbox = (40.1213, 40.0996, -88.1981, -88.2455)

routex = np.random.uniform(low=bbox[0], high=bbox[1], size=2)
routey = np.random.uniform(low=bbox[2], high=bbox[3], size=2)

nroutes = 30

orig = random.sample(list(G), nroutes)
dest = random.sample(list(G), nroutes)

route = []
for i in range(len(orig)):
    try:
        r = ox.shortest_path(G, orig[i], dest[i], weight="travel_time")
        route.append(r)
    except Exception as ex:
        print(ex)

# fig, ax = ox.plot_footprints(gdf, alpha=0.5, show=False, close=False, bgcolor="#ffffff")
# fig, ax = ox.plot_footprints(gdf_parks, ax=ax, alpha=0.5, color="green", show=False, bgcolor="#ffffff")
# fig, ax = ox.plot_footprints(gdf_water, ax=ax, alpha=0.5, color="blue", show=False, bgcolor="#ffffff")
fig, ax = ox.plot_graph(G, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                        bgcolor="#ffffff", show=False)

fig, ax = ox.plot_graph_routes(
    G, ax=ax, routes=route, route_colors=ox.plot.get_colors(len(route)), route_linewidth=3, node_size=0, save=True,
    dpi=600, close=False, show=False
)

plt.show()

route = ox.k_shortest_paths(G,orig[5],dest[5],5,weight="length")
route = list(route)

fig, ax = ox.plot_graph(G, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                        bgcolor="#ffffff", show=False, dpi = 3000,figsize=(20,20))

fig, ax = ox.plot_graph_routes(
    G, ax=ax, routes=route, route_colors=ox.plot.get_colors(len(route)), route_linewidth=3, node_size=0,
    close=False, show=False, dpi = 1000,figsize=(20,20)
); plt.show()



places = ['Champaign, Illinois, USA', 'Urbana, Illinois, USA']
places = 'Chicago, Illinois, USA'       # orig[0]
                                        # Out[177]: 5891694350
                                        # dest[0]
                                        # Out[178]: 2844260418
G = ox.graph_from_place(places, network_type="drive",simplify=False)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
G = ox.bearing.add_edge_bearings(G)

nroutes = 30

orig = random.sample(list(G), nroutes)
dest = random.sample(list(G), nroutes)

route1 = ox.shortest_path(G, orig[0], dest[0], weight="travel_time")
im.reload(utils)
x, y, s, t, d = utils.computeIntervals(G, route1, 0.0);
r = utils.iRoute(x, y, s, t, d, 0, 0, route1)

i = 0

while not r.completed and i < 100000:
    print(r.completed)
    i += 1
    r.step()
    r.printDeg(G)


print(r.local_time)


im.reload(utils)

node_list = []

for idx in range(100,len(route1)-1):
    node_list, an = utils.OneDegSep(G, route1[idx], route1[idx - 1], route1[idx+1])
    if len(node_list) > 1: break


newRoute = utils.reRoute(G,route1[idx],node_list[0],dest[0])
newAlt = utils.reRoute(G,route1[idx],node_list[1],dest[0])

for idxn in range(5,len(newRoute)-1):
    node_listn, ann = utils.OneDegSep(G, newRoute[idxn], newRoute[idxn - 1], newRoute[idxn+1])
    if len(node_listn) > 1: break

newNewRoute = utils.reRoute(G,newRoute[idxn],node_listn[0],dest[0])
routes = [route1, newRoute, newAlt, newNewRoute]

bbox = ox.utils_geo.bbox_from_point((G.nodes[newRoute[0]]['y'], G.nodes[newRoute[0]]['x']), dist=500)

fig, ax = ox.plot_graph(G, node_size=50, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=2,
                        bgcolor="#ffffff", show=False, dpi = 600,figsize=(20,20), bbox=bbox);

fig, ax = ox.plot_graph_routes(
    G, ax=ax, routes=routes, route_colors=ox.plot.get_colors(len(routes)), route_linewidth=10,
    close=False, show=False, save=True, filepath='images/reroute_consequences.png');
ax = utils.scatterGraph(node_list,G,ax)
ax.scatter(G.nodes[route1[idx]]['x'],G.nodes[route1[idx]]['y'],c='red',s=30)
ax.scatter(G.nodes[route1[idx+1]]['x'],G.nodes[route1[idx+1]]['y'],c='blue',s=30)

plt.show()
