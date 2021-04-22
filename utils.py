import numpy as np
import multiprocessing as mp
import networkx as nx
import matplotlib.pyplot as plt
import math
import geopy.distance as gd

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

print(gd.distance(coords_1, coords_2).km)

import osmnx as ox

cpus = mp.cpu_count() - 1
ox.config(use_cache=True, log_console=True)
sim_env_dt = 0.1  # envinronment updates every 1s
EARTH_RADIUS_M = 6_371_009


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
        _G = ox.graph_from_place("Champaign County, Illinois, USA", network_type="drive",simplify=False)
        print("Saving to file!! ./data/data_cc.graphml")
        ox.save_graphml(_G, "./data/data_cc.graphml")

        _G = ox.add_edge_speeds(_G)
        _G = ox.add_edge_travel_times(_G)
        _G = ox.bearing.add_edge_bearings(_G)
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
        _G = ox.bearing.add_edge_bearings(_G)
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
    s = []
    t = []
    d = []
    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
        if "geometry" in data:
            # if geometry attribute exists, add all its coords to list
            xs, ys = data["geometry"].xy
            x.extend(xs)
            y.extend(ys)
            s.extend([a for a in data.values()][-3] * np.ones(len(xs)))
            t.extend([a for a in data.values()][-2] * np.ones(len(xs)) / len(xs))
        else:
            # otherwise, the edge is a straight line from node to node
            x.extend((G.nodes[u]["x"], G.nodes[v]["x"]))
            y.extend((G.nodes[u]["y"], G.nodes[v]["y"]))
            s.extend([a for a in data.values()][-3] * np.ones(2))
            d.extend([a for a in data.values()][-4] * np.ones(2))
            t.extend([a for a in data.values()][-2] * np.ones(2) / 2)
            # print(data)
            # print([a for a in data.values()][-2])
            # print(t)

    lx = len(x)
    ly = len(y)
    ls = len(s)
    lt = len(t)

    if lx != ly | ls != lx | lt != lx:
        raise Exception("Something went wrong, route data doesnt have the same dimensions")

    return np.array(x), np.array(y), np.array(s), np.array(t), np.array(d)

#a workable route object
class iRoute:
    def __init__(self, x, y, s, v, t, n, d, pos):
        self.x = x  #position
        self.y = y
        self.s = s  #speeds
        self.v = v  #clculated velocity
        self.t = t  #time for transition
        self.n = n  #simulation steps to transition
        self.d = d  #distance to cover
        self.pos = pos  #position (node index)

    def __setattr__(self, name, value):
        if name == 'pos':
            if value[1] > self.n[value[0]]:
                raise Exception(f"Cannot change value of {name}; attempted higher step than allowed.")

        self.__dict__[name] = value


def computeIntervals(G, route, vDev):

    x, y, s, t, d = getRouteData(G, route)
    s = np.array(s) / 3.6  # to m/s

    plt.scatter(x, y);
    plt.show()
    v = []
    n = []
    popi = []
    for i in range(len(x) - 1):
        distance = [x[i + 1] - x[i], y[i + 1] - y[i]]
        norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
        if norm == 0:  # points on top of each other
            popi.append(i)
        else:
            # faster drivers spend less time at any given stretch
            td = t[i]
            if vDev == None:
                td += vDev(td)

            steps = td/sim_env_dt
            steps_rounded = int(round(steps,1))
            print("Time in segment:", td, "Number of steps:", steps, "Rounded:", steps_rounded)
            n.append(steps_rounded)
            direction = [distance[0] / norm, distance[1] / norm]  # normalized direction
            v.append([direction[0] * s[i], direction[1] * s[i]])
    x = np.delete(x, popi)
    y = np.delete(y, popi)
    s = np.delete(s, popi)

    return x, y, s, v, t, n, d

