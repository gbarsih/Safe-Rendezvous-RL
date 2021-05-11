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


# def getUcMap(_filepath=None, _gdf_filepath=None):
#     ox.config(use_cache=True, log_console=True)
#     _gdf1 = ox.geometries_from_place("Urbana, Illinois, USA", {"building": True})
#     _gdf_parks1 = ox.geometries_from_place("Urbana, Illinois, USA", {"leisure": 'park'})
#     _gdf_water1 = ox.geometries_from_place("Urbana, Illinois, USA", {"water": True})
#     _gdf2 = ox.geometries_from_place("Champaign, Illinois, USA", {"building": True})
#     _gdf_parks2 = ox.geometries_from_place("Champaign, Illinois, USA", {"leisure": 'park'})
#     _gdf_water2 = ox.geometries_from_place("Champaign, Illinois, USA", {"water": True})
#     _gdf = _gdf1.append(_gdf2)
#     _gdf_water = _gdf_water1.append(_gdf_water2)
#     _gdf_parks = _gdf_parks1.append(_gdf_parks2)
#     if _filepath is None:
#         _G1 = ox.graph_from_place("Urbana, Illinois, USA", network_type="drive")
#         _G2 = ox.graph_from_place("Champaign, Illinois, USA", network_type="drive")
#         _G = nx.compose(_G1, _G2)
#         print("Saving to file!!")
#         ox.save_graphml(_G, filepathuc)
#         return _G, _gdf, _gdf_water, _gdf_parks
#
#     else:
#         print("Loading graph from file!!")
#         _filepath = "./data/data.graphml"
#         _G = ox.load_graphml(filepath)
#
#         return _G, _gdf, _gdf_water, _gdf_parks


def getChampCountyMap(_filepath=None):
    ox.config(use_cache=True, log_console=True)
    _gdf = ox.geometries_from_place("Champaign County, Illinois, USA", {"building": True})
    _gdf_parks = ox.geometries_from_place("Champaign County, Illinois, USA", {"leisure": 'park'})
    _gdf_water = ox.geometries_from_place("Champaign County, Illinois, USA", {"water": True})
    if _filepath is None:
        _G = ox.graph_from_place("Champaign County, Illinois, USA", network_type="drive", simplify=False)
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


def getUCMap(_filepath=None):
    ox.config(use_cache=True, log_console=True)
    places = ['Champaign, Illinois, USA', 'Urbana, Illinois, USA']
    _gdf = ox.geometries_from_place(places, {"building": True})
    _gdf_parks = ox.geometries_from_place(places, {"leisure": 'park'})
    _gdf_water = ox.geometries_from_place(places, {"water": True})
    if _filepath is None:
        _G = ox.graph_from_place(places, network_type="drive", simplify=False)
        print("Saving to file!! ./data/data_cc.graphml")
        ox.save_graphml(_G, "./data/data_cc.graphml")

        _G = ox.add_edge_speeds(_G)
        _G = ox.add_edge_travel_times(_G)
        _G = ox.bearing.add_edge_bearings(_G)
        return _G, _gdf, _gdf_water, _gdf_parks

    else:
        print("Loading graph from file!!")
        _G = ox.load_graphml(_filepath)
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
            t.extend([a for a in data.values()][-2] * np.ones(len(xs)))
        else:
            # otherwise, the edge is a straight line from node to node
            x.extend((G.nodes[u]["x"], G.nodes[v]["x"]))
            y.extend((G.nodes[u]["y"], G.nodes[v]["y"]))
            s.extend([a for a in data.values()][-3] * np.ones(2))
            d.extend([a for a in data.values()][-4] * np.ones(2))
            t.extend([a for a in data.values()][-2] * np.ones(1))
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


# a workable route object
class iRoute:

    def __init__(self, G, orig, dest, p=0.0, t0=0.0, pos=0):
        self.route = ox.shortest_path(G, orig, dest, weight="travel_time")
        x, y, s, t, d = getRouteData(G, self.route)
        # self.x = x  # position
        # self.y = y
        # self.s = s  # speeds
        # # self.v = v  # calculated velocity
        self.t = t  # time for transition
        # self.n = n  # simulation steps to transition
        # self.d = d  # distance to cover
        self.pos = pos  # position (node index)
        # self.gt0 = t0
        self.edge_timer = 0.0
        self.local_time = 0.0
        self.dt = sim_env_dt
        # self.route_steps = sum(self.n)
        self.lt = len(self.t)
        self.completed = False
        self.nodes = [orig]
        self.times = [t0]
        self.p = p
        self.orig = orig
        self.dest = dest

    # this function progresses to the next node
    def progress(self, G):

        #check if there's a possibility of rerouting
        if self.pos >= 0:
            node_list, an = OneDegSep(G, self.route[self.pos], self.route[self.pos-1], self.route[self.pos+1])
            if len(node_list) > 1 and np.random.uniform(0,1) < self.p:
                # print("rerouting!!",node_list, self.pos)
                self.route = reRoute(G, self.route[self.pos], node_list[np.random.randint(0,len(node_list))], self.dest)
                x, y, s, t, d = computeIntervals(G, self.route, 0.0);
                self.t = t  # time for transition
                self.pos = 0
                self.lt = len(self.t)
            else:
                self.pos += 1  # advance one node

        self.nodes.append(self.route[self.pos])
        self.times.append(self.local_time)
        self.edge_timer = self.dt * 0.1  # 1 if progressing to the next node is an action
        return True

    # this function advances one time step
    def step(self, G):
        self.edge_timer += self.dt
        self.local_time += self.dt
        stat = None
        if self.edge_timer >= self.t[self.pos] and self.pos < self.lt-1 and self.completed == False:
            #print("Pos:", self.pos, "Travel Time:", self.t[self.pos], "Edge Time:", self.edge_timer, "Local Time:",
            #      self.local_time)
            stat = self.progress(G)
        elif self.pos == self.lt-1 and self.completed == False:
            self.completed = True
            #print("At end of route, pos=", self.pos, "length of t=", self.lt, "Completed:", self.completed)

        return stat

    def printDeg(self, G):
        if self.pos < len(self.route):
            print("Node deg:", G.nodes[self.route[self.pos]]["street_count"])


def computeIntervals(G, route, vDev):
    x, y, s, t, d = getRouteData(G, route)
    s = np.array(s) / 3.6  # to m/s
    # # plt.scatter(x, y)
    # # plt.show()
    # v = []
    # n = []
    # popi = []
    # for i in range(len(x) - 1):
    #     distance = [x[i + 1] - x[i], y[i + 1] - y[i]]
    #     norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
    #     if norm == 0:  # points on top of each other
    #         popi.append(i)
    #     # else:
    #     #     # faster drivers spend less time at any given stretch
    #     #     td = t[i]
    #     #     # if vDev is None:
    #     #     #     td += vDev(td)
    #     #
    #     #     steps = td / sim_env_dt
    #     #     steps_rounded = int(round(steps, 1))
    #     #     # print("Time in segment:", td, "Number of steps:", steps, "Rounded:", steps_rounded)
    #     #     n.append(int(np.maximum(0, steps_rounded)))
    #     #     direction = [distance[0] / norm, distance[1] / norm]  # normalized direction
    #     #     v.append([direction[0] * s[i], direction[1] * s[i]])
    # x = np.delete(x, popi)
    # y = np.delete(y, popi)
    # s = np.delete(s, popi)

    return x, y, s, t, d


def scatterGraph(node_list, G: nx.classes.multidigraph.MultiDiGraph, ax, color='green'):
    for i in range(len(node_list)):
        ax.scatter(G.nodes[node_list[i]]['x'], G.nodes[node_list[i]]['y'], c=color, s=10)

    return ax


def OneDegSep(G: nx.classes.multidigraph.MultiDiGraph, node, prev_node=None, next_node=None):
    an = nx.generators.ego_graph(G, node)
    node_list = list(an.nodes)
    node_list.remove(node)
    if prev_node is not None:
        try:
            node_list.remove(prev_node)
        except Exception as ex:
            node_list

    if next_node is not None:
        node_list.remove(next_node)

    return node_list, an


def reRoute(G, prev_node, curr_node, dest):
    # Finds a new route that doesn't go through any of the past nodes as defined in argin

    try:
        temp = G[curr_node][prev_node][0]['travel_time']
        G[curr_node][prev_node][0]['travel_time'] = 1e6
        newRoute = ox.shortest_path(G, curr_node, dest, weight="travel_time")
        G[curr_node][prev_node][0]['travel_time'] = temp
    except Exception as ex:
        print("street is one-way, no weight assgntm")
        newRoute = ox.shortest_path(G, curr_node, dest, weight="travel_time")

    return newRoute

def iRouteIterator(routes,G):
    for i in range(len(routes)):
        # print("stepping route", i)
        routes[i].step(G)

    return routes

def compRoute(r,G):
    i = 0
    while not r.completed and i < 1e10:
        i += 1
        r.step(G)
        # r.printDeg(G)

def CheckCompletion(r, verbose=False):
    comp = []
    for x in r:
        if verbose:
            print(x.completed)
        comp.append(x.completed)

    return all(comp)

def GatherRoutes(r):
    nodes = []
    times = []
    for x in r:
        nodes.append(x.nodes)
        times.append(x.times)

    return nodes, times

