import multiprocessing as mp
import networkx as nx
import geopy.distance as gd
import pandas as pd
import random
from multiprocessing import Pool as ThreadPool
import itertools
import time
import datetime
import seaborn as sns
import pickle
from os import listdir
from os.path import isfile, join
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
from matplotlib import pyplot as plt

import csv

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

depot = (40.11237526379417, -88.24327192934085)
depot_champaign = (40.11237526379417, -88.24327192934085)
depot_chicago = (41.84477746620278, -87.68189749403619)
depot_dtchicago = (41.88325889448088, -87.66564849555748)
depot_rio = (-22.922332034177217, -43.19078181012658)

depot_color = 'green'
dest_color = 'cyan'

figsize = (6, 6)
dpi = 200
edge_lw = 0.5
route_lw = 5
node_lw = 1
ns = 5

print(gd.distance(coords_1, coords_2).km)

import osmnx as ox

cpus = mp.cpu_count() - 1
ox.config(use_cache=True, log_console=True)
sim_env_dt = 1.0  # environment updates every 1s
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
    # for u, v in zip(route[:-1], route[1:]):
    #     # if there are parallel edges, select the shortest in length
    #     data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
    #     if "geometry" in data:
    #         # if geometry attribute exists, add all its coords to list
    #         xs, ys = data["geometry"].xy
    #         x.extend(xs)
    #         y.extend(ys)
    #         s.extend([a for a in data.values()][-3] * np.ones(len(xs)))
    #         t.extend([a for a in data.values()][-2] * np.ones(len(xs)))
    #     else:
    #         # otherwise, the edge is a straight line from node to node
    #         x.append(G.nodes[u]["x"])
    #         y.append(G.nodes[u]["y"])
    #         s.extend([a for a in data.values()][-3] * np.ones(2))
    #         d.extend([a for a in data.values()][-4] * np.ones(2))
    #         t.extend([a for a in data.values()][-2] * np.ones(1))
    #         # print(data)
    #         # print([a for a in data.values()][-2])
    #         # print(t)

    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
        x.append(G.nodes[u]["x"])
        y.append(G.nodes[u]["y"])
        s.extend([a for a in data.values()][-3] * np.ones(2))
        d.extend([a for a in data.values()][-4] * np.ones(2))
        t.extend([a for a in data.values()][-2] * np.ones(1))

    lx = len(x)
    ly = len(y)
    ls = len(s)
    lt = len(t)

    if lx != ly | ls != lx | lt != lx:
        raise Exception("Something went wrong, route data doesnt have the same dimensions")

    return x, y, s, t, d


# a workable route object
class iRoute:

    def __init__(self, G, city, orig, dest, p=0.0, t0=0.0, pos=0):
        self.route = ox.shortest_path(G, orig, dest, weight="travel_time")
        x, y, s, t, d = getRouteData(G, self.route)
        self.x = x  # position
        self.y = y
        self.t = t  # time for transition
        self.optNode = computeOptRdvNode(x, y, t, city)
        self.tr = self.t[self.optNode]
        self.optNode = self.route[self.optNode]
        self.pos = pos  # position (node index)
        self.edge_timer = 0.0
        self.local_time = 0.0
        self.dt = sim_env_dt
        self.lt = len(self.t)
        self.completed = False
        self.nodes = [orig]
        self.times = [t0]
        self.t_0 = t0
        self.p = p
        self.orig = orig
        self.dest = dest
        self.risk = 0.0
        self.deviated = False

    # this function progresses to the next node
    def progress(self, G, local_state=None):

        # check if there's a possibility of rerouting
        if self.pos >= 0:
            node_list, an = OneDegSep(G, self.route[self.pos], self.route[self.pos - 1], self.route[self.pos + 1])
            if local_state is not None:
                jump = local_state.uniform(0, 1) < self.p and len(self.route) > 2 and self.lt - 1 > self.pos > 1
            else:
                jump = np.random.uniform(0, 1) < self.p and len(self.route) > 2 and self.lt - 1 > self.pos > 1

            if len(node_list) > 1 and jump:
                # print("rerouting!!",node_list, self.pos)
                newRoute = reRoute(G, self.route[self.pos], node_list[np.random.randint(0, len(node_list))],
                                   self.dest)
                if len(newRoute) > 3:
                    self.deviated = True
                    self.route = newRoute
                    x, y, s, t, d = computeIntervals(G, self.route, 0.0);
                    self.t = t  # time for transition
                    self.pos = 0
                    self.lt = len(self.t)
                else:
                    self.pos += 1  # ignore new route, too short
            else:
                self.pos += 1  # advance one node

        self.nodes.append(self.route[self.pos])
        self.times.append(self.t[self.pos] + self.t_0)
        # if len(self.t) < self.pos:
        #     self.times.append(self.t[self.pos] + self.t_0)
        # elif len(self.t) > 0:
        #     self.times.append(self.t[len(self.t)-1] + self.t_0)
        # else:
        #     print('comp_status', self.completed)
        #     print('t', self.t)
        #     print('pos', self.pos)
        #     print('nodes', self.nodes)
        #     print('route', self.route)
        #     pprint(locals())
        #     raise Exception('null length in t')
        # self.edge_timer = self.dt * 0.1  # 1 if progressing to the next node is an action
        return True

    # this function advances one time step
    def step(self, G, local_state=None):
        # self.edge_timer += self.dt
        # self.local_time += self.dt
        stat = None

        if self.pos < self.lt - 1 and self.completed == False:
            stat = self.progress(G, local_state)
        elif self.pos == self.lt - 1 and self.completed == False:
            self.completed = True

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
        ax.scatter(G.nodes[node_list[i]]['x'], G.nodes[node_list[i]]['y'], c=color, s=100)

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
        # print("street is one-way, no weight assgntm")
        newRoute = ox.shortest_path(G, curr_node, dest, weight="travel_time")

    return newRoute


def iRouteIterator(routes, G):
    for i in range(len(routes)):
        # print("stepping route", i)
        routes[i].step(G)

    return routes


def compRoute(r, G, seed=123):
    local_state = np.random.RandomState(seed)
    # random.seed(seed)
    i = 0
    while not r.completed and i < 1e3:
        i += 1
        r.step(G, local_state)
        # r.printDeg(G)

    return r


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


def nDetourOpportunities(r, G):
    l = len(r)
    d = 0
    for i in range(1, l - 1):
        node_list, an = OneDegSep(G, r[i], r[i - 1], r[i + 1])
        if len(node_list) > 1:
            d += 1

    return d


def clip(x, l, u):
    return l if x < l else u if x > u else x


def routeTotalDetourProbality(G, city, orig, dest, Edetours, t_0=0.0):
    nom_route = ox.shortest_path(G, orig, dest, weight='travel_time')
    d = nDetourOpportunities(nom_route, G)
    if d > 2:
        # E[detours] = p*d -> p = Edetours/d
        p = Edetours / d
        p = clip(p, 0, 1)
    else:
        p = 0.0

    return iRoute(G, city, orig, dest, p, t_0)


def createDataSetPar(G, nroutes=100, Edetours=1, sparse=False):
    pool = ThreadPool(cpus)
    orig = random.sample(list(G), nroutes)
    dest = random.sample(list(G), nroutes)

    if not sparse:
        orig = [orig[0] for i in range(0, nroutes)]
        dest = [dest[0] for i in range(0, nroutes)]

    t_0 = 0.0

    routes = pool.starmap(createRoute,
                          zip(itertools.repeat(G), orig, dest, itertools.repeat(Edetours), itertools.repeat(t_0)))
    routes = [i for i in routes if i]
    nroutes = len(routes)
    results = pool.starmap(compRoute, zip(routes, itertools.repeat(G), np.random.randint(2000, size=nroutes)))
    pool.close()
    return results


def createBigData(G, nroutes=100, npairs=100, Edetours=1, pool_size=cpus):
    # pool = ThreadPool(30)
    orig = random.sample(list(G), npairs)  # orig/dest pairs
    dest = random.sample(list(G), npairs)
    t_0 = 0.0

    routes = []
    risks = []
    loop_start = time.time()
    for i in range(0, npairs):
        start = time.time()
        results, risk = routeRiskBatch(G, orig[i], dest[i], Edetours, t_0, nroutes, pool_size)
        routes.append(results)
        risks.append(risk)
        end = time.time()
        elapsed = end - start
        print("Finished iteration ", i, " of ", npairs, " in ", elapsed, "s")

    loop_end = time.time()
    elapsed = loop_end - loop_start
    print("Finished everything  in ", elapsed)
    # pool.close()
    return routes, risks


def routeRiskBatch(G, o, d, Edetours, t_0, nroutes, pool_size=cpus):
    pool_size = np.minimum(pool_size, cpus)
    pool = ThreadPool(pool_size)
    o = [o for i in range(0, nroutes)]
    d = [d for i in range(0, nroutes)]
    routes = pool.starmap(createRoute,
                          zip(itertools.repeat(G), o, d, itertools.repeat(Edetours), itertools.repeat(t_0)))
    routes = [i for i in routes if i]
    results = pool.starmap(compRoute, zip(routes, itertools.repeat(G)))
    # optnodes, opttimes, E, idxs, risk = computeCompositeRisk(results, G)
    risk = getRiskValue(results, G)
    pool.close()

    return results, risk


def routeRiskSingle(G, o, d, Edetours, t_0, nroutes, city):
    # print('routeRiskSingle start:', city)
    routes = []
    start = time.time()
    invalid = False

    for i in range(nroutes):
        try:
            if i == 0:
                Edetours_l = 0.0
            else:
                Edetours_l = Edetours

            routes.append(routeTotalDetourProbality(G, city, o, d, Edetours_l, t_0))
        except Exception as ex:
            print(ex)
            print("Skipping this dest/orig pair")
            invalid = True
            routes = []
            break

    nroutes = len(routes)
    risk = 1e10
    if not invalid:
        for i in range(nroutes):
            compRoute(routes[i], G, np.random.randint(1, 1e5))

        risk = getRiskValue(routes, G, city)
        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now(), "Finished batch of risk computation in", elapsed)
        delta = risk - getSingleRouteCost(o, d, G, city)
        # pprint(locals())
        return routes[0], risk, delta

    else:
        return


def fastBigData(G, city, nroutes=100, npairs=100, Edetours=1, pool_size=cpus):
    # print('fastBigData', city)
    pool_size = np.minimum(pool_size, cpus)
    pool = ThreadPool(pool_size)
    orig_b = random.choices(list(G), k=npairs)  # orig/dest pairs
    dest_b = random.choices(list(G), k=npairs)

    # instead of doing this, lets compute djikstra

    # y_o = [G.nodes[node]['y'] for node in orig_b]
    # x_o = [G.nodes[node]['x'] for node in orig_b]
    # y_d = [G.nodes[node]['y'] for node in dest_b]
    # x_d = [G.nodes[node]['x'] for node in dest_b]

    print('Selecting good routes')

    idxs = [i for i in range(npairs) if len(ox.shortest_path(G, orig_b[i], dest_b[i], weight="travel_time")) > 5]

    orig = [orig_b[i] for i in idxs]
    dest = [dest_b[i] for i in idxs]

    print('Selected', len(idxs), 'pairs for evaluation out of a possible', npairs)

    t_0 = 0.0
    start = time.time()
    results = pool.starmap(routeRiskSingle,
                           zip(itertools.repeat(G), orig, dest, itertools.repeat(Edetours), itertools.repeat(t_0),
                               itertools.repeat(nroutes), itertools.repeat(city)))
    # results = routeRiskSingle(G,orig[0],dest[0],Edetours,t_0,nroutes,city)
    pool.close()
    end = time.time()
    elapsed = end - start
    print("Finished everything in", elapsed, city)
    return results


def createDataSetSingle(G, nroutes=100, Edetours=1, sparse=False):
    orig = random.sample(list(G), nroutes)
    dest = random.sample(list(G), nroutes)
    t_0 = 0.0
    routes = []

    if not sparse:
        orig = [orig[0] for i in range(0, nroutes)]
        dest = [dest[0] for i in range(0, nroutes)]

    for i in range(nroutes):
        try:
            routes.append(routeTotalDetourProbality(G, orig[i], dest[i], Edetours, t_0))
        except Exception as ex:
            print(ex)

    nroutes = len(routes)

    for i in range(nroutes):
        compRoute(routes[i], G)

    return routes


def createRoute(G, orig, dest, Edetours, t_0):
    try:
        return routeTotalDetourProbality(G, orig, dest, Edetours, t_0)
    except Exception as ex:
        print(ex)


def plotRoutes(routes, nroutes, G, color='r', save=False, filepath='images/routes.png'):
    routes = routes[0:nroutes - 1]
    fig, ax = ox.plot_graph(G, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                            bgcolor="#ffffff", show=False, dpi=600, figsize=(20, 20))

    fig, ax = ox.plot_graph_routes(
        G, ax=ax, routes=routes, route_colors=color, route_linewidth=3, node_size=0,
        route_alpha=0.1,
        close=False, show=False, dpi=600, figsize=(20, 20), save=save, filepath=filepath
    );
    plt.show()

    return fig, ax


def costFun(d, t):
    alpha = 1
    t = np.maximum(t, 0.1)
    return (d ** 2) / t + alpha * t
    # return d


def computeOptRdvNode(x, y, t, city):
    # pprint(locals())
    # print('computeOptRdvNode', city)
    depot_local = getDepotLocation(city)

    n = len(x) - 1
    E = []
    tr = np.cumsum(t)
    # print(len(x), len(t), len(tr))
    for i in range(n):
        d = gd.distance((y[i], x[i]), depot_local).m
        e = costFun(d, tr[i])
        E.append(e)

    if len(E) == 0:
        return None
    else:
        return np.argmin(E)


def computeRendezvousEnergy(nodes, times, G):
    n = len(nodes)
    if n != len(times):
        print(len(times), n)
        raise Exception("Input dim error computeRendezvousEnergy")
    E = []
    for node, time in zip(nodes, times):
        y = G.nodes[node]['y']
        x = G.nodes[node]['x']
        d = gd.distance((y, x), depot).m
        e = costFun(d, time)
        E.append(e)

    return E


def getxy(node, G):
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]
    return x, y


def getDepotNodeDistance(node, G):
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]


def computeCompositeRisk(routes, G, city):
    # print('computeCompositeRisk', city)
    # we are given many routes, we want to compute the risk of this set of routes.
    # we need the optimum rdv node and time for ever singe one:
    optnodes = []
    opttimes = []
    idxs = []
    for r in routes:
        x = []
        y = []
        for i in r.nodes:
            xn, yn = getxy(i, G)
            x.append(xn)
            y.append(yn)

        # compute distances to depot
        # print('just before computeOptRdvNode:', city)
        idx = computeOptRdvNode(x, y, r.times, city)
        if idx is not None:
            idxs.append(idx)
            optnode = r.nodes[idx]
            optnodes.append(optnode)
            opttime = r.times[idx]
            opttimes.append(opttime)

        # p1 = merge(y,x)
        # result = list(map(gd.distance, p1, itertools.repeat(depot)))
        # d = [o.m for o in result]

    # now we can compute some risk measures.
    E = computeRendezvousEnergy(optnodes, opttimes, G)
    n_max_values = 5
    maxel = sorted(range(len(E)), key=lambda k: E[k])[-n_max_values:]
    # print("energies in cvar", [E[i] for i in maxel])
    if len(E) == 0:
        print(routes[0].nodes)

    risk = np.mean([E[i] for i in maxel])

    if len(maxel) == 0:
        risk = 1e10

    return optnodes, opttimes, E, idxs, risk


def getSingleRouteCost(o, d, G, city):
    r = iRoute(G, city, o, d)
    compRoute(r, G)
    optnodes = []
    opttimes = []
    idxs = []
    x = []
    y = []
    for i in r.nodes:
        xn, yn = getxy(i, G)
        x.append(xn)
        y.append(yn)

    idx = computeOptRdvNode(x, y, r.t, city)
    idxs.append(idx)
    optnodes.append(r.nodes[idx])
    opttimes.append(r.times[idx])
    E = computeRendezvousEnergy(optnodes, opttimes, G)

    return min(E)


def getRiskValue(routes, G, city):
    # print('getRiskValue start:', city)
    optnodes, opttimes, E, idxs, risk = computeCompositeRisk(routes, G, city)
    return risk


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


def merge4(list1, list2, list3, list4):
    merged_list = [(list1[i], list2[i], list3[i], list4[i]) for i in range(0, len(list1))]
    return merged_list


def mapRange(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def collectDataCriteria(route, risk, G, d_threshold, r_threshold):
    o = route[0].orig
    d = route[0].dest
    r = route[0].optNode
    dist = gd.distance((G.nodes[r]['y'], G.nodes[r]['x']), depot).m
    if len(route) > 0 and r_threshold > risk > 0.0 and dist > d_threshold:
        oxp = G.nodes[o]['x']
        oyp = G.nodes[o]['y']
        dxp = G.nodes[d]['x']
        dyp = G.nodes[d]['y']

        rp = r
        rxp = G.nodes[r]['x']
        ryp = G.nodes[r]['y']

        rrv = risk
        return oxp, oyp, dxp, dyp, rrv, rp, rxp, ryp


def collectDataMP(routes, risks, G, data_size, d_threshold, r_threshold, pool_size=cpus):
    pool_size = np.minimum(pool_size, cpus)
    pool = ThreadPool(pool_size)
    results = pool.starmap(collectDataCriteria, zip(routes, risks, itertools.repeat(G), itertools.repeat(d_threshold),
                                                    itertools.repeat(r_threshold)))

    pool.close()

    return results


def TwoDPredictions(model, destx, desty, device, numticks=10, lr=0.0, ur=1.0):
    xvals = np.tile(np.linspace(lr, ur, numticks), numticks)
    yvals = np.linspace(lr, ur, numticks)
    yvals = np.transpose([yvals] * numticks)
    yvals = yvals.flatten()

    destx = destx * np.ones(numticks ** 2)
    desty = desty * np.ones(numticks ** 2)

    ds = list(zip(xvals, yvals, destx, desty))

    eval_set = torch.tensor(ds, dtype=torch.float64).to(device=device)
    print(eval_set.shape)
    with torch.no_grad():
        model.eval()
        predictions = model(eval_set.float())
        predictions = predictions.cpu().numpy()

    return predictions.reshape(numticks, numticks)[::-1]


def InferNodeRisk(G, node, destx, desty, model, device, coord_bounds):
    xval = G.nodes[node]['x']
    yval = G.nodes[node]['y']
    xval = mapRange(xval, coord_bounds['lbx'], coord_bounds['ubx'], coord_bounds['lr'], coord_bounds['ur'])
    yval = mapRange(yval, coord_bounds['lby'], coord_bounds['uby'], coord_bounds['lr'], coord_bounds['ur'])
    ds = list(zip([xval], [yval], [destx], [desty]))
    eval_set = torch.tensor(ds, dtype=torch.float64).to(device=device)
    with torch.no_grad():
        model.eval()
        prediction = model(eval_set.float()).cpu().numpy()

    return prediction


def InferNodeRiskMultiple(G, nodes, destx, desty, model, device, coord_bounds):
    xvals = [G.nodes[node]['x'] for node in nodes]
    yvals = [G.nodes[node]['y'] for node in nodes]
    xvals = [mapRange(x, coord_bounds['lbx'], coord_bounds['ubx'], coord_bounds['lr'], coord_bounds['ur']) for x in
             xvals]
    yvals = [mapRange(y, coord_bounds['lby'], coord_bounds['uby'], coord_bounds['lr'], coord_bounds['ur']) for y in
             yvals]
    destx = destx * np.ones(len(xvals))
    desty = desty * np.ones(len(xvals))
    ds = list(zip(xvals, yvals, destx, desty))
    eval_set = torch.tensor(ds, dtype=torch.float64).to(device=device)
    with torch.no_grad():
        model.eval()
        predictions = model(eval_set.float()).cpu().numpy()

    return predictions, nodes


def getGraphWithSetting(city):
    if city is 'champaign':
        places = ['Champaign, Illinois, USA', 'Urbana, Illinois, USA']
        # places = 'Chicago, Illinois, USA'       # orig[0]
        #                                         # Out[177]: 5891694350
        #                                         # dest[0]
        #                                         # Out[178]: 2844260418
        G = ox.graph_from_place(places, network_type="drive", simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.bearing.add_edge_bearings(G)
        G = ox.utils_graph.get_largest_component(G, strongly=True)

    elif city is 'chicago':
        places = ['Chicago, Illinois, USA']
        G = ox.graph_from_place(places, network_type="drive", simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.bearing.add_edge_bearings(G)
        G = ox.utils_graph.get_largest_component(G, strongly=True)

    elif city is 'janeiro':
        places = ['Rio de Janeiro, Rio de Janeiro, Brazil']
        G = ox.graph_from_place(places, network_type="drive", simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.bearing.add_edge_bearings(G)
        G = ox.utils_graph.get_largest_component(G, strongly=True)

    elif city is 'dtchicago':
        places = ['201 E Randolph St, Chicago, IL, 60602']
        G = ox.graph_from_address(places, network_type="drive", simplify=True, dist=5000)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.bearing.add_edge_bearings(G)
        G = ox.utils_graph.get_largest_component(G, strongly=True)

    return G


def getGdfsFeature(city):
    if city is 'champaign':
        places = ['Champaign, Illinois, USA', 'Urbana, Illinois, USA']
        _gdf = ox.geometries_from_place(places, {"building": True})
        _gdf_parks = ox.geometries_from_place(places, {"leisure": 'park'})
        _gdf_water = ox.geometries_from_place(places, {"water": True})

    elif city is 'chicago':
        places = ['Chicago, Illinois, USA']
        _gdf = ox.geometries_from_place(places, {"building": True})
        _gdf_parks = ox.geometries_from_place(places, {"leisure": 'park'})
        _gdf_water = ox.geometries_from_place(places, {"water": True})

    elif city is 'janeiro':
        places = ['Rio de Janeiro, Rio de Janeiro, Brazil']
        _gdf = ox.geometries_from_place(places, {"building": True})
        _gdf_parks = ox.geometries_from_place(places, {"leisure": 'park'})
        _gdf_water = ox.geometries_from_place(places, {"water": True})

    elif city is 'dtchicago':
        places = ['Chicago, Illinois, USA']
        _gdf = ox.geometries_from_place(places, {"building": True})
        _gdf_parks = ox.geometries_from_place(places, {"leisure": 'park'})
        _gdf_water = ox.geometries_from_place(places, {"water": True})

    else:
        _gdf = None
        _gdf_parks = None
        _gdf_water = None

    return _gdf, _gdf_parks, _gdf_water


def plotRouteExamples(city, fmt='png', fs=figsize, dpi=300, save=True, nroutes=10):
    G = getGraphWithSetting(city)
    filepath = 'images/' + city + '_ex.' + fmt

    orig = random.sample(list(G), nroutes)
    dest = random.sample(list(G), nroutes)

    route = []
    for i in range(len(orig)):
        try:
            r = ox.shortest_path(G, orig[i], dest[i], weight="travel_time")
            route.append(r)
        except Exception as ex:
            print(ex)

    fig, ax = ox.plot_graph(G, node_size=node_s, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=edge_lw,
                            bgcolor="#ffffff", show=False, dpi=dpi, figsize=fs)

    fig, ax = ox.plot_graph_routes(
        G, ax=ax, routes=route, route_colors=ox.plot.get_colors(len(route)), route_linewidth=route_lw, node_size=0,
        close=False, show=False, dpi=dpi, figsize=fs, save=save, filepath=filepath
    )
    plt.show()


def plotRouteDistr(city, fmt='png', fs=figsize, dpi=300, save=True):
    G = getGraphWithSetting(city)
    filepath = 'images/' + city + '_distr.' + fmt
    nroutes = 30

    orig = random.sample(list(G), nroutes)
    dest = random.sample(list(G), nroutes)

    routes = []

    for i in range(100):
        try:
            routes.append(iRoute(G, city, orig[0], dest[0], 0.05))
        except Exception as ex:
            print(ex)

    i = 0
    while not CheckCompletion(routes) and i < 1e6:
        i += 1
        iRouteIterator(routes, G)

    nodes, times = GatherRoutes(routes)

    fig, ax = ox.plot_graph(G, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                            bgcolor="#ffffff", show=False, dpi=dpi, figsize=fs)

    ox.plot_graph_routes(
        G, ax=ax, routes=nodes, route_colors='r', route_linewidth=3, node_size=0, route_alpha=0.2,
        close=False, show=False, dpi=dpi, figsize=fs, save=save, filepath=filepath);
    plt.show()

def plotCityMapForSlides(city, fmt='png'):
    G = getGraphWithSetting(city)
    filepath = 'images/' + city + '_map.' + fmt

    if city == 'janeiro' or city == 'chicago':
        node_size = 0
        edge_lw_local = edge_lw
    else:
        node_size = 10
        edge_lw_local = 1

    ox.plot_graph(G, node_size=node_size, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=edge_lw_local,
                  bgcolor="#ffffff", show=False, dpi=dpi * 1, figsize=(10, 10), save=True, filepath=filepath);
    plt.show()

def plotPlaceMapForSlides(place, name=None, ns=10, lw=0.5, fmt='png'):
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    G = ox.utils_graph.get_largest_component(G, strongly=True)

    if name is None:
        filepath = 'images/' + place[0:3] + '_map.' + fmt
    else:
        filepath = 'images/' + name + '_map.' + fmt

    node_size = ns
    edge_lw_local = lw

    ox.plot_graph(G, node_size=node_size, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=edge_lw_local,
                  bgcolor="#ffffff", show=False, dpi=dpi * 1, figsize=(10, 10), save=True, filepath=filepath);
    plt.show()

def plotCityMapWithFeatures(city, fmt='png'):
    G = getGraphWithSetting(city)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    filepath = 'images/' + city + '_feat.' + fmt
    ax = None
    if city == 'janeiro' or city == 'champaign':
        _gdf, _gdf_parks, _gdf_water = getGdfsFeature(city)
        fig, ax = ox.plot_footprints(_gdf, alpha=0.5, show=False, close=False, bgcolor="#ffffff")
        fig, ax = ox.plot_footprints(_gdf_parks, ax=ax, alpha=0.5, color="green", show=False, bgcolor="#ffffff")
        fig, ax = ox.plot_footprints(_gdf_water, ax=ax, alpha=0.5, color="blue", show=False, bgcolor="#ffffff")

    if city == 'janeiro' or city == 'chicago':
        node_size = 0
        edge_lw_local = edge_lw
    else:
        node_size = 40
        edge_lw_local = 2

    ox.plot_graph(G, ax=ax, node_size=node_size, node_color="#a3a3a3", edge_color="#a3a3a3",
                  edge_linewidth=edge_lw_local,
                  bgcolor="#ffffff", show=False, dpi=dpi * 2, figsize=(20, 20), save=True, filepath=filepath);
    plt.show()

    plt.show()


def reverse_bearing(x):
    return x + 180 if x < 180 else x - 180


def count_and_merge(n, bearings):
    # make twice as many bins as desired, then merge them in pairs
    # prevents bin-edge effects around common values like 0° and 90°
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    count, _ = np.histogram(bearings, bins=bins)

    # move the last bin to the front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    return count[::2] + count[1::2]


def polar_plot(ax, bearings, n=36, title=''):
    bins = np.arange(n + 1) * 360 / n
    count = count_and_merge(n, bearings)
    _, division = np.histogram(bearings, bins=bins)
    frequency = count / count.sum()
    division = division[0:-1]
    width = 2 * np.pi / n

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')

    x = division * np.pi / 180
    bars = ax.bar(x, height=frequency, width=width, align='center', bottom=0, zorder=2,
                  color='#003366', edgecolor='k', linewidth=0.5, alpha=0.7)

    ax.set_ylim(top=frequency.max())

    title_font = {'family': 'Century Gothic', 'size': 16, 'weight': 'bold'}
    xtick_font = {'family': 'Century Gothic', 'size': 10, 'weight': 'bold', 'alpha': 1.0, 'zorder': 3}
    ytick_font = {'family': 'Century Gothic', 'size': 9, 'weight': 'bold', 'alpha': 0.2, 'zorder': 3}

    ax.set_title(title.upper(), y=1.05, fontdict=title_font)

    ax.set_yticks(np.linspace(0, max(ax.get_ylim()), 5))
    yticklabels = ['{:.2f}'.format(y) for y in ax.get_yticks()]
    yticklabels[0] = ''
    ax.set_yticklabels(labels=yticklabels, fontdict=ytick_font)

    xticklabels = ['N', '', 'E', '', 'S', '', 'W', '']
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis='x', which='major', pad=-2)


def plotStreetBearings():
    places = {  # 'Champaign': 'Champaign, IL, USA',
        # 'Urbana': 'Urbana, IL, USA',
        'Champaign-Urbana': 'champaign',
        'Downtown Chicago': 'dtchicago',
        'Chicago': 'Chicago, IL, USA',
        'Rio de Janeiro': 'Rio de Janeiro, Rio de Janeiro, Brazil',
        # 'Sao Paulo': 'Sao Paulo, Sao Paulo, Brazil',
        # 'Athens': 'Athens, Greece',
        # 'New Delhi': 'New Delhi, India',
        # 'Yerevan': 'Yerevan, Armenia',
        # 'San Francisco': 'San Francisco, CA, USA',
        # 'Belgrade': 'Belgrade, Serbia',
        # 'Modena': 'Modena, Italy',
    }

    bearings = {}
    weight_by_length = False
    for place in sorted(places.keys()):
        print(datetime.datetime.now(), place)

        # get the graph
        query = places[place]
        print('getting graph from', place)
        if place == 'Athens':
            G = ox.graph_from_address('Athens, Greece', dist=7000, network_type='drive')
        elif query == 'dtchicago' or query == 'champaign':
            print('GETTING CUSTOM QUERY from utils.py')
            G = getGraphWithSetting(query)
        else:
            G = ox.graph_from_place(query, network_type='drive')

        # calculate edge bearings
        print('getting undirected graph from', place)
        Gu = ox.add_edge_bearings(ox.get_undirected(G))

        if weight_by_length:
            # weight bearings by length (meters)
            city_bearings = []
            for u, v, k, d in Gu.edges(keys=True, data=True):
                city_bearings.extend([d['bearing']] * int(d['length']))
            b = pd.Series(city_bearings)
            bearings[place] = pd.concat([b, b.map(reverse_bearing)]).reset_index(drop='True')
        else:
            print('getting bearings from', place)
            # don't weight bearings, just take one value per street segment
            b = pd.Series([d['bearing'] for u, v, k, d in Gu.edges(keys=True, data=True)])
            bearings[place] = pd.concat([b, b.map(reverse_bearing)]).reset_index(drop='True')

    n = len(places)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    figsize = (ncols * 5, nrows * 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={'projection': 'polar'})

    # plot each city's polar histogram
    for ax, place in zip(axes.flat, sorted(places.keys())):
        polar_plot(ax, bearings[place].dropna(), title=place)

    # add super title and save full image
    suptitle_font = {'family': 'Century Gothic', 'fontsize': 40, 'fontweight': 'normal', 'y': 0.98}
    fig.suptitle('City Street Network Orientation', **suptitle_font)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig('images/street-orientations.png', dpi=120, bbox_inches='tight')
    plt.show()


def getNominalRisk(route, risk, G, city):
    orig = route.orig
    dest = route.dest
    nom_route = iRoute(G, city, orig, dest)
    compRoute(nom_route, G)
    try:
        optnodes, opttimes, E, idxs, risk_nom = computeCompositeRisk([nom_route], G, city)
        delta = risk - E[0]
        return delta
    except Exception as ex:
        print(ex)
        print(route, nom_route, route.nodes)


def plotRoutesAndOptNode(G, city, nroutes=100, Edetours=1):
    o = random.choices(list(G), k=1)[0]  # orig/dest pairs
    d = random.choices(list(G), k=1)[0]
    # print('routeRiskSingle start:', city)
    routes = []
    start = time.time()
    invalid = False

    for i in range(nroutes):
        try:
            if i == 0:
                Edetours_l = 0.0
            else:
                Edetours_l = Edetours

            routes.append(routeTotalDetourProbality(G, city, o, d, Edetours_l, 0.0))
        except Exception as ex:
            print(ex)
            print("Skipping this dest/orig pair")
            invalid = True
            routes = []
            break

    nroutes = len(routes)
    risk = 1e10
    if not invalid:
        for i in range(nroutes):
            compRoute(routes[i], G, np.random.randint(1, 1e5))
            # print(routes[i].deviated, routes[i].p)

        optnodes, opttimes, E, idxs, risk = computeCompositeRisk(routes, G, city)
        # print(optnodes)
        # print(routes[0].orig, routes[0].dest)
        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now(), "Finished batch of risk computation in", elapsed)
        delta = risk - getSingleRouteCost(o, d, G, city)
        print("delta", delta, "mincost", getSingleRouteCost(o, d, G, city))

    nodes, times = GatherRoutes(routes)

    plotNodeEnergies(nodes[0], times[0], G, city, E)

    fig, ax = ox.plot_graph(G, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                            bgcolor="#ffffff", show=False, dpi=600, figsize=(20, 20))

    fig, ax = ox.plot_graph_routes(
        G, ax=ax, routes=nodes, route_colors='r', route_linewidth=3, node_size=0,
        route_alpha=0.1,
        close=False, show=False, dpi=600, figsize=(20, 20));
    ax = scatterGraph(optnodes, G, ax)
    plt.show()

    return nodes, times


def plotRoutesAndOptNodeAndEnergyDist(city, nroutes=100, Edetours=1, fmt='png', fs=figsize, dpi=300):
    G = getGraphWithSetting(city)
    filepath = 'images/' + city + '_energies.' + fmt

    node_list = list(G.nodes)

    xs = [G.nodes[i]['x'] for i in node_list]
    ys = [G.nodes[i]['y'] for i in node_list]

    xmax = np.max(xs)
    xmin = np.min(xs)

    print(xmax, xmin, xmax - xmin)

    ymax = np.max(ys)
    ymin = np.min(ys)

    xloc1 = xmax - 0.04
    xloc2 = xmax - 0.03

    yloc1 = ymax
    yloc2 = ymin

    o = ox.get_nearest_nodes(G, [xloc2], [yloc2])[0]
    d = ox.get_nearest_nodes(G, [xloc1], [yloc1])[0]

    routes = []
    start = time.time()
    invalid = False

    depot_local = getDepotLocation(city)
    depot_node = ox.get_nearest_nodes(G, [depot_local[1]], [depot_local[0]])
    for i in range(nroutes):
        try:
            if i == 0:
                Edetours_l = 0.0
            else:
                Edetours_l = Edetours

            routes.append(routeTotalDetourProbality(G, city, o, d, Edetours_l, 0.0))
        except Exception as ex:
            print(ex)
            print("Skipping this dest/orig pair")
            invalid = True
            routes = []
            break

    nroutes = len(routes)
    risk = 1e10
    if not invalid:
        for i in range(nroutes):
            compRoute(routes[i], G, np.random.randint(1, 1e5))
            # print(routes[i].deviated, routes[i].p)

        optnodes, opttimes, E, idxs, risk = computeCompositeRisk(routes, G, city)
        # print(optnodes)
        # print(routes[0].orig, routes[0].dest)
        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now(), "Finished batch of risk computation in", elapsed)
        delta = risk - getSingleRouteCost(o, d, G, city)
        print("delta", delta, "mincost", getSingleRouteCost(o, d, G, city))

    # nodes, times = GatherRoutes(routes)

    # plotNodeEnergies(nodes[0], times[0], G, city, E)
    fig = plt.figure(figsize=fs, dpi=dpi)
    ax121 = fig.add_subplot(122)
    ax122 = fig.add_subplot(121)
    ox.plot_graph(G, ax=ax121, node_size=node_lw, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=edge_lw,
                  bgcolor="#ffffff", show=False)

    ox.plot_graph_route(
        G, ax=ax121, route=routes[0].nodes, route_colors='r', route_linewidth=route_lw, node_size=0,
        route_alpha=1.0, close=False, show=False);

    scatterGraph(depot_node, G, ax121, depot_color)
    scatterGraph(optnodes, G, ax121, 'blue')

    g = sns.histplot(E, bins=40, ax=ax122, kde=True, log_scale=False)
    ax122.set_title('Histogram of $\mathcal{E}$')
    ax122.set_xlabel('$E^\star$')
    ax122.set_ylabel('Count')
    plt.xlim(0, 1e10)

    plt.savefig(filepath)
    plt.show()

    # TODO: see what this looks like for deltas

    # return nodes, times


def plotNodeEnergies(nodes, times, G, city):
    depot_local = getDepotLocation(city)

    tr = np.cumsum(times)
    E = []
    D = []
    for n, t in zip(nodes, tr):
        d = gd.distance((G.nodes[n]['y'], G.nodes[n]['x']), depot_local).m
        e = costFun(d, t)
        E.append(e)
        D.append(d)

    fig = plt.figure(figsize=(25, 15), dpi=100)
    plt.subplot(2, 2, 1)
    plt.plot(E)
    plt.title('Nom Route Energy')
    plt.subplot(2, 2, 2)
    plt.plot(D)
    plt.title('Nom Route Distance')
    plt.subplot(2, 2, 3)
    plt.plot(tr)
    plt.title('Nom Route Times')
    plt.subplot(2, 2, 4)
    plt.plot(times)
    plt.title('non - cumsum')
    plt.show()


def getDepotLocation(city):
    if city == 'champaign':
        depot_local = depot_champaign

    elif city == 'chicago':
        depot_local = depot_chicago

    elif city == 'janeiro':
        depot_local = depot_rio

    elif city == 'dtchicago':
        depot_local = depot_dtchicago

    else:
        print('!!!!!!!!! No preset city available, returning NONE !!!!!!!!!')
        return None

    return depot_local


def plotCityAndDepot(G=None, city='champaign'):
    if G is None:
        G = getGraphWithSetting(city)

    depot_local = getDepotLocation(city)
    if depot_local is None:
        depot_local = CreateDepotLocation(G)
        print(depot_local)

    depot_node = ox.get_nearest_nodes(G, [depot_local[1]], [depot_local[0]])

    fig, ax = ox.plot_graph(G, node_size=1, node_color="#a3a3a3", edge_color="#a3a3a3", edge_linewidth=0.5,
                            bgcolor="#ffffff", show=False, dpi=600, figsize=(20, 20));
    ax = scatterGraph(depot_node, G, ax, color='blue')
    plt.show()


def CreateDepotLocation(G):
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    center_lat = np.mean(gdf_nodes.y.values)
    center_lon = np.mean(gdf_nodes.x.values)
    return (center_lat, center_lon)


def UnitToLatLon(x, y, coord_bounds):
    lon = mapRange(x, coord_bounds['lr'], coord_bounds['ur'], coord_bounds['lbx'], coord_bounds['ubx'])
    lat = mapRange(y, coord_bounds['lr'], coord_bounds['ur'], coord_bounds['lby'], coord_bounds['uby'])
    return lon, lat


def LatLonToUnit(lat, lon, coord_bounds):
    y = mapRange(lat, coord_bounds['lby'], coord_bounds['uby'], coord_bounds['lr'], coord_bounds['ur'])
    x = mapRange(lon, coord_bounds['lbx'], coord_bounds['ubx'], coord_bounds['lr'], coord_bounds['ur'])
    return x, y


def plotDataCorrelation(city, lim=1.0, fmt='png'):
    G = getGraphWithSetting(city)

    risks, deltas, routes = getDatasetFromCity(city)

    data_size = len(risks)

    rrv = []
    rrd = []
    rp = []

    d_threshold = 1000
    deltas_threshold = np.percentile(deltas, 99)
    depot_local = getDepotLocation(city)

    for i in range(data_size):
        r = routes[i].optNode
        dist = gd.distance((G.nodes[r]['y'], G.nodes[r]['x']), depot_local).m
        if len(routes[i].nodes) > 0 and 0 < deltas[i] < deltas_threshold and dist > d_threshold:
            rp.append(r)

            rrv.append(risks[i])
            rrd.append(deltas[i])

    ur = 1.0
    lr = -0.0

    ubr = np.maximum(np.max(rrv), np.max(rrv))
    lbr = np.minimum(np.min(rrv), np.min(rrv))
    ubd = np.maximum(np.max(rrd), np.max(rrd))
    lbd = np.minimum(np.min(rrd), np.min(rrd))
    rrv = [mapRange(x, lbr, ubr, lr, ur) for x in rrv]
    ddv = [mapRange(x, lbd, ubd, lr, ur) for x in rrd]

    g = sns.jointplot(x=random.choices(ddv, k=100000), y=random.choices(rrv, k=100000), s=10, alpha=.5, linewidth=0,
                      marginal_kws=dict(bins=100, kde=True), height=figsize[0]);
    plt.margins(x=1, y=1)
    g.ax_joint.set_xlim([0, lim]);
    g.ax_joint.set_ylim([0, lim]);
    g.ax_joint.set_xlabel("Normalized $\delta_E$");
    g.ax_joint.set_ylabel("Normalized $\mathrm{CVaR}_{1-\gamma}(\mathcal{E})$");
    g.ax_joint.set(xticks=[0, 0.1, 0.2, lim], yticks=[0, 0.1, 0.2, lim]);
    plt.suptitle('Data Spread: ' + getCityName(city), fontsize=16, y=0.98);
    plt.gcf().subplots_adjust(bottom=0.1, left=0.1);
    filepath = 'images/' + city + '_dataviz.' + fmt
    plt.savefig(filepath, dpi=dpi)
    plt.show()


def getDatasetFromCity(city):
    directory = 'dataset_' + city + '_max5risk/'
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

    return risks, deltas, routes


def unPackDataset(city, G, risks, deltas, routes):
    data_size = len(risks)

    oxp = []
    oyp = []
    dxp = []
    dyp = []
    rrv = []
    rrd = []

    rp = []
    rxp = []
    ryp = []

    d_threshold = 100
    deltas_threshold = np.percentile(deltas, 99)

    depot_local = getDepotLocation(city)
    depot_node = ox.get_nearest_nodes(G, [depot_local[1]], [depot_local[0]])

    for i in range(data_size):
        o = routes[i].orig
        d = routes[i].dest
        r = routes[i].optNode
        dist = gd.distance((G.nodes[r]['y'], G.nodes[r]['x']), depot_local).m
        if len(routes[i].nodes) > 0 and 0 < deltas[i] < deltas_threshold and dist > d_threshold:
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
    ubd = np.maximum(np.max(rrd), np.max(rrd))
    lbd = np.minimum(np.min(rrd), np.min(rrd))
    oxp = [mapRange(x, lbx, ubx, lr, ur) for x in oxp]
    oyp = [mapRange(x, lby, uby, lr, ur) for x in oyp]
    dxp = [mapRange(x, lbx, ubx, lr, ur) for x in dxp]
    dyp = [mapRange(x, lby, uby, lr, ur) for x in dyp]
    rrv = [mapRange(x, lbr, ubr, lr, ur) for x in rrv]
    rxp = [mapRange(x, lbx, ubx, lr, ur) for x in rxp]
    ryp = [mapRange(x, lby, uby, lr, ur) for x in ryp]
    ddv = [mapRange(x, lbd, ubd, lr, ur) for x in rrd]
    coord_bounds = {'ubx': ubx, 'lbx': lbx, 'uby': uby, 'lby': lby, 'ur': ur, 'lr': lr, 'lbd': lbd, 'ubd': ubd}

    return oxp, oyp, dxp, dyp, rrv, rxp, ryp, ddv, coord_bounds, depot_local, depot_node


def cityAndDataStats():
    city_list = ['champaign', 'dtchicago', 'chicago', 'janeiro']

    with open('city_stats.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["city", "dataset_name", "n_nodes", "n_edges", "data_size"])
        for city in city_list:
            if city is city_list[0]:
                city_name = 'Champaign-Urbana'
            elif city is city_list[1]:
                city_name = 'Downtown Chicago'
            elif city is city_list[2]:
                city_name = 'Chicago'
            elif city is city_list[3]:
                city_name = 'Rio de Janeiro'
            G = getGraphWithSetting(city)
            n_nodes = len(G.nodes)
            n_edges = len(G.edges)
            risks, deltas, routes = getDatasetFromCity(city)
            data_size = len(risks)
            fields = [city_name, city, n_nodes, n_edges, data_size]
            writer.writerow(fields)

def getCityName(city):
    if city is 'champaign':
        city_name = 'Champaign-Urbana'
    elif city is 'dtchicago':
        city_name = 'Downtown Chicago'
    elif city is 'chicago':
        city_name = 'Chicago'
    elif city is 'janeiro':
        city_name = 'Rio de Janeiro'
    else:
        city_name = None

    return city_name

def findOptSamp(reals, predictions, v1, v2, pt=60):
    r = np.arange(v1,v2,0.0005)
    vals = []
    for e in r:
        diffs = [np.abs(np.abs(reals[i] - predictions[i]) - e) for i in range(len(reals))]
        mx = np.argmin(diffs)
        acc = 0
        tot = 0
        diffs = []
        rths = np.percentile(reals, pt)
        for i in range(len(reals)):
            if reals[i] < rths:
                tot += 1
                if (reals[mx] >= reals[i] and predictions[mx] >= predictions[i]) or (
                        reals[mx] < reals[i] and predictions[mx] < predictions[i]):
                    acc += 1
                else:
                    d1 = np.abs(predictions[i] - predictions[mx]) / predictions[i]
                    d2 = np.abs(predictions[i] - predictions[mx]) / predictions[mx]
                    diffs.append(np.minimum(d1, d2))

        pct_err = acc / tot
        # print(pct_err)
        # print(np.mean(diffs))
        vals.append(pct_err)

    idx = np.argmax(vals)
    return r[idx], vals[idx]

def plotActivation(act='SELU'):
    x = torch.arange(-6.0, 6.0, 0.1, requires_grad=True)

    if act == 'SELU':
        y = F.selu(x)
    elif act == 'ReLU':
        y = x.relu()
    elif act == 'Sigmoid':
        y = x.sigmoid()
    elif act == 'GELU':
        y = F.gelu(x)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x.detach().numpy(), y.detach().numpy(), label=act, linewidth=1.5)
    plt.style.use("seaborn")
    filepath = 'images/' + act + '.png'
    plt.savefig(filepath)
    plt.show()