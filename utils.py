import numpy as np
import multiprocessing as mp
import networkx as nx
import matplotlib.pyplot as plt
import math
import geopy.distance as gd
import random
from multiprocessing import Pool as ThreadPool
import itertools
import statistics
import time
import datetime

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

depot = (40.11237526379417, -88.24327192934085)

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
            x.append(G.nodes[u]["x"])
            y.append(G.nodes[u]["y"])
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
        self.x = x  # position
        self.y = y
        # self.s = s  # speeds
        # # self.v = v  # calculated velocity
        self.t = t  # time for transition
        self.optNode = computeOptRdvNode(x, y, t)
        self.tr = self.t[self.optNode]
        self.optNode = self.route[self.optNode]
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
        self.t_0 = t0
        self.p = p
        self.orig = orig
        self.dest = dest
        self.risk = 0.0

    # this function progresses to the next node
    def progress(self, G, local_state=None):

        # check if there's a possibility of rerouting
        if self.pos >= 0:
            node_list, an = OneDegSep(G, self.route[self.pos], self.route[self.pos - 1], self.route[self.pos + 1])
            if local_state is not None:
                jump = local_state.uniform(0, 1) < self.p
            else:
                jump = np.random.uniform(0, 1) < self.p

            if len(node_list) > 1 and jump:
                # print("rerouting!!",node_list, self.pos)
                self.route = reRoute(G, self.route[self.pos], node_list[np.random.randint(0, len(node_list))],
                                     self.dest)
                x, y, s, t, d = computeIntervals(G, self.route, 0.0);
                self.t = t  # time for transition
                self.pos = 0
                self.lt = len(self.t)
            else:
                self.pos += 1  # advance one node

        self.nodes.append(self.route[self.pos])
        self.times.append(self.local_time + self.t_0)
        self.edge_timer = self.dt * 0.1  # 1 if progressing to the next node is an action
        return True

    # this function advances one time step
    def step(self, G, local_state=None):
        self.edge_timer += self.dt
        self.local_time += self.dt
        stat = None
        # if self.edge_timer >= self.t[self.pos] and self.pos < self.lt-1 and self.completed == False:
        #     #print("Pos:", self.pos, "Travel Time:", self.t[self.pos], "Edge Time:", self.edge_timer, "Local Time:",
        #     #      self.local_time)
        #     stat = self.progress(G)
        # elif self.pos == self.lt-1 and self.completed == False:
        #     self.completed = True
        #     #print("At end of route, pos=", self.pos, "length of t=", self.lt, "Completed:", self.completed)

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
    random.seed(seed)
    i = 0
    while not r.completed and i < 1e10:
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


def routeTotalDetourProbality(G, orig, dest, Edetours, t_0=0.0):
    nom_route = ox.shortest_path(G, orig, dest, weight='travel_time')
    d = nDetourOpportunities(nom_route, G)
    if d > 2:
        # E[detours] = p*d -> p = Edetours/d
        p = Edetours / d
        p = clip(p, 0, 1)
    else:
        p = 0.0

    return iRoute(G, orig, dest, p, t_0)


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


def routeRiskSingle(G, o, d, Edetours, t_0, nroutes):
    routes = []
    start = time.time()
    invalid = False
    for i in range(nroutes):
        try:
            if i == 0:
                Edetours_l = 0.0
            else:
                Edetours_l = Edetours

            routes.append(routeTotalDetourProbality(G, o, d, Edetours_l, t_0))
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
            compRoute(routes[i], G)

        risk = getRiskValue(routes, G)
        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now(), "Finished batch of risk computation in", elapsed)

    if len(routes) > 0:
        routes = [routes[0]]

    return routes, risk


def fastBigData(G, nroutes=100, npairs=100, Edetours=1, pool_size=cpus):
    pool_size = np.minimum(pool_size, cpus)
    pool = ThreadPool(pool_size)
    orig = random.choices(list(G), k=npairs)  # orig/dest pairs
    dest = random.choices(list(G), k=npairs)
    t_0 = 0.0
    start = time.time()
    results = pool.starmap(routeRiskSingle,
                           zip(itertools.repeat(G), orig, dest, itertools.repeat(Edetours), itertools.repeat(t_0),
                               itertools.repeat(nroutes)))

    pool.close()
    end = time.time()
    elapsed = end - start
    print("Finished everything in", elapsed)
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
    return 1 / 2 * d ** 2 * t + alpha * t
    # return d


def computeOptRdvNode(x, y, t):
    n = len(x) - 1
    E = []
    for i in range(n):
        d = gd.distance((y[i], x[i]), depot).m
        tr = sum(t[0:i + 1])
        e = costFun(d, tr)
        E.append(e)

    return np.argmin(E)


def computeRendezvousEnergy(nodes, times, G):
    n = len(nodes)
    if n is not len(times):
        raise Exception("Input dim error computeRendezvousEnergy")
    E = []
    for node, time in zip(nodes, times):
        y = G.nodes[node]['y']
        x = G.nodes[node]['x']
        d = gd.distance((y, x), depot).m
        tr = time
        e = costFun(d, tr)
        E.append(e)

    return E


def computeRisk(optnode, tr, nodes, times):
    # find the closest time:
    ti = np.argmin()


def getxy(node, G):
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]
    return x, y


def getDepotNodeDistance(node, G):
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]


def computeCompositeRisk(routes, G):
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
        idx = computeOptRdvNode(x, y, r.t)
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
    n_max_values = 3
    maxel = sorted(range(len(E)), key=lambda k: E[k])[-n_max_values:]
    risk = np.mean(E)

    if len(maxel) == 0:
        risk = 1e10

    return optnodes, opttimes, E, idxs, risk


def getRiskValue(routes, G):
    optnodes, opttimes, E, idxs, risk = computeCompositeRisk(routes, G)
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

