import utils

utils.plotStreetBearings()


# im.reload(utils)
# city = 'champaign'
# G = utils.getGraphWithSetting(city)
#
# import random
# import time
# im.reload(utils)
# nroutes = 100
# Edetours = 1
#
# o = random.choices(list(G), k=1)[0]  # orig/dest pairs
# d = random.choices(list(G), k=1)[0]
# # print('routeRiskSingle start:', city)
# routes = []
# start = time.time()
# invalid = False
#
# for i in range(nroutes):
#     try:
#         if i == 0:
#             Edetours_l = 0.0
#         else:
#             Edetours_l = Edetours
#
#         routes.append(utils.routeTotalDetourProbality(G, city, o, d, Edetours_l, 0.0))
#     except Exception as ex:
#         print(ex)
#         print("Skipping this dest/orig pair")
#         invalid = True
#         routes = []
#         break
#
# nroutes = len(routes)
# risk = 1e10
# if not invalid:
#     for i in range(nroutes):
#         utils.compRoute(routes[i], G, np.random.randint(1,1e5))
#         print(routes[i].deviated, routes[i].p)
#
#     # optnodes, opttimes, E, idxs, risk = utils.computeCompositeRisk(routes, G, city)
#
#     optnodes = []
#     opttimes = []
#     idxs = []
#
#     r = routes[1]
#
#     x = []
#     y = []
#     for i in r.nodes:
#         xn, yn = utils.getxy(i, G)
#         x.append(xn)
#         y.append(yn)
#
#     idx = utils.computeOptRdvNode(x, y, r.times, city)
