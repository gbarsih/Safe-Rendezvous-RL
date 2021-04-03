using OpenStreetMapX
using OpenStreetMapXPlot
using Plots
using Random

pth = joinpath("UC.osm")
m =  OpenStreetMapX.get_map_data(pth,use_cache = false);


Random.seed!(0);
pointA = point_to_nodes(generate_point_in_bounds(m), m)
pointB = point_to_nodes(generate_point_in_bounds(m), m)
sr = shortest_route(m, pointA, pointB)[1]

p = OpenStreetMapXPlot.plotmap(m,width=1200,height=800);
addroute!(p,m,sr;route_color="red");
p
