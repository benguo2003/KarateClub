from sklearn import cluster
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


def draw_communities(G, y_true, pos):
    fig, ax = plt.subplots(figsize=(16, 9))

    # Convert membership list to a dict where key=club, value=list of students in club
    club_dict = defaultdict(list)
    for student, club in enumerate(y_true):
        club_dict[club].append(student)

    # Normalize number of clubs for choosing a color
    norm = colors.Normalize(vmin=0, vmax=len(club_dict.keys()))

    for club, members in club_dict.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=members,
                               node_color=cm.jet(norm(club)),
                               node_size=500,
                               alpha=0.8,
                               ax=ax)

    # Draw edges (social connections) and show final plot
    plt.title("Zachary's Karate Club")
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    plt.show()

G = nx.karate_club_graph()

#pos = nx.spring_layout(G)
# True labels of the group each student (node) ended up in. Found via the original paper
#y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#draw_communities(G, y_true, pos)
edges = nx.edge_betweenness_centrality(G)
print(edges)

max = -1
edgey = ""
edgex = ""


for i in range(11):
    for ex, ey in edges:
        #print "(" + str(ex) + ", " + str(ey) + "): " + str(edges[(ex, ey)])
        if edges[(ex, ey)] > max:
            max = edges[(ex, ey)]
            edgex = ex
            edgey = ey

    print("WINNER: (" + str(edgex) + ", " + str(edgey) + "): " + str(max))
    del edges[(edgex, edgey)]
    max = -1
    for ex, ey in edges:
        print("(" + str(ex) + ", " + str(ey) + "): " + str(edges[(ex, ey)]))
    G.remove_edge(edgex, edgey)
    edges = nx.edge_betweenness_centrality(G)

    graphs = list(nx.connected_component_subgraphs(G))


for ex, ey in edges:
    #print "(" + str(ex) + ", " + str(ey) + "): " + str(edges[(ex, ey)])
    pass

pos = nx.spring_layout(G)
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

g = nx.minimum_spanning_tree(G)
for n in g:
    print(n)

draw_communities(G, y_true, pos)
print("MISSES NODE 2 and 8")