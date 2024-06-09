import networkx as nx
import random
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import json

def connSW(n, beta=None):
    g = nx.connected_watts_strogatz_graph(n, 10, 0.1)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        # weight = round(weight / 100, 2)
        if beta:
            weight = beta
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)
    return g, config

def weibo(static_data, dynamic_data, seed_set, vertex_dict, vertex2userdict):
    g = nx.Graph()
    # idx = 1
    # vertex_dict = {}
    # vertex2userdict = {}
    # for user_id in static_data:
    #     vertex_dict.setdefault(user_id, idx)
    #     vertex2userdict.setdefault(idx, user_id)
    #     idx += 1
    for user_id in dynamic_data:
        if user_id not in seed_set:
            continue
        edges = []
        # print(user_id)
        interact_list = list(set([uttr["interact_id"] for uttr in dynamic_data[user_id]]))
        reserved_list = []
        backup_list = []
        for interact_id in interact_list:
            if static_data[str(interact_id)]["user_followers"] > 100000:
                reserved_list.append(str(interact_id))
            else:
                backup_list.append(str(interact_id))
        backup_list = random.sample(backup_list, min(len(backup_list), 100))
        if len(reserved_list) > 0:
            reserved_list.extend(backup_list)
        else:
            reserved_list = backup_list
        # print(len(reserved_list))
        # 不做sample，给定seed_set
        # reserved_list = random.sample(reserved_list, min(len(reserved_list), 20))
        # if len(dynamic_data[user_id]) > 100:
        #     dynamic_data[user_id] = random.sample(dynamic_data[user_id], 100)
        for item in dynamic_data[user_id]:
            # if str(item["interact_id"]) not in reserved_list:
            #     continue
            if (user_id in seed_set and str(item["interact_id"]) in reserved_list) or str(item["interact_id"]) in seed_set:
                from_vertex = vertex_dict[user_id]
                to_vertex = vertex_dict[str(item["interact_id"])]
                edges.append((from_vertex, to_vertex))
                # edges.append((to_vertex, from_vertex))
        g.add_edges_from(edges)
    # g = nx.connected_watts_strogatz_graph(n, 10, 0.1)

    config = mc.Configuration()

    for a, b in g.edges():
        # weight = random.randrange(40,80)
        # weight = round(weight / 100, 2)
        # if beta:
        #     weight = beta
        # g[a][b]['weight'] = weight
        user_from = vertex2userdict[a]
        user_to = vertex2userdict[b]
        interact_list = []
        if user_from in dynamic_data:
            interact_list = [item for item in dynamic_data[user_from] if str(item["interact_id"]) == user_to]
        weight = len(interact_list)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)
    return g, config

def BA():
    g = nx.barabasi_albert_graph(1000, 5)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        # weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def ER():
    g = nx.erdos_renyi_graph(5000, 0.002)

    while nx.is_connected(g) == False:
        g = nx.erdos_renyi_graph(5000, 0.002)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config


connSW(1000, 0.1)
