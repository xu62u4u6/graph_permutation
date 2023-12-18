import pandas as pd 
import numpy as np 
import networkx as nx
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir("/home/blue/code/graph_permutation")

edges_df = pd.read_csv("data/edges.csv", index_col=0)
node_df = pd.read_csv("data/nodes.csv")
base_graph = nx.Graph()
base_graph.add_edges_from(zip(edges_df.src, edges_df.dst))
cancer_proteins = node_df[node_df.is_cancer_protein].node_name
print('1. load data done')
cancer_one_hop_set = set()
for cancer_protein in cancer_proteins:
    neighbors = set(dict(base_graph[cancer_protein]).keys())
    cancer_one_hop_set = cancer_one_hop_set.union(neighbors)
print("2. one-hop done")
# K-3 motif
cancer_one_hop_subgraph = base_graph.subgraph(cancer_one_hop_set)
cancer_one_hop_arr = np.array(cancer_one_hop_subgraph.nodes)
cancer_proteins_in_subnetwork = cancer_proteins[cancer_proteins.isin(cancer_one_hop_arr)]

motif_list = []
for n1 in cancer_proteins_in_subnetwork:
    for n2 in cancer_one_hop_subgraph[n1]:
        for n3 in cancer_one_hop_subgraph[n2]:
            if n1 in cancer_one_hop_subgraph[n3]:
                motif_list.append((n1, n2, n3))

# 去除重複的 A-B-C = A-C-B 利用set完成排序後去除
deduplicate_motif_list = set([tuple(set(motif)) for motif in motif_list])
motif_df = pd.DataFrame(deduplicate_motif_list, columns=["n1", "n2", "n3"])
print("3. dedup done")

# motif 的邊列表
motif_edges_df = pd.DataFrame(columns=["n1n2", "n2n3", "n3n1"])
motif_edges_df["n1n2"] = motif_df.n1 + "-" + motif_df.n2 
motif_edges_df["n2n3"] = motif_df.n2 + "-" + motif_df.n3
motif_edges_df["n3n1"] = motif_df.n3 + "-" + motif_df.n1 

# permutation
n_permutation = 100
node_dict = node_df.node_name.to_dict()

n_nodes = len(base_graph.nodes)
n_edges = len(base_graph.edges)

result_df = pd.DataFrame(index=motif_df.index)
for i in range(n_permutation):
    print(i, end=",")
    random_graph = nx.gnm_random_graph(n_nodes, n_edges)
    random_graph = nx.relabel_nodes(random_graph, node_dict) # 重抓1-hop subgraph
    random_one_hop_subgraph = nx.subgraph(random_graph, cancer_one_hop_set)
    random_edge_df = pd.DataFrame(random_one_hop_subgraph.edges, columns=["src", "dst"])
    random_edge_df["edge"] = random_edge_df.src + "-" + random_edge_df.dst 
    result_df[i] = motif_edges_df.isin(random_edge_df.edge.values).all(axis=1)
    if i % 100 == 0:
        result_df.to_csv(f"output/motif_permutation/result{i}.csv")
        result_df = pd.DataFrame(index=motif_df.index)
        print("")

print("5. all done")