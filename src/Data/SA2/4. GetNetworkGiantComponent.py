import networkx as nx
import pandas as pd


def getGCC(SA1s, links, linksSaveName, SA1SaveName):
    # Filter to only included SA1s
    links = links[links['src_SA2_MAIN16'].isin(SA1s) & links['nbr_SA2_MAIN16'].isin(SA1s)]
    
    G = nx.Graph()
    
    for edge in links.itertuples():
        if edge[3] > 1E-5:  # Filter out SA1s just touching by the corners
            G.add_edge(edge[1], edge[2])
    
    Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse = True)
    G0 = Gcc[0]
    
    
    # Save links of giant component
    GccSA1s = pd.Series(list(G0.nodes))
    
    links = links[links['src_SA2_MAIN16'].isin(GccSA1s) & links['nbr_SA2_MAIN16'].isin(GccSA1s)]
    links = links[links['LENGTH'] > 1E-5]  # Filter out SA1s just touching by the corners
    links.to_csv(linksSaveName, index = None)
    
    # Save SA1s of giant component
    SA1s = SA1s[SA1s.isin(GccSA1s)]
    SA1s.to_csv(SA1SaveName, index = None)


### SYD
linkFile = "2018-05-31-SYD-SA2-Neighbouring_Suburbs_With_Bridges.csv"
links = pd.read_csv(linkFile)
SA1s = pd.read_csv('2018-05-31-SA2Input-Normalised.csv', usecols = [0], squeeze = True)
getGCC(SA1s, links, '2018-05-31-SYD-SA2-Neighbouring_Suburbs_With_Bridges-GCC.csv', '2018-06-01-SYD-SA1s.csv')


