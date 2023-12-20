import pickle
import numpy as np
import pandas as pd
import networkx as nx

def random_weights(graph: nx.Graph,
                   rs: Optional[np.random.RandomState] = None,
                   type: str = 'bimodal'):

    if rs is None:
        rs = np.random
    elif not isinstance(rs, np.random.RandomState):
        raise ValueError("Invalid random state: {}".format(rs))

    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        if type == 'bimodal':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
        elif type == 'gaussian':
            problem_graph.add_edge(n1, n2, weight=rs.randn())
        elif type == 'one':
            problem_graph.add_edge(n1, n2, weight=rs.choice([1]))

    return problem_graph


def read_cage_instances(path_to_cage_instances: str):

    list_of_cage_instances = []

    for graph in os.listdir(path_to_cage_instances):
        if graph.endswith('.dr'):
            # print(f'Solving {graph}')
            with open(f'{path_to_cage_instances}/{graph}', 'r') as f:
                first_line = f.readline()
                remaining_lines = f.readlines()
                
            G = nx.Graph()
            
            for i, line in enumerate(remaining_lines):
                degree = int(graph[6])
                for vertex in range(degree):     # degree of a vertex because of the input of the graph
                    G.add_edge(i, int(line.replace(';\n', '').replace('.\n', '').split(',')[vertex]))

        # print(f'Solving {graph}')
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        G = random_weights(G, np.random.RandomState(42), 'bimodal')
        list_of_cage_instances.append(G)

    return list_of_cage_instances

def three_reg_instances(path_to_3_reg_instnaces: str):

    list_of_three_reg_instances = []

    for graph in os.listdir(path_to_cage_instances):
        if graph.endswith('.pckl'):
            G = pickle.load(open(f'{path_to_cage_instances}/{graph}', 'rb'))
            
        list_of_three_reg_instances.append(G)

    return list_of_three_reg_instances

def rqaoa_hard_instances(path_to_results_dataframe: str, weight_type: str):

    list_of_hard_instances = []
    
    if weight_type == 'bimodal':
        df_results = pd.read_csv(f'{path_to_results_dataframe}/hard_instances_bimodal_ar0p95_nc8.csv')
    elif weight_type == 'gaussian':
        df_results = pd.read_csv(f'{path_to_results_dataframe}/hard_instances_gaussian_ar0p95_nc4.csv')

    graph_names = list(df.Graph_Name)

    for i in range(len(graph_names)):
        graph = graph_names[i]
        char1 = 'dist_'
        char2 = 'seed'
        char3 = 'd_'
        char4 = 'n'
        degree = int(graph[:graph.find(char3)])
        nodes_ = int(graph[graph.find(char3)+2:graph.find(char4)])
        generator_seed = int(graph[graph.find(char1)+5:graph.find(char2)])
        distribution = graph[graph.find(char4)+2:graph.find(char1)]

        G = nx.random_regular_graph(d=degree, n=nodes_, seed=generator_seed)
        G = random_weights(G, np.random.RandomState(42), distribution)

        list_of_hard_instances.append(G)

    return list_of_hard_instances
        
    

    
