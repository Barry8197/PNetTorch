import re
import networkx as nx
import numpy as np
import pandas as pd
from os.path import join
from gmt_reader import GMT

reactome_base_dir = './paper_data/pathways/Reactome/'
relations_file_name = 'ReactomePathwaysRelation.txt'
pathway_names = 'ReactomePathways.txt'
pathway_genes = 'ReactomePathways.gmt'


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers


class Reactome():

    def __init__(self):
        self.pathway_names = self.load_names()
        self.hierarchy = self.load_hierarchy()
        self.pathway_genes = self.load_genes()

    def load_names(self):
        filename = join(reactome_base_dir, pathway_names)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'species']
        return df

    def load_genes(self):
        filename = join(reactome_base_dir, pathway_genes)
        gmt = GMT()
        df = gmt.load_data(filename, pathway_col=1, genes_col=3)
        return df

    def load_hierarchy(self):
        filename = join(reactome_base_dir, relations_file_name)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['child', 'parent']
        return df


class ReactomeNetwork():

    def __init__(self , genes_of_interest = None , n_levels = 5):
        self.reactome = Reactome()  # low level access to reactome pathways and genes
        self.netx = self.get_reactome_networkx()
        self.genes_of_interest = genes_of_interest
        self.n_levels = n_levels

    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):

        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')

        return G

    def get_completed_network(self):
        G = complete_network(self.netx, self.n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G
    
    def get_pathway_masks(self, pathways_of_interest=list()) :
        net = self.get_completed_network()
        
        adj_matrix = nx.adjacency_matrix(net)
        # Convert to a pandas DataFrame
        df_adj_matrix = pd.DataFrame(adj_matrix.toarray(), index=net.nodes(), columns=net.nodes())

        pathway_masks = []
        pathways_of_interest = list(pathways_of_interest)
        for i in range(1,self.n_levels) : 
            if pathways_of_interest :
                root_level_genes  = list(set(get_nodes_at_level(net , i))   & set(pathways_of_interest))
                upper_level_genes = list(set(get_nodes_at_level(net , i+1)) & set(pathways_of_interest))
                mask = df_adj_matrix.loc[root_level_genes,upper_level_genes]
                mask.index   = [re.sub('_copy.*', '', p) for p in mask.index]
                mask.columns = [re.sub('_copy.*', '', p) for p in mask.columns]
                pathway_masks.append(mask)
            else : 
                mask = df_adj_matrix.loc[get_nodes_at_level(net , i) , get_nodes_at_level(net , i+1)]
                mask.index   = [re.sub('_copy.*', '', p) for p in mask.index]
                mask.columns = [re.sub('_copy.*', '', p) for p in mask.columns]
                pathway_masks.append(mask)
             
        
        return pathway_masks
    
    def get_gene_mask(self) : 
        df  = self.reactome.pathway_genes
        net = self.get_completed_network()
        
        df['value'] = 1
        gene_mask = pd.pivot_table(df, values='value', index='gene', columns='group', aggfunc="sum")
        gene_mask = gene_mask.fillna(0)

        cols_df = pd.DataFrame(index=self.genes_of_interest)
        gene_mask = cols_df.merge(gene_mask, right_index=True, left_index=True, how='left')
        gene_mask = gene_mask.fillna(0)
        
        terminal_nodes = [re.sub('_copy.*', '', n) for n, d in net.out_degree() if d == 0]  

        missing_pathways    = list(set(terminal_nodes) - set(gene_mask.columns))
        missing_pathways_df = pd.DataFrame(np.zeros((len(gene_mask.index) , len(missing_pathways))) , index=gene_mask.index , columns=missing_pathways)

        gene_mask = pd.concat([gene_mask , missing_pathways_df] , axis=1)

        self.pathways_of_interest = gene_mask.columns
        
        return gene_mask
    
    def get_masks(self , filter_pathways=False) : 
        gene_mask = self.get_gene_mask()
        if filter_pathways : 
            gene_mask = gene_mask.loc[:, (gene_mask != 0).any(axis=0)]
            pathway_masks = [mask.T for mask in self.get_pathway_masks(gene_mask.columns)]
        else : 
            pathway_masks = [mask.T for mask in self.get_pathway_masks()]
        gene_mask = gene_mask.loc[: , pathway_masks[-1].index]
        pathway_masks = [mask.values for mask in pathway_masks]
        
        return gene_mask.values , pathway_masks[::-1] 