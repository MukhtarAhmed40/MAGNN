import numpy as np
import torch
from torch_geometric.utils import to_undirected

class GraphAugmentor:
    """Graph augmentation for contrastive learning"""
    def __init__(self, augment_ratio=0.2, restart_prob=0.3):
        self.augment_ratio = augment_ratio
        self.restart_prob = restart_prob
    
    def edge_modification(self, data):
        """Edge modification augmentation"""
        edge_index = data.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]
        num_modify = int(num_edges * self.augment_ratio)
        
        # Edge deletion
        del_idx = np.random.choice(num_edges, num_modify//2, replace=False)
        mask = np.ones(num_edges, dtype=bool)
        mask[del_idx] = False
        edge_index = edge_index[:, mask]
        
        # Edge addition
        num_nodes = data.x.shape[0]
        new_edges = np.random.randint(0, num_nodes, size=(2, num_modify//2))
        edge_index = np.concatenate([edge_index, new_edges], axis=1)
        
        # Ensure undirected
        edge_index = to_undirected(torch.LongTensor(edge_index))
        
        return Data(
            x=data.x.clone(),
            edge_index=edge_index.to(data.x.device),
            edge_attr=data.edge_attr[mask].clone() if data.edge_attr is not None else None,
            temporal=data.temporal.clone()
        )
    
    def random_walk_subgraph(self, data, target_nodes):
        """Random walk with restart for subgraph sampling"""
        adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
        num_nodes = adj.shape[0]
        
        subgraphs = []
        for node in target_nodes:
            # Initialize random walk
            p = np.zeros(num_nodes)
            p[node] = 1.0
            
            # Iterative random walk with restart
            for _ in range(10):  # Number of iterations
                p = (1 - self.restart_prob) * adj.dot(p) + self.restart_prob * (p == node)
            
            # Sample nodes based on stationary distribution
            sampled_nodes = np.where(p > np.percentile(p, 70))[0]
            subgraphs.append(sampled_nodes)
        
        return subgraphs
