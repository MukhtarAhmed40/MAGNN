import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class MAGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, temporal_dim, 
                 hidden_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        
        # Graph encoder
        self.encoder = GraphEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, num_heads, num_layers
        )
        
        # Contrastive modules
        self.temporal_contrast = TemporalNodeContrast(hidden_dim)
        self.edge_contrast = EdgeLevelContrast(hidden_dim)
        self.hierarchical_contrast = MultiHeadHierarchicalContrast(
            hidden_dim, num_heads
        )
        
        # Prediction heads
        self.malicious_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        self.temporal_predictor = nn.Sequential(
            nn.Linear(hidden_dim + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, temporal_dim)
        )
    
    def forward(self, data, aug_data=None):
        # Encode original graph
        z = self.encoder(data.x, data.edge_index, data.edge_attr)
        
        # Multi-task learning
        malicious_scores = self.malicious_score(z)
        temporal_pred = self.temporal_predictor(
            torch.cat([z, data.temporal], dim=1)
        )
        
        # Contrastive learning if augmented views provided
        if aug_data is not None:
            z_aug = self.encoder(aug_data.x, aug_data.edge_index, aug_data.edge_attr)
            
            # Compute contrastive losses
            loss_temporal = self.temporal_contrast(z, z_aug, data.temporal)
            loss_edge = self.edge_contrast(z, z_aug, data.edge_index)
            loss_hier = self.hierarchical_contrast(z, z_aug, data.batch)
            
            return malicious_scores, temporal_pred, (loss_temporal, loss_edge, loss_hier)
        
        return malicious_scores, temporal_pred

class GraphEncoder(nn.Module):
    """Multi-scale graph encoder with GAT layers"""
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Initial feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # Graph attention layers
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim//num_heads, heads=num_heads, 
                   edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        return x
