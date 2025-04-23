class TemporalNodeContrast(nn.Module):
    """Temporal-node contrastive learning module"""
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, z1, z2, temporal):
        # Project embeddings
        h1 = self.proj(z1)
        h2 = self.proj(z2)
        
        # Normalize
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.exp(torch.mm(h1, h2.t()) / self.temp)
        
        # Positive pairs (same node across views)
        pos_sim = torch.diag(sim_matrix)
        
        # Negative pairs
        neg_sim = sim_matrix.sum(dim=1) - pos_sim
        
        # Contrastive loss
        loss = -torch.log(pos_sim / neg_sim).mean()
        
        return loss

class EdgeLevelContrast(nn.Module):
    """Edge-level contrastive learning module"""
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.edge_proj = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, z1, z2, edge_index):
        # Get edge embeddings for both views
        src, dst = edge_index
        edge_emb1 = torch.cat([z1[src], z1[dst]], dim=1)
        edge_emb2 = torch.cat([z2[src], z2[dst]], dim=1)
        
        # Project embeddings
        h1 = self.edge_proj(edge_emb1)
        h2 = self.edge_proj(edge_emb2)
        
        # Normalize
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        
        # Compute similarity
        sim_matrix = torch.exp(torch.mm(h1, h2.t()) / self.temp)
        
        # Positive pairs
        pos_sim = torch.diag(sim_matrix)
        
        # Negative pairs
        neg_sim = sim_matrix.sum(dim=1) - pos_sim
        
        # Contrastive loss
        loss = -torch.log(pos_sim / neg_sim).mean()
        
        return loss

class MultiHeadHierarchicalContrast(nn.Module):
    """Multi-head hierarchical contrastive learning"""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, z1, z2, batch):
        # Compute subgraph embeddings using attention heads
        subgraph_embs1 = []
        subgraph_embs2 = []
        
        for head in self.attention:
            # Node-level attention
            attn1 = torch.sigmoid(head(z1))
            attn2 = torch.sigmoid(head(z2))
            
            # Subgraph pooling
            sub_emb1 = global_mean_pool(z1 * attn1, batch)
            sub_emb2 = global_mean_pool(z2 * attn2, batch)
            
            subgraph_embs1.append(sub_emb1)
            subgraph_embs2.append(sub_emb2)
        
        # Global graph embedding
        global_emb1 = self.global_proj(global_mean_pool(z1, batch))
        global_emb2 = self.global_proj(global_mean_pool(z2, batch)))
        
        # Compute contrastive loss for each head
        loss = 0
        for sub_emb1, sub_emb2 in zip(subgraph_embs1, subgraph_embs2):
            # Positive pairs
            pos_sim = F.cosine_similarity(sub_emb1, global_emb2)
            pos_sim += F.cosine_similarity(sub_emb2, global_emb1)
            
            # Negative pairs (inter-subgraph)
            neg_sim = F.cosine_similarity(sub_emb1.unsqueeze(1), 
                                        sub_emb2.unsqueeze(0), dim=2)
            neg_sim = neg_sim.mean()
            
            loss += -torch.log(pos_sim / (neg_sim + 1e-8))
        
        return loss / self.num_heads
