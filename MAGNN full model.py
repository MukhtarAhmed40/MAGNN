#Jupyter Notebook implementation of MAGNN that you can run directly in your notebook environment:
======================================================================================================

# MAGNN Implementation in Jupyter Notebook
# Multi-scale Adaptive Graph Neural Networks for Malicious Traffic Detection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration (matches paper parameters)
config = {
    'hidden_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'batch_size': 64,
    'lambda_cls': 1.0,
    'lambda_reg': 0.5,
    'lambda_temp': 0.3,
    'lambda_edge': 0.3,
    'lambda_hier': 0.3,
    'augment_ratio': 0.2,
    'restart_prob': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {config['device']}")

# 1. Graph Construction (Example for CTU-13 dataset)
# Note: In practice you would load your actual network traffic data here
def create_synthetic_data(num_nodes=100, num_edges=300, num_features=10, temporal_dim=5):
    """Create synthetic graph data for demonstration"""
    node_features = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, num_features)
    temporal_features = torch.randn(num_nodes, temporal_dim)
    labels = torch.randint(0, 2, (num_nodes,))  # Binary classification
    
    # Make graph undirected
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        temporal=temporal_features,
        y=labels
    )

# Create synthetic dataset
print("\nCreating synthetic dataset...")
dataset = [create_synthetic_data() for _ in range(100)]  # 100 graphs
train_dataset = dataset[:80]
val_dataset = dataset[80:]

print(f"Sample graph stats: {dataset[0]}")
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of validation graphs: {len(val_dataset)}")

# 2. MAGNN Model Implementation
class TemporalNodeContrast(nn.Module):
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, z1, z2):
        h1 = self.proj(z1)
        h2 = self.proj(z2)
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        
        sim_matrix = torch.exp(torch.mm(h1, h2.t()) / self.temp)
        pos_sim = torch.diag(sim_matrix)
        neg_sim = sim_matrix.sum(dim=1) - pos_sim
        loss = -torch.log(pos_sim / neg_sim).mean()
        return loss

class EdgeLevelContrast(nn.Module):
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.edge_proj = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, z1, z2, edge_index):
        src, dst = edge_index
        edge_emb1 = torch.cat([z1[src], z1[dst]], dim=1)
        edge_emb2 = torch.cat([z2[src], z2[dst]], dim=1)
        
        h1 = self.edge_proj(edge_emb1)
        h2 = self.edge_proj(edge_emb2)
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        
        sim_matrix = torch.exp(torch.mm(h1, h2.t()) / self.temp)
        pos_sim = torch.diag(sim_matrix)
        neg_sim = sim_matrix.sum(dim=1) - pos_sim
        loss = -torch.log(pos_sim / neg_sim).mean()
        return loss

class MultiHeadHierarchicalContrast(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, z1, z2, batch):
        subgraph_embs1 = []
        subgraph_embs2 = []
        
        for head in self.attention:
            attn1 = torch.sigmoid(head(z1))
            attn2 = torch.sigmoid(head(z2))
            sub_emb1 = global_mean_pool(z1 * attn1, batch)
            sub_emb2 = global_mean_pool(z2 * attn2, batch)
            subgraph_embs1.append(sub_emb1)
            subgraph_embs2.append(sub_emb2)
        
        global_emb1 = self.global_proj(global_mean_pool(z1, batch))
        global_emb2 = self.global_proj(global_mean_pool(z2, batch))
        
        loss = 0
        for sub_emb1, sub_emb2 in zip(subgraph_embs1, subgraph_embs2):
            pos_sim = F.cosine_similarity(sub_emb1, global_emb2)
            pos_sim += F.cosine_similarity(sub_emb2, global_emb1)
            neg_sim = F.cosine_similarity(sub_emb1.unsqueeze(1), 
                                        sub_emb2.unsqueeze(0), dim=2)
            neg_sim = neg_sim.mean()
            loss += -torch.log(pos_sim / (neg_sim + 1e-8))
        
        return loss / self.num_heads

class MAGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, temporal_dim, config):
        super().__init__()
        self.config = config
        
        # Graph encoder
        self.node_proj = nn.Linear(node_feat_dim, config['hidden_dim'])
        self.edge_proj = nn.Linear(edge_feat_dim, config['hidden_dim'])
        self.convs = nn.ModuleList([
            GATConv(config['hidden_dim'], config['hidden_dim']//config['num_heads'], 
                   heads=config['num_heads'], edge_dim=config['hidden_dim'])
            for _ in range(config['num_layers'])
        ])
        
        # Contrastive modules
        self.temporal_contrast = TemporalNodeContrast(config['hidden_dim'])
        self.edge_contrast = EdgeLevelContrast(config['hidden_dim'])
        self.hierarchical_contrast = MultiHeadHierarchicalContrast(
            config['hidden_dim'], config['num_heads']
        )
        
        # Prediction heads
        self.malicious_score = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']//2),
            nn.ReLU(),
            nn.Linear(config['hidden_dim']//2, 1),
            nn.Sigmoid()
        )
        
        self.temporal_predictor = nn.Sequential(
            nn.Linear(config['hidden_dim'] + temporal_dim, config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], temporal_dim)
        )
    
    def forward(self, data, aug_data=None):
        # Encode original graph
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)
        
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)
            x = F.relu(x)
        
        # Multi-task learning
        malicious_scores = self.malicious_score(x)
        temporal_pred = self.temporal_predictor(
            torch.cat([x, data.temporal], dim=1)
        )
        
        # Contrastive learning if augmented views provided
        if aug_data is not None:
            # Encode augmented graph
            x_aug = self.node_proj(aug_data.x)
            edge_attr_aug = self.edge_proj(aug_data.edge_attr)
            
            for conv in self.convs:
                x_aug = conv(x_aug, aug_data.edge_index, edge_attr_aug)
                x_aug = F.relu(x_aug)
            
            # Compute contrastive losses
            loss_temporal = self.temporal_contrast(x, x_aug)
            loss_edge = self.edge_contrast(x, x_aug, data.edge_index)
            
            # Create batch vector (assuming each graph is separate)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            loss_hier = self.hierarchical_contrast(x, x_aug, batch)
            
            return malicious_scores, temporal_pred, (loss_temporal, loss_edge, loss_hier)
        
        return malicious_scores, temporal_pred

# 3. Graph Augmentation
class GraphAugmentor:
    def __init__(self, augment_ratio=0.2, restart_prob=0.3):
        self.augment_ratio = augment_ratio
        self.restart_prob = restart_prob
    
    def edge_modification(self, data):
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
        
        # Ensure undirected and return
        edge_index = to_undirected(torch.LongTensor(edge_index))
        edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None
        
        return Data(
            x=data.x.clone(),
            edge_index=edge_index.to(data.x.device),
            edge_attr=edge_attr.clone() if edge_attr is not None else None,
            temporal=data.temporal.clone(),
            y=data.y.clone()
        )

# 4. Training Setup
def train(model, train_data, optimizer, config):
    model.train()
    total_loss = 0
    augmentor = GraphAugmentor(config['augment_ratio'], config['restart_prob'])
    
    for data in train_data:
        data = data.to(config['device'])
        aug_data = augmentor.edge_modification(data)
        
        # Forward pass
        scores, temp_pred, (loss_temp, loss_edge, loss_hier) = model(data, aug_data)
        
        # Classification loss
        loss_cls = F.binary_cross_entropy(scores.squeeze(), data.y.float())
        
        # Regression loss
        loss_reg = F.mse_loss(temp_pred, data.temporal)
        
        # Total loss
        loss = (config['lambda_cls'] * loss_cls + 
               config['lambda_reg'] * loss_reg + 
               config['lambda_temp'] * loss_temp + 
               config['lambda_edge'] * loss_edge + 
               config['lambda_hier'] * loss_hier)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_data)

def validate(model, val_data, config):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_data:
            data = data.to(config['device'])
            scores, temp_pred = model(data)
            
            # Compute loss
            loss_cls = F.binary_cross_entropy(scores.squeeze(), data.y.float())
            loss_reg = F.mse_loss(temp_pred, data.temporal)
            loss = config['lambda_cls'] * loss_cls + config['lambda_reg'] * loss_reg
            
            val_loss += loss.item()
            all_preds.append(scores.squeeze())
            all_labels.append(data.y.float())
    
    # Compute metrics
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    
    acc = ((preds > 0.5) == labels).float().mean()
    f1 = f1_score(labels.cpu(), (preds > 0.5).cpu())
    
    return val_loss / len(val_data), acc.item(), f1

# 5. Initialize Model and Optimizer
print("\nInitializing model...")
sample_data = dataset[0]
model = MAGNN(
    node_feat_dim=sample_data.x.size(1),
    edge_feat_dim=sample_data.edge_attr.size(1),
    temporal_dim=sample_data.temporal.size(1),
    config=config
).to(config['device'])

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config['lr'], 
    weight_decay=config['weight_decay']
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.1
)

# 6. Training Loop
print("\nStarting training...")
train_losses = []
val_losses = []
val_accs = []
val_f1s = []

best_val_loss = float('inf')
best_model = None

for epoch in tqdm(range(config['epochs'])):
    # Train
    train_loss = train(model, train_dataset, optimizer, config)
    
    # Validate
    val_loss, val_acc, val_f1 = validate(model, val_dataset, config)
    scheduler.step(val_loss)
    
    # Save metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

# Save the best model
torch.save(best_model, 'magnn_best_model.pth')

# 7. Plot Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Accuracy')
plt.plot(val_f1s, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Validation Metrics')

plt.tight_layout()
plt.show()

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final validation accuracy: {val_accs[-1]:.4f}")
print(f"Final validation F1 score: {val_f1s[-1]:.4f}")
