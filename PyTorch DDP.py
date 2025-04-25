complete PyTorch implementation for the distributed training experiments. 
This includes DDP setup, MAGNN modifications for multi-GPU training, and metric logging:
=========================================================================================

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from magnn import MAGNN  # Your MAGNN model class
from dataset import GraphDataset  # Your dataset class

def setup(rank, world_size):
    # Initialize distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class MAGNN_DDP(torch.nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.model = orig_model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, graphs, features):
        # Forward pass with multi-scale objectives
        node_logits, subgraph_logits, graph_logits = self.model(graphs, features)
        return node_logits, subgraph_logits, graph_logits

def train(rank, world_size, config):
    setup(rank, world_size)
    
    # 1. Prepare distributed data loader
    dataset = GraphDataset(config['data_path'])
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize model with DDP
    model = MAGNN(
        in_dim=config['in_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes']
    ).to(rank)
    
    ddp_model = DDP(
        MAGNN_DDP(model),
        device_ids=[rank],
        output_device=rank
    )

    # 3. Training setup
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config['lr'])
    epoch_times = []
    
    for epoch in range(config['epochs']):
        sampler.set_epoch(epoch)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for batch in dataloader:
            graphs, features, labels = batch
            graphs = graphs.to(rank)
            features = features.to(rank)
            labels = labels.to(rank)
            
            optimizer.zero_grad()
            node_logits, subgraph_logits, graph_logits = ddp_model(graphs, features)
            
            # Multi-scale loss computation
            loss = (0.4 * ddp_model.module.loss_fn(node_logits, labels) +
                   0.3 * ddp_model.module.loss_fn(subgraph_logits, labels) +
                   0.3 * ddp_model.module.loss_fn(graph_logits, labels))
            
            loss.backward()
            optimizer.step()
        
        end_time.record()
        torch.cuda.synchronize()
        epoch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        epoch_times.append(epoch_time)
        
        if rank == 0:
            print(f'Epoch {epoch} | Time: {epoch_time:.2f}s | Loss: {loss.item():.4f}')

    # 4. Save results from master process
    if rank == 0:
        torch.save({
            'config': config,
            'epoch_times': epoch_times,
            'model_state': ddp_model.module.state_dict()
        }, 'distributed_results.pt')
    
    cleanup()

if __name__ == "__main__":
    config = {
        'data_path': './data',
        'in_dim': 128,
        'hidden_dim': 256,
        'num_classes': 10,
        'batch_size': 32,
        'epochs': 100,
        'lr': 0.001
    }
    
    world_size = 2  # Number of GPUs
    mp.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )



# Results Analysis:

import numpy as np
results = torch.load('distributed_results.pt')
avg_time = np.mean(results['epoch_times'][10:])  # Skip first 10 warmup epochs
print(f"Average epoch time: {avg_time:.2f}s")
