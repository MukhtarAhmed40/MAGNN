import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

class MAGNNTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.1
        )
        self.augmentor = GraphAugmentor(
            augment_ratio=config.augment_ratio,
            restart_prob=config.restart_prob
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for data in self.train_loader:
            data = data.to(self.config.device)
            
            # Create augmented view
            aug_data = self.augmentor.edge_modification(data)
            
            # Forward pass
            scores, temp_pred, (loss_temp, loss_edge, loss_hier) = self.model(data, aug_data)
            
            # Classification loss
            loss_cls = F.binary_cross_entropy(
                scores.squeeze(), 
                data.y.float()
            )
            
            # Regression loss
            loss_reg = F.mse_loss(
                temp_pred, 
                data.temporal
            )
            
            # Total loss
            loss = (self.config.lambda_cls * loss_cls + 
                   self.config.lambda_reg * loss_reg + 
                   self.config.lambda_temp * loss_temp + 
                   self.config.lambda_edge * loss_edge + 
                   self.config.lambda_hier * loss_hier)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.config.device)
                
                # Forward pass
                scores, temp_pred = self.model(data)
                
                # Compute loss
                loss_cls = F.binary_cross_entropy(
                    scores.squeeze(), 
                    data.y.float()
                )
                loss_reg = F.mse_loss(temp_pred, data.temporal)
                loss = self.config.lambda_cls * loss_cls + self.config.lambda_reg * loss_reg
                
                val_loss += loss.item()
                all_preds.append(scores.squeeze())
                all_labels.append(data.y.float())
        
        # Compute metrics
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        
        acc = ((preds > 0.5) == labels).float().mean()
        f1 = f1_score(labels.cpu(), (preds > 0.5).cpu())
        
        return val_loss / len(self.val_loader), acc.item(), f1
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.validate()
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
