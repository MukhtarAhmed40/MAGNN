#Understanding Error Metrics in MAGNN
#The paper uses these regression metrics alongside classification metrics because MAGNN is a multi-task model that performs both:

#Classification (malicious/benign)

#Regression (temporal traffic prediction)

==========================================================================================================================================

# Enhanced Evaluation Metrics
def calculate_error_metrics(y_true, y_pred):
    """
    Calculate regression metrics for temporal prediction task
    Args:
        y_true: Ground truth values (temporal features)
        y_pred: Predicted values
    Returns:
        Dictionary of error metrics
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    metrics = {}
    
    # 1. Mean Absolute Error (MAE)
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    
    # 2. Mean Squared Error (MSE)
    metrics['MSE'] = np.mean((y_true - y_pred)**2)
    
    # 3. Root Mean Squared Error (RMSE)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # 4. Mean Absolute Percentage Error (MAPE)
    epsilon = 1e-8  # Small constant to avoid division by zero
    metrics['MAPE'] = 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))
    
    return metrics

# Update validate function to include error metrics
def validate(model, val_data, config):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_temp_preds = []
    all_temp_true = []
    
    with torch.no_grad():
        for data in val_data:
            data = data.to(config['device'])
            scores, temp_pred = model(data)
            
            # Classification loss
            loss_cls = F.binary_cross_entropy(scores.squeeze(), data.y.float())
            
            # Regression loss
            loss_reg = F.mse_loss(temp_pred, data.temporal)
            loss = config['lambda_cls'] * loss_cls + config['lambda_reg'] * loss_reg
            
            val_loss += loss.item()
            all_preds.append(scores.squeeze())
            all_labels.append(data.y.float())
            all_temp_preds.append(temp_pred)
            all_temp_true.append(data.temporal)
    
    # Classification metrics
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = ((preds > 0.5) == labels).float().mean()
    f1 = f1_score(labels.cpu(), (preds > 0.5).cpu())
    
    # Regression metrics
    temp_preds = torch.cat(all_temp_preds)
    temp_true = torch.cat(all_temp_true)
    error_metrics = calculate_error_metrics(temp_true, temp_preds)
    
    return {
        'loss': val_loss / len(val_data),
        'accuracy': acc.item(),
        'f1_score': f1,
        'error_metrics': error_metrics
    }

===========================================================

# Training Visualization

===========================================================
# Update your training loop to track error metrics
val_error_metrics = {
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'MAPE': []
}

for epoch in tqdm(range(config['epochs'])):
    # Train step (unchanged)
    train_loss = train(model, train_dataset, optimizer, config)
    
    # Validate with enhanced metrics
    val_results = validate(model, val_dataset, config)
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_results['loss'])
    val_accs.append(val_results['accuracy'])
    val_f1s.append(val_results['f1_score'])
    
    for metric in val_error_metrics:
        val_error_metrics[metric].append(val_results['error_metrics'][metric])
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_results['loss']:.4f}")
        print(f"Val Acc: {val_results['accuracy']:.4f} | Val F1: {val_results['f1_score']:.4f}")
        print("Error Metrics:")
        for metric, value in val_results['error_metrics'].items():
            print(f"{metric}: {value:.4f}")

# Enhanced visualization
plt.figure(figsize=(15, 10))

# Plot 1: Losses
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot 2: Classification Metrics
plt.subplot(2, 2, 2)
plt.plot(val_accs, label='Accuracy')
plt.plot(val_f1s, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Classification Metrics')

# Plot 3: Error Metrics
plt.subplot(2, 2, 3)
for metric, values in val_error_metrics.items():
    plt.plot(values, label=metric)
plt.xlabel('Epoch')
plt.ylabel('Error Value')
plt.legend()
plt.title('Regression Error Metrics')

# Plot 4: MAPE separately for clarity
plt.subplot(2, 2, 4)
plt.plot(val_error_metrics['MAPE'], label='MAPE', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Percentage Error')
plt.legend()
plt.title('Mean Absolute Percentage Error')

plt.tight_layout()
plt.show()
