import torch
import time
import copy

def train_model_with_metadata(model, optimizer, criterion, train_loader, val_loader,
                epochs, device, patience=10):
    model.to(device)
    train_losses = []
    val_accuracies = []
    val_losses = []
    
    # Early stopping setup
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for images, shapes, margins, other_metadatas, labels in train_loader:
            images, shapes, margins, other_metadatas, labels = images.to(device), shapes.to(device), margins.to(device), other_metadatas.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, shapes, margins, other_metadatas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, shapes, margins, other_metadatas, labels in val_loader: #ESTO PETA
                images, shapes, margins, other_metadatas, labels = images.to(device), shapes.to(device), margins.to(device), other_metadatas.to(device),labels.to(device)
                outputs = model(images, shapes, margins, other_metadatas)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1).view(1,-1)  # Select class index with max score
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.4f} | "
              f"Time: {epoch_time:.2f} seconds")
        
        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Return metrics for graphing
    return train_losses, val_losses, val_accuracies, best_model