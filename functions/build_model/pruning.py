import copy
import torch
import torch.nn as nn

def prune_weights(initial_model, trained_model, pruning_rate=0.1):
    pruned_model = copy.deepcopy(initial_model)  # Clone the initial model
    for initial_layer, trained_layer in zip(pruned_model.modules(), trained_model.modules()):
        if isinstance(initial_layer, (nn.Conv2d, nn.Linear)):
            trained_weights = trained_layer.weight.data
            initial_weights = initial_layer.weight.data
            
            # Determine pruning threshold from the trained model's weights
            threshold = min(torch.quantile(trained_weights.abs(), pruning_rate), pruning_rate)
            
            # Apply pruning to the initial model's weights
            mask = trained_weights.abs() >= threshold
            initial_weights *= mask  # Zero out weights below the threshold
            
    return pruned_model