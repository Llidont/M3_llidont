import copy
import torch
import torch.nn as nn

def prune_weights(initial_model, trained_model, pruning_rate=0.1):
    pruned_model = copy.deepcopy(initial_model)
    for initial_layer, trained_layer in zip(pruned_model.modules(), trained_model.modules()):
        if isinstance(initial_layer, (nn.Conv2d, nn.Linear)):
            trained_weights = trained_layer.weight.data
            initial_weights = initial_layer.weight.data
            
            # Vamos a encontrar el umbral del 10%
            threshold = min(torch.quantile(trained_weights.abs(), pruning_rate), pruning_rate)
            
            # Y nos quedamos solo con esos valores iniciales
            mask = trained_weights.abs() >= threshold
            initial_weights *= mask  # Mandamos a 0 el resto de valores
            
    return pruned_model