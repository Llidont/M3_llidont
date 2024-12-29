import copy
import numpy as np
import torch
import torch.nn as nn

def prune_weights_local(initial_model, trained_model, pruning_rate=0.1):
    ''' Función de poda capa a capa'''
    pruned_model = copy.deepcopy(initial_model)
    for initial_layer, trained_layer in zip(pruned_model.modules(), trained_model.modules()):
        if isinstance(initial_layer, (nn.Conv2d, nn.Linear)):
            absolute_weights = trained_layer.weight.data.abs()
            
            # Vamos a encontrar el umbral del 10%
            threshold = np.quantile(absolute_weights.cpu().numpy(), pruning_rate)
            
            # Y nos quedamos solo con los valores iniciales correspondientes
            mask = absolute_weights >= threshold
            initial_layer.weight.data *= mask.float()  # Mandamos a 0 el resto de valores
            
    return pruned_model

def prune_weights(initial_model, trained_model, pruning_rate=0.1):
    '''Función de poda de pesos globales'''
    pruned_model = copy.deepcopy(initial_model)
    # Obtenemos todos los pesos
    all_weights = []
    for trained_layer in trained_model.modules():
        if isinstance(trained_layer, (nn.Conv2d, nn.Linear)):
            all_weights.append(trained_layer.weight.data.abs().flatten())
    all_weights = torch.cat(all_weights)
    # Buscamos el umbral del 10%
    threshold = np.quantile(all_weights.cpu().numpy(), pruning_rate)
    
    for initial_layer, trained_layer in zip(pruned_model.modules(), trained_model.modules()):
        if isinstance(initial_layer, (nn.Conv2d, nn.Linear)):
            mask = trained_layer.weight.data.abs() >= threshold
            initial_layer.weight.data *= mask.float()
        
    return pruned_model
