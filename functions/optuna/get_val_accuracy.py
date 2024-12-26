
import torch
from functions.networks.linear import Simple_Linear
from functions.networks.linear_Meta import Simple_Linear_Meta
from functions.networks.linear import Linear
from functions.networks.linear_Meta import Linear_Meta
from functions.networks.simple_cnn import SimpleCNN
from functions.networks.simple_cnn_Meta import SimpleCNN_Meta
#from functions.networks.transformer_Meta import Transf_Meta


def get_val_accuracy(test_loader, best_history):
    model_classes = {
        "Simple_Linear": Simple_Linear,
        "Simple_Linear_Meta": Simple_Linear_Meta,
        "Linear": Linear,
        "Linear_Meta": Linear_Meta,
        "SimpleCNN": SimpleCNN,
        "SimpleCNN_Meta": SimpleCNN_Meta,
        #"Transf_Meta": Transf_Meta,
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                    'mps' if torch.backends.mps.is_built() else 
                    'cpu')
    
    correct = 0
    total = 0
    # Cargamos el modelo indicado
    model_name = best_history["model"]
    model_class = model_classes.get(model_name)
    metadata = True if model_name[-4:]=='Meta' else False
    if model_class is None:
        raise ValueError(f"Model '{model_name}' is not defined in model_classes.")
    
    # Pasamos los hiperparámetros excepto los de entrenamiento
    filtered_hyperparameters = {
        k: v for k, v in best_history["hyperparameters"].items() 
        if k not in ["dropout_rate", "learning_rate"]
    }
    model = model_class(**filtered_hyperparameters).to(DEVICE)
    model.load_state_dict(best_history["best_model"])
    model.eval()

    # Evaluamos el test_loader
    with torch.no_grad():
        for batch in test_loader:
            # Comprobamos si hay metadatos
            if metadata:
                images, shapes, margins, other_metadatas, labels = batch
                images, shapes, margins, other_metadatas, labels = (
                    images.to(DEVICE), shapes.to(DEVICE),
                    margins.to(DEVICE), other_metadatas.to(DEVICE),
                    labels.to(DEVICE)
                )
                outputs = model(images, shapes, margins, other_metadatas)
            else:
                images, labels = batch
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
            # Predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculamos la accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy of the best model ({model_name}): {accuracy:.2f}%")
    return accuracy
