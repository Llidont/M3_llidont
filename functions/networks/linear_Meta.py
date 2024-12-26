import torch
import torch.nn as nn

class Simple_Linear_Meta(nn.Module):
    def __init__(self, dropout_rate=0.2, num_classes = 2, input_size = 500):
        super(Simple_Linear_Meta, self).__init__()       
        self.fc1 = nn.Linear(input_size*input_size, 4)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.shapes_embedding = nn.Embedding(20, 3)
        self.margins_embedding = nn.Embedding(20, 3)
        self.linear_unifier = nn.Linear(14, num_classes)
    
    def forward(self, x, shapes, margins, other_metadatas):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout_fc(x)  
        # Procesamiento de metadatos
        shapes = torch.clamp(shapes, min=0, max=19)
        margins = torch.clamp(margins, min=0, max=19)
        shapes = self.shapes_embedding(shapes)
        margins = self.margins_embedding(margins)
        x = torch.cat((x, shapes, margins, other_metadatas), dim=1)
        # Output final
        x = self.linear_unifier(x)
        return x



class Linear_Meta(nn.Module):
    def __init__(self, num_neurons = 64, dropout_rate=0.2, num_classes = 2, input_size = 500):
        super(Linear_Meta, self).__init__()       
        self.fc1 = nn.Linear(input_size*input_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, 4)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.shapes_embedding = nn.Embedding(20, 3)
        self.margins_embedding = nn.Embedding(20, 3)
        self.linear_unifier = nn.Linear(14, num_classes)
    
    def forward(self, x, shapes, margins, other_metadatas):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout_fc(x)  
        x = self.fc2(x)
        x = self.dropout_fc(x)  
        # Procesamiento de metadatos
        shapes = torch.clamp(shapes, min=0, max=19)
        margins = torch.clamp(margins, min=0, max=19)
        shapes = self.shapes_embedding(shapes)
        margins = self.margins_embedding(margins)
        x = torch.cat((x, shapes, margins, other_metadatas), dim=1)
        # Output final
        x = self.linear_unifier(x)
        return x