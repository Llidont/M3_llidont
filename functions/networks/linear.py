import torch
import torch.nn as nn

class Simple_Linear(nn.Module):
    def __init__(self, dropout_rate=0.2, num_classes = 2, input_size = 500):
        super(Simple_Linear, self).__init__()       
        self.fc1 = nn.Linear(input_size*input_size, num_classes)
        self.dropout_fc = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout_fc(x)  
        return x

class Linear(nn.Module):
    def __init__(self, num_neurons = 64, dropout_rate=0.2, num_classes = 2, input_size = 500):
        super(Linear, self).__init__()       
        self.fc1 = nn.Linear(input_size*input_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_classes)
        self.dropout_fc = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout_fc(x)  
        x = self.fc2(x)
        x = self.dropout_fc(x)  
        return x