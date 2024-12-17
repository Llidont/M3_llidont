import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, kernel_size=16, stride = 2, num_neurons = 32, layer_filter = 1, input_size = 500, in_channels=1, num_classes = 2, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, layer_filter, kernel_size = kernel_size, stride = stride, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        self.output_size = self._calculate_output_size(self, input_size, kernel_size, stride)
        self.fc1 = nn.Linear(self.output_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_classes)
        self.dropout_fc = nn.Dropout(dropout_rate)

    
    def _calculate_output_size(self, input_size, kernel_size, stride):
        # Formula teniendo en cuenta que padding siempre es kernel//2
        return (input_size + 2 * (kernel_size//2) - kernel_size) // stride + 1
    
    def forward(self, images):
        images = self.conv(images)
        images = self.sigmoid(images)
        images = images.view(images.size(0), -1)
        images = self.fc1(images)
        images = self.dropout_fc(images)
        images = self.fc2(images)
        return images