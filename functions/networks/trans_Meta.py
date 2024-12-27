from functions.image_classification_transformer.transformer import Transformer
import torch
import torch.nn as nn

class trans_Meta(nn.Module):
    def __init__(self, input_size=500, layer_filter=1, in_channels=1, d_model=256, num_heads=26, num_layers=1, 
                 d_ff=32, num_labels=2, trans_dropout=0.2, kernel_size=16, stride=4):
        super(trans_Meta, self).__init__()
        self.conv = nn.Conv2d(in_channels, layer_filter, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        self.output_size = self._calculate_output_size(input_size, kernel_size, stride)
        #self.flatten_size = self._get_flatten_size(500)
        self.transformer = Transformer(d_model=d_model, num_heads=num_heads, num_layers=num_layers,
                                       d_ff=d_ff, max_seq_length=self.output_size, num_labels=num_labels,
                                       dropout=trans_dropout)

    
    def _calculate_output_size(self, input_size, kernel_size, stride):
        # Formula teniendo en cuenta que padding siempre es kernel//2
        output_size = (input_size + 2 * (kernel_size//2) - kernel_size) // stride + 1
        return output_size*output_size
    
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), -1)
        #x = x.squeeze(1)
        #print(x.size())
        x = self.transformer(x)
        
        return x
