from image_classification_transformer.transformer import Transformer
import torch
import torch.nn as nn

class ConvPlusTransformer(nn.Module):
    def __init__(self, layer_filter = 1, in_channels=1):
        super(ConvPlusTransformer, self).__init__()
        self.conv = nn.Conv2d(in_channels, layer_filter, kernel_size = 16, stride = 4, padding=16)
        self.sigmoid = nn.Sigmoid()
        self.flatten_size = self._get_flatten_size(500)
        self.transformer = Transformer(d_model = 130, num_heads = 26, num_layers = 1,
                                       d_ff = 32, max_seq_length = 130, num_labels = 2, dropout = 0.2)

    
    def _get_flatten_size(self, input_size):
        # Dummy tensor to pass through the layers and get the output size
        x = torch.randn(1, 1, input_size, input_size)
        x = self.conv(x)
        return x.numel()  # Get the number of elements in the final tensor
    
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        #x = x.view(x.size(0), -1)
        
        x = x.squeeze(1)
        print(x.size())
        #x = torch.rand([130, 130]).to("mps") #mocking transformer input
        x = self.transformer(x)
        
        return x
