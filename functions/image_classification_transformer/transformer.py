import torch.nn as nn
from .transformer_elements import PositionalEncoding
from .transformer_elements import EncoderLayer

''' by Datacamp tutorial, https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch, modified'''

# Defining the Transformer
class Transformer(nn.Module):
    """Defines an encoder-only transformer for classification of
    all independent elements in sequences.
    """
    def __init__(self, d_model, num_heads, num_layers, \
                  d_ff, max_seq_length, num_labels, dropout):
        """Instantiates the encoder-only transformer.

        Args:
            d_model (int): The dimensionality as semantic vector that word indices have.
            num_heads (int): Number of heads of the attention mechanism.
            num_layers (int): Number of encoder_layers.
            d_ff (int): Dimension of the feed forward network.
            max_seq_length (int): Max. length of the sequences to process.
            num_labels (int): Number of different possible labels for sequence elements.
            dropout (int): Dropout value for training.
        """
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) \
                                              for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(d_model, num_labels)  # Output labels for each token
        self.dropout = nn.Dropout(dropout)
    def generate_mask(self, src):
        """Generates a mask for the feed forward.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    def forward(self, src):
        """Processes a sequence from an indices list to logits to generate probabilities
        of the labels for each sequence element.

        Args:
            src (torch.tensor): The source to process.
        """
        print("Begining transformer iteration!")
        src_mask = self.generate_mask(src)
        print("Mask finished!")
        src_embedded = self.dropout(self.positional_encoding(src))
        print("Positional encoding finished!")
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            print("Finished encoding iteration!")
        print("Finished encoding!")
        output = self.fc(self.sigmoid(self.linear(enc_output)))
        print(f"Output of size {output.size()} obtained!")
        return output
    