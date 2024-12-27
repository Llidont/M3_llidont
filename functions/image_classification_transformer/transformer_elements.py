import torch
import torch.nn as nn
import math

''' by Datacamp tutorial, https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch, modified'''

class PositionalEncoding(nn.Module):
    """Defines the positional encoding of sequences."""
    def __init__(self, d_model, max_seq_length):
        """Instantiates the Positional Encoding layer.
        
        Args:
            d_model (int): The dimensionality as semantic vector that word indices have.
            max_seq_length (int): Max. length of the sequences to process.
        """
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        """Computes and add the positional encoding of a tensor
        
        Args:
            x (torch.tensor): The tensor whose positional encoding we compute and add.
        """
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    """Defines an encoder layer. The encoder is built by stacking this layers."""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """Instantiates an encoder layer.

        Args:
            d_model (int): The dimensionality as semantic vector that word indices have.
            num_heads (int): Number of heads of the attention mechanism.
            d_ff (int): Dimension of the feed forward network.
            dropout (int): Dropout value for training.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        """Processes a torch tensor through an encoder layer.

        Args:
            x (torch.tensor): The tensor to process.
            mask (torch.tensor of bools): A mask to apply to the source.
        """
        print("Entering encoder layer!")
        attn_output = self.self_attn(x, x, x, mask)
        #attn_output = self.self_attn(x, mask)
        print("Mask applied!")
        x = self.norm1(x + self.dropout(attn_output))
        print("Data noramlized!")
        ff_output = self.feed_forward(x)
        print("Finished feedforwar!")
        x = self.norm2(x + self.dropout(ff_output))
        print("Normalized again!")
        return x

class MultiHeadAttention(nn.Module):
    """Defines the multihead attention mechanism."""
    def __init__(self, d_model, num_heads):
        """Instantiates the multihead attention mechanism.

        Args:
            d_model (int): The dimensionality as semantic vector that word indices have.
            num_heads (int): Number of heads of the attention mechanism.
        """
        super().__init__()
        assert d_model % num_heads == 0 #d_model must be divisible by num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Performs the scaled dot product used in the attention mechanism.

        Args:
            Q (torch.tensor): The query matrix.
            K (torch.tensor): The key matrix.
            V (torch.tensor): The value matrix.
            mask (mask): A mask.
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    def split_heads(self, x):
        """Splits the data for each head of the attention mechanism.

        Args:
            x (torch.tensor): The tensor to split.

        Returns:
            fragments of x
        """
        batch_size, seq_length, d_model = x.size()
        print(f"Batch size detected: {batch_size}!")
        print(f"Model dimension detected: {d_model}!")
        print(f"Sequence length detected: {seq_length}!")
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    def combine_heads(self, x):
        """Combines the tensors processed by the heads of the attention mechanism.

        Args:
            x (torch.tensor): The heads outputs that will be combined.

        Returns:
            the combined tensors.
        """
        #batch_size, _, seq_length, d_k = x.size()
        print(f"Dimensions to unpack: {x.size()}")
        batch_size, _, _, seq_length, d_k = x.size()
        
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    def forward(self, Q, K, V, mask=None):
        """Performs the attention mechanism steps.

        Args:
            Q (torch.tensor): The query matrix.
            K (torch.tensor): The key matrix.
            V (torch.tensor): The value matrix.
            mask (mask): A mask.

        Returns:
            output (torch.tensor): The sequences processed by the multihead attention mechanism.
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        print("Heads splitted!")
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        print(f"Product attention performed! Combine heads will have a {attn_output.size()} input!")
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    """Defines the position-wise feed forward network."""
    def __init__(self, d_model, d_ff):
        """Instantiates the position-wise feed forward network.
        
        Args:
            d_model (int): The dimensionality as semantic vector that word indices have.
            d_ff (int): Dimension of the feed forward network.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        """Processes tensors using the position-wise feed forward network

        Args:
            x (torch.tensor): The tensor to process.

        Returns:
            processed tensor.
        """
        return self.fc2(self.relu(self.fc1(x)))
    