import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):


    def __init__(self, attn_dropout):
        
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, query, key, value, mask=None):
        
        # Compute the raw attention scores
        attn = torch.matmul(query, key.transpose(-2, -1))
        d_k = query.size(-1)
        
        # Scale the attention scores
        attn = attn / (d_k ** 0.5)
        
        # If a mask is provided, apply it to the attention scores
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        # Apply dropout to the attention scores
        attn = self.dropout(F.softmax(attn,-1))
        output = torch.matmul(attn, value)
        
        return output, attn
    
    
    
class MultiHeadAttention(nn.Module):


    def __init__(self, num_heads, d_model, dropout):
        
        super(MultiHeadAttention, self).__init__()
        # Ensure the model's dimension is divisible by the number of heads
        assert d_model % num_heads == 0

        # Layers and parameters definition
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query_ff = nn.Linear(d_model, d_model)
        self.key_ff = nn.Linear(d_model, d_model)
        self.value_ff = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention = Attention(attn_dropout=dropout)


    def forward(self, query, key, value, mask=None, return_attention=False):
        
        # Expand mask dimensions
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)
        
        # Prepare the query, key, and value tensors
        query = self.query_ff(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_ff(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_ff(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply the attention mechanism to each set of heads.
        x, self.attn = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        
        if return_attention:
            return self.attn_ff(x), self.attn
        
        return x
    
    
    
class EncoderBlock(nn.Module):


    def __init__(self, num_heads, d_model, time_in, time_out, num_joints, dropout):
        
        super().__init__()

        # Layers and parameters definition
        self.num_joints = num_joints
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(time_in, time_out, 3, padding=1)


    def forward(self, xs, mask=None):
        
        x = xs
        
        # Apply self-attention to the input
        att = self.self_attn(x, x, x, mask)
        
        # Apply ReLU activation to the sum of the input and the attention output
        x = self.relu(x + att)
        
        # Normalize the output and apply convolution
        x = self.norm1(x)
        x = self.conv(x)

        return x