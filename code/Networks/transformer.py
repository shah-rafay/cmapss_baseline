import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, 
                 input_size: int = 14, 
                 n_outputs: int = 2, 
                 p_dropout: float = 0, 
                 num_heads: int = 2,
                 num_layers: int = 1,
                 d_model: int = 256, 
                 hidden_size = 128, 
                 use_pe: bool = False, 
                 batch_first: bool = True):
        super().__init__()

        self.register_buffer('use_pe', 
                             torch.tensor(use_pe)
                             )
        self.pe = PositionalEncoding(d_model = input_size, 
                                     dropout = p_dropout, 
                                     max_len = 64
                                     )
        self.linear_embeddings = torch.nn.Linear(input_size, 
                                                 d_model
                                                 )
        self.encoder = TransformerEncoder(d_model = d_model, 
                                          num_heads = num_heads, 
                                          num_layers = num_layers, 
                                          use_pe = use_pe, 
                                          p_dropout = p_dropout, 
                                          batch_first = batch_first
                                          )
        self.ruler = TransformerRuler(d_model = d_model, 
                                      hidden_size = hidden_size,
                                      n_outputs = n_outputs, 
                                      p_dropout = p_dropout
                                      )
    def forward(self, 
                input_data: torch.Tensor, 
                padding_mask: torch.Tensor = None):
        
        if input_data.ndim == 2:
            input_data = input_data.unsqueeze(0)
        input_data = input_data.transpose(1, 2)
        if self.use_pe:
            input_data = self.pe(input_data)
        
        latent_space = self.encoder(self.linear_embeddings(input_data), 
                                    src_key_padding_mask = padding_mask)
        return self.ruler(latent_space)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int, 
                 use_pe: bool,
                 pool_size: int = 16,
                 p_dropout: float = 0,
                 batch_first: bool = True) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                   nhead = num_heads,
                                                   dim_feedforward = d_model * 2,
                                                   dropout = p_dropout,
                                                   batch_first = batch_first
                                                   )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers
                                                 )
        self.projection = nn.Sequential(nn.AdaptiveAvgPool1d(pool_size),
                                        nn.Flatten(start_dim = 1),
                                        nn.Linear(in_features = d_model * pool_size, 
                                                  out_features = d_model)
                                        )
    def forward(self, input_data: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        output = self.transformer(src = input_data, 
                                  src_key_padding_mask = src_key_padding_mask)
        output = output.transpose(1, 2)
        return self.projection(output)


class TransformerRuler(torch.nn.Module):
    def __init__(self, 
                 d_model, 
                 hidden_size,
                 n_outputs, 
                 p_dropout):
        super().__init__()

        self.model = torch.nn.Sequential(nn.Linear(in_features = d_model, 
                                                   out_features = hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(in_features = hidden_size, 
                                                   out_features = hidden_size // 2),
                                         nn.LeakyReLU(),
                                         nn.Dropout(p = p_dropout),
                                         nn.Linear(in_features = hidden_size // 2, 
                                                   out_features = n_outputs)
                                         )
    def forward(self, latent_space: torch.Tensor):
        return self.model(latent_space)
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    