import torch
import torch.nn as nn

from config import DEVICE

class LSTM(nn.Module):
    def __init__(self, 
                 input_size: int = 14, 
                 num_layers: int = 1, 
                 n_outputs: int = 1, 
                 latent_size: int = 256, 
                 hidden_size: int = 64,
                 p_dropout: int = 0.0, 
                 bidirectional: bool = False, 
                 batch_first: bool = True) -> None:
        super().__init__()
        
        self.encoder = LSTMEncoder(input_size, 
                                   latent_size, 
                                   num_layers, 
                                   bidirectional,
                                   p_dropout, 
                                   batch_first,)
        self.classifier = LSTMClassifier(latent_size = latent_size, 
                                         hidden_size = hidden_size,
                                         out_classes = n_outputs)
        
    def forward(self, input_data) -> torch.Tensor:
        if input_data.ndim == 2:
            input_data = input_data.unsqueeze(0)
        input_data = input_data.transpose(1, 2)            
        latent = self.encoder(input_data)
        return self.classifier(latent)


class LSTMEncoder(nn.Module):
    def __init__(self, 
                 input_size: int = 14, 
                 latent_size: int = 128,
                 num_layers: int = 1, 
                 bidirectional: int = False, 
                 p_dropout: float = 0.0, 
                 batch_first: bool = True) -> None:
        super().__init__()
        
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.D = [2 if bidirectional else 1][0]  
        
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = latent_size,
                            num_layers = num_layers,
                            dropout = p_dropout,
                            bidirectional = bidirectional, 
                            batch_first = batch_first)

        self.linear = nn.Linear(in_features = self.D * self.num_layers * self.latent_size,
                                out_features = latent_size)

    def forward(self, input_data) -> torch.Tensor:
        N = [input_data.shape[0] if self.batch_first else input_data.shape[1]][0]
        h0, c0 = (torch.zeros(self.D * self.num_layers, N, self.latent_size).to(DEVICE), 
                  torch.zeros(self.D * self.num_layers, N, self.latent_size).to(DEVICE))
        output, (hn, cn) = self.lstm(input_data, (h0, c0))
        hn = hn.reshape(N, -1)
        return self.linear(hn)


class LSTMClassifier(nn.Module):
    def __init__(self, 
                 latent_size: int = 64, 
                 hidden_size: int = 128,
                 out_classes: int = 1) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(torch.nn.ReLU(),
                                         torch.nn.Linear(in_features = latent_size, 
                                                         out_features = hidden_size),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(in_features = hidden_size, 
                                                         out_features = hidden_size // 2),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(in_features = hidden_size // 2, 
                                                         out_features = out_classes)
                                         )

    def forward(self, latent_embeddings) -> torch.Tensor:
        return self.model(latent_embeddings)