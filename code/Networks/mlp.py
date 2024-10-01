import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, 
                 input_size: int = 14, 
                 window_length: int = 64, 
                 n_outputs: int = 1, 
                 latent_size: int = 256, 
                 hidden_size: int = 64,
                 p_dropout: int = 0.0) -> None:
        super().__init__()

        self.encoder = MLPEncoder(input_size, 
                                  window_length,
                                  latent_size, 
                                  p_dropout)
        self.classifier = MLPClassifier(latent_size = latent_size, 
                                        hidden_size = hidden_size, 
                                        out_classes = n_outputs)
        
    def forward(self, input_data) -> torch.Tensor:
        if input_data.ndim == 2:
            input_data = input_data.unsqueeze(0)        
        input_data = input_data.transpose(1, 2)        
        return self.classifier(self.encoder(input_data))


class MLPEncoder(nn.Module):
    def __init__(self, 
                 input_size: int = 14, 
                 window_length: int = 64, 
                 latent_size: int = 128,
                 p_dropout: float = 0.0) -> None:
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features = input_size, 
                                             out_features = latent_size * 4), 
                                   nn.Dropout(p = p_dropout), 
                                   nn.LeakyReLU(), 
                                   nn.Linear(in_features = latent_size * 4, 
                                             out_features = latent_size * 2), 
                                   nn.Dropout(p = p_dropout), 
                                   nn.LeakyReLU(), 
                                   nn.Linear(in_features = latent_size * 2, 
                                             out_features = latent_size), 
                                   nn.Dropout(p = p_dropout), 
                                   nn.Flatten(start_dim = 1), 
                                   nn.Linear(in_features = window_length * latent_size, 
                                             out_features = latent_size))

    def forward(self, input_data) -> torch.Tensor:
        return self.model(input_data)


class MLPClassifier(nn.Module):
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