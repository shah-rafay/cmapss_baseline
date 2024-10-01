import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, 
                 in_channels: int = 14, 
                 out_channels: int = 64,
                 kernel_size: int = 3, 
                 n_outputs: int = 1, 
                 latent_size: int = 256, 
                 hidden_size: int = 64,
                 p_dropout: int = 0.0) -> None:
        super().__init__()

        self.encoder = CnnEncoder(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  latent_size, 
                                  p_dropout)
        self.classifier = CnnClassifier(latent_size = latent_size, 
                                        out_classes = n_outputs)

    def forward(self, input_data) -> torch.Tensor:
        if input_data.ndim == 2:
            input_data = input_data.unsqueeze(0)

        latent = self.encoder(input_data)
        return self.classifier(latent)


class CnnEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 latent_size: int, 
                 p_dropout: float) -> None:
        super().__init__()

        conv1 = torch.nn.Conv1d(in_channels = in_channels, 
                                out_channels = out_channels, 
                                kernel_size = 5)
        conv2 = torch.nn.Conv1d(in_channels = out_channels, 
                                out_channels = out_channels // 2, 
                                kernel_size=kernel_size)
        conv3 = torch.nn.Conv1d(in_channels = out_channels // 2, 
                                out_channels = out_channels // 4, 
                                kernel_size = kernel_size)
        pool_size = 100
        fc = nn.Linear(in_features = pool_size * out_channels // 4, 
                       out_features = latent_size)
        self.model = torch.nn.Sequential(conv1,
                                         torch.nn.BatchNorm1d(out_channels),
                                         torch.nn.ReLU(),
                                         conv2,
                                         torch.nn.BatchNorm1d(out_channels//2),
                                         torch.nn.ReLU(),
                                         conv3,
                                         torch.nn.BatchNorm1d(out_channels//4),
                                         torch.nn.ReLU(),            
                                         torch.nn.AdaptiveAvgPool1d(pool_size),
                                         torch.nn.Flatten(),
                                         torch.nn.Dropout(p_dropout),
                                         fc
                                         )
    def forward(self, input_data) -> torch.Tensor:
        return self.model(input_data)


class CnnClassifier(nn.Module):
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