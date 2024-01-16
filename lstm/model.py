from typing import Any, Optional

import torch
from torch import Tensor, nn


def build_model(name: str, out_size: int):
    if name == "LSTM":
        model = Seq2Seq_LSTM(in_channels=7, out_size=out_size)
    elif name == 'MyModel':
        model = Seq2Seq_MyModel(in_channels=7,out_size=out_size)
    else:
        raise NotImplementedError()
    return model


class Seq2Seq_LSTM(nn.Module):
    def __init__(self, in_channels: int, out_size: int) -> None:
        super().__init__()

        self.out_size = out_size

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.encoder = nn.LSTM(256, 256, num_layers=2)
        self.decoder = nn.LSTM(256, 256, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        N, L, C = inputs.shape    
        inputs = inputs.transpose(0, 1)     # [N, L, C] -> [L, N, C]   96 1024 7
        inputs = self.embedding(inputs) # 96 1024 256 
        _, (h, c) = self.encoder(inputs)
        y = torch.zeros((1, N, 256), device=inputs.device)
        outputs = []
        for t in range(self.out_size):
            y, (h, c) = self.decoder(y, (h, c))
            outputs.append(y)
            # if t < self.out_size - 1 and targets is not None:
            #     y = self.embedding(targets[:, t].unsqueeze(dim=0))
        outputs = torch.cat(outputs)
        outputs = self.fc(outputs)
        outputs = outputs.transpose(0, 1)   # [L, N, C] -> [N, L, C]
        return outputs


class Seq2Seq_MyModel(nn.Module):
    def __init__(self,in_channels:int,out_size:int) -> None:
        super().__init__()
        self.out_size = out_size
        self.num_filter = 256
        self.CNN = nn.Sequential(
            nn.Conv2d(1,self.num_filter,kernel_size=(6,1 * in_channels)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.GRU = nn.GRU(256,256,num_layers=2)

        self.LSTM = nn.LSTM(256,256,num_layers=2)

        self.auto_list = nn.ModuleList([
            nn.Linear(in_channels, 1) for _ in range(in_channels)
        ])

        self.neural_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs:Tensor, targets: Optional[Tensor] = None) -> Tensor:
        N,L,C = inputs.shape
        inputs = inputs.transpose(1, 2).unsqueeze(1)  # [N, L, C] -> [N, 1, C, L]    
        # print(inputs.shape)   # 1024 1 7 96
        # CNN Component
        cnn_features = self.CNN(inputs).view(N, -1, 1 * 256)
        # print(cnn_features.shape)   # 1024  180  256
        # exit()
        cnn_features = cnn_features.transpose(0,1)
        # LSTM Component
        # print(cnn_features.shape)
        # exit()
        _, (h, c) = self.LSTM(cnn_features)

        y = torch.zeros((1, N, 256), device=inputs.device)
        outputs = []
        # GRU Component
        for t in range(self.out_size):
            y, h = self.GRU(y, h)
            outputs.append(y)
        # _, h_gru = self.GRU(cnn_features)
        outputs = torch.cat(outputs)   # 96 1024 256
        outputs = self.neural_fc(outputs)
        outputs = outputs.transpose(0, 1)   # [L, N, C] -> [N, L, C]
        return outputs
        # Autoregressive Component
        # ar_output = torch.cat([auto(inputs[:, :, i:i+1]) for i, auto in enumerate(self.auto_list)], dim=1)

        # Prediction Component
        # neural_output = self.neural_fc(h_lstm + h_gru)
        # model_output = neural_output + ar_output

        # return model_output
    