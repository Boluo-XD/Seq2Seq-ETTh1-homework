import math

import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, :x.size(1)]


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()

        self.token_embed = nn.Conv1d(num_embeddings, embedding_dim, kernel_size=3,
                                     padding=1, padding_mode="circular")
        self.pos_embed = PositionalEmbedding(embedding_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.token_embed(x.permute(0, 2, 1)).permute(0, 2, 1) + self.pos_embed(x)


class ETTTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder_embed = Embedding(7, 512)
        self.decoder_embed = Embedding(7, 512)
        self.transformer = nn.Transformer(batch_first=True)
        self.projection = nn.Linear(512, 7)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src = self.encoder_embed(src)
        tgt = self.decoder_embed(tgt)
        output = self.transformer(src, tgt)
        output = self.projection(output)
        return output.sigmoid()


class ETTLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder_embed = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )
        self.decoder_embed = nn.Linear(7, 512)
        self.encoder = nn.LSTM(512, 2048, batch_first=True)
        self.decoder = nn.LSTM(512, 2048, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 7),
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src = self.encoder_embed(src)
        tgt = self.decoder_embed(tgt)
        _, (h, c) = self.encoder(src)
        output, _ = self.decoder(tgt, (h, c))
        output = self.projection(output)
        return output.sigmoid()
