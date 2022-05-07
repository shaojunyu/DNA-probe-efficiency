import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import *


class ProbeDataset(Dataset):
    """DNA Probe dataset"""

    def __init__(self, args, dataframe):
        self.args = args
        self.kmer_dict = build_kmers_dict(args.kmer)
        self.dataframe = dataframe.apply(lambda x: x.to_list(), axis=1).values

    def kmer_encoding(self, seq):
        return [self.kmer_dict[kmer] for kmer in
                build_kmers(seq, self.args.kmer)]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.args.use_struct:
            seq, struct, target = self.dataframe[idx]
            struct = torch.tensor([struct_dict[c] for c in struct])
        else:
            seq, target = self.dataframe[idx]

        seq = torch.tensor(self.kmer_encoding(seq))
        target = torch.tensor(target, dtype=torch.float32)

        if self.args.onehot:
            if self.args.use_struct:
                return torch.cat((F.one_hot(seq, 4), F.one_hot(struct, 3)),
                                 1).to(torch.float32), target
            else:
                return F.one_hot(seq, 4).to(torch.float32), target
        else:
            return seq, target


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AttentionLSTM(nn.Module):
    def __init__(self, args):
        super(AttentionLSTM, self).__init__()

        if args.onehot:
            self.embed = None
            self.embed_dim = 4
            if args.use_struct:
                self.embed_dim = 7
        else:
            self.embed = nn.Embedding(num_embeddings=len(args.kmers_dict),
                                      embedding_dim=args.embed_dim)
            self.embed_dim = args.embed_dim

        self.fc_in = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=args.fc_in_dim),
            nn.ReLU(), nn.LayerNorm(args.fc_in_dim), nn.Dropout(args.dropout))
        self.pos_encoder = PositionalEncoding(args.fc_in_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.fc_in_dim,
                                                   nhead=1,
                                                   dim_feedforward=args.fc_in_dim,
                                                   dropout=args.dropout)
        encoder_norm = nn.LayerNorm(args.fc_in_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1,
                                                 norm=encoder_norm)

        self.lstm = nn.LSTM(input_size=args.fc_in_dim, hidden_size=64,
                            bidirectional=True, dropout=args.dropout,
                            num_layers=2)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5),
            nn.ReLU(), nn.LayerNorm([32, args.seq_len - 4]),
            nn.Dropout(args.dropout))

        self.fc_out = nn.Sequential(
            nn.Linear(in_features=32 * (args.seq_len - 4), out_features=128),
            nn.ReLU(), nn.LayerNorm(128), nn.Dropout(args.dropout),
            nn.Linear(in_features=128, out_features=128), nn.ReLU(),
            nn.LayerNorm(128), nn.Dropout(args.dropout),
            nn.Linear(in_features=128, out_features=1))

    def forward(self, x):
        batch_size = x.shape[0]
        if self.embed:
            x = self.embed(x)
        x = self.fc_in(x)  # N x L x C
        x = self.pos_encoder(x.permute(1, 0, 2))  # L x N x C
        x = self.transformer(x)
        x, _ = self.lstm(x)  # L x N x C
        x = self.conv1(x.permute(1, 2, 0))
        x = self.fc_out(x.view(batch_size, -1))
        return x.view(-1)
