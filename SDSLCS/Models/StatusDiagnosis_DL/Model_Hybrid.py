import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(0)

class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, cnn_channels=64, dropout=0.2):
        super(CNNTransformerModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_channels*2, out_channels=cnn_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.input_fc = nn.Linear(cnn_channels * 4 * (window_size // 4), model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.output_fc = nn.Linear(model_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply CNN
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten the output of CNN
        x = self.input_fc(x)
        x = self.dropout_layer(x)
        # Apply Transformer
        x = x.unsqueeze(1)  # Add sequence dimension
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, model_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, model_dim)
        x = x[:, -1, :]  # Use the output of the last time step
        x = self.output_fc(x)
        x = self.softmax(x)
        return x

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size + 1):
        seq = data.iloc[i:i + window_size][features].values
        label = data.iloc[i + window_size - 1][target]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)