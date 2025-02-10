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

from Models.StatusDiagnosis_DL.Model_Hybrid import set_seed, CNNTransformerModel, TimeSeriesDataset, create_sequences

features = [
    'CenterX_0','CenterY_0','CenterX_1', 'CenterY_1'
]
target = 'Mode'
data_log = pd.read_csv('DataLog_Robot.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


### Run Setting ###
label_encoder = LabelEncoder()
num_classes = len(label_encoder.classes_)
window_size = 64
smote = SMOTE(random_state=42)

### Model Setting ###
model = CNNTransformerModel(input_dim=len(features), model_dim=16, num_heads=1, num_layers=1, output_dim=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

sequences, targets = create_sequences(data_log, window_size)
data_log[target] = label_encoder.fit_transform(data_log[target])
n_samples, n_timestamps, n_features = sequences.shape
X_flat = sequences.reshape(n_samples, -1)

# Training the model
scaler = StandardScaler()
X_resampled, y_resampled = smote.fit_resample(X_flat, targets)
# Reshape the data back to the original shape
sequences, targets = create_sequences(data_log, window_size)
X_resampled = X_resampled.reshape(-1, window_size, n_features)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

X_test_shape = X_test.shape
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test_shape)
test_dataset = TimeSeriesDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Evaluating the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')