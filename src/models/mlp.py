from __future__ import annotations

import torch
from torch import nn

class AgroMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.network(x)

def train_mlp(X_train, y_train, X_test, epochs=250, lr=0.005,
              weight_decay=1e-3, seed=42):
    torch.manual_seed(seed)
    model = AgroMLP(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(Xte).numpy().ravel()
    return model, predictions
