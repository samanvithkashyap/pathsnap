"""
GPU-Accelerated PyTorch MLP Candidate Classifier for PathSnap.

Provides an sklearn-compatible fit / predict_proba interface with:
- Internal StandardScaler (fitted strictly on training data inside fit())
- Architecture: Linear -> BatchNorm1d -> ReLU -> Dropout across configurable hidden layers
- BCEWithLogitsLoss + AdamW + CosineAnnealingLR
- Genuine epoch-wise training & validation Log Loss history tracking
- CPU-safe serialization (weights moved to CPU for portable joblib/FastAPI loading)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchMLPCandidateClassifier:
    """Sklearn-like binary classifier wrapping a PyTorch MLP with epoch-wise history."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.15,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4096,
        max_epochs: int = 25,
        patience: int = 5,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler_: Optional[StandardScaler] = None
        self.in_dim_: Optional[int] = None
        self.state_dict_cpu_: Optional[Dict[str, torch.Tensor]] = None
        self.history_: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": []}
        self.best_epoch_: int = 0
        self.best_val_loss_: Optional[float] = None

    def _set_seed(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_net(self, in_dim: int, device: torch.device) -> _TabularMLP:
        return _TabularMLP(in_dim, self.hidden_dims, self.dropout).to(device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        fixed_epochs: Optional[int] = None,
    ) -> "TorchMLPCandidateClassifier":
        self._set_seed()
        device = torch.device(self.device_str if (self.device_str == "cpu" or torch.cuda.is_available()) else "cpu")

        self.scaler_ = StandardScaler()
        X_tr = self.scaler_.fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        y_tr = np.asarray(y, dtype=np.float32)
        self.in_dim_ = int(X_tr.shape[1])

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_va = self.scaler_.transform(np.asarray(X_val, dtype=np.float32)).astype(np.float32)
            y_va = np.asarray(y_val, dtype=np.float32)

        # Keep tensors on GPU directly (1.3M x 33 float32 is only ~175 MB — fits easily in 8GB VRAM!)
        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        if has_val:
            X_va_t = torch.from_numpy(X_va).to(device)
            y_va_t = torch.from_numpy(y_va).to(device)

        n_samples = X_tr_t.shape[0]
        n_epochs = int(fixed_epochs) if fixed_epochs is not None else self.max_epochs

        net = self._build_net(self.in_dim_, device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

        self.history_ = {"epoch": [], "train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_ep = n_epochs
        no_improve = 0

        for epoch in range(1, n_epochs + 1):
            net.train()
            perm = torch.randperm(n_samples, device=device)
            tr_loss_sum = 0.0
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_tr_t[idx]
                yb = y_tr_t[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                tr_loss_sum += float(loss.item()) * len(idx)

            scheduler.step()
            train_loss = tr_loss_sum / n_samples
            self.history_["epoch"].append(epoch)
            self.history_["train_loss"].append(train_loss)

            if has_val:
                net.eval()
                val_loss_sum = 0.0
                n_val = X_va_t.shape[0]
                with torch.no_grad():
                    for start in range(0, n_val, self.batch_size * 4):
                        xb = X_va_t[start : start + self.batch_size * 4]
                        yb = y_va_t[start : start + self.batch_size * 4]
                        logits = net(xb)
                        loss = criterion(logits, yb)
                        val_loss_sum += float(loss.item()) * len(xb)
                val_loss = val_loss_sum / n_val
                self.history_["val_loss"].append(val_loss)

                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_ep = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if fixed_epochs is None and no_improve >= self.patience:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                best_ep = epoch

        self.state_dict_cpu_ = best_state
        self.best_epoch_ = best_ep
        self.best_val_loss_ = best_val if has_val else None

        # Free GPU tensors
        del X_tr_t, y_tr_t, net
        if has_val:
            del X_va_t, y_va_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.state_dict_cpu_ is None or self.in_dim_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        device = torch.device(self.device_str if (self.device_str == "cpu" or torch.cuda.is_available()) else "cpu")
        X_s = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)

        net = self._build_net(self.in_dim_, device)
        net.load_state_dict({k: v.to(device) for k, v in self.state_dict_cpu_.items()})
        net.eval()

        n = X_s.shape[0]
        probs = np.empty(n, dtype=np.float64)
        bs = self.batch_size * 4
        with torch.no_grad():
            for start in range(0, n, bs):
                xb = torch.from_numpy(X_s[start : start + bs]).to(device)
                p = torch.sigmoid(net(xb)).cpu().numpy()
                probs[start : start + bs] = p

        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return np.column_stack([1.0 - probs, probs])
