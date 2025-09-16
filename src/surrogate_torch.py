# src/surrogate_torch.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def _select_device(prefer: str | None = None):
    # Try CUDA, XPU (Intel), DirectML, MPS, then CPU
    if prefer is None:
        prefer = os.environ.get("TORCH_DEVICE_PREFERENCE")
    if prefer:
        p = prefer.lower()
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if p in ("dml", "directml"):
            try:
                import torch_directml
                return torch_directml.device()
            except Exception:
                pass
        if p == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=805, hidden=[256, 256, 128], dropout_rate=0.1):
        super().__init__()
        layers = []
        last_dim = in_dim

        for h in hidden:
            layers.extend([
                nn.Linear(last_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout_rate),
            ])
            last_dim = h

        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


def train_and_save_torch(
    X,
    Y,
    path="models/mlp_torch.pt",
    epochs=20,
    batch_size=128,
    lr=1e-3,
    hidden=[256, 256, 128],
    validation_split=0.1,
    patience=5,
    prefer_device: str | None = None,
):
    device = _select_device(prefer_device)
    print(f"Using device: {device}")

    dataset_size = len(X)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    train_dataset = TensorDataset(X_tensor[train_indices], Y_tensor[train_indices])
    val_dataset = TensorDataset(X_tensor[val_indices], Y_tensor[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = SurrogateMLP(in_dim=X.shape[1], out_dim=Y.shape[1], hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None

    from tqdm import tqdm

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')

        for xb, yb in train_batches:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_batches.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = train_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_batches = tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [Valid]')
            for xb, yb in val_batches:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_batches.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_val_loss = val_loss / len(val_dataset)
        scheduler.step(avg_val_loss)

        print(f"[Epoch {epoch}/{epochs}] train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'hidden_layers': hidden,
    }, path)
    print("Model saved:", path)
    return path


def load_model_torch(path, in_dim=2, out_dim=805, hidden=[256, 256, 128], device=None, prefer_device: str | None = None):
    try:
        if device is None:
            device = _select_device(prefer_device)

        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and 'hidden_layers' in checkpoint:
            hidden = checkpoint['hidden_layers']

        model = SurrogateMLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden).to(device)

        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    except Exception as e:
        print(f"Load error: {str(e)}")
        print(f"Model path: {path}")
        print(f"Args: in_dim={in_dim}, out_dim={out_dim}, hidden={hidden}")
        raise

