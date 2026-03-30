import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import (
    HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    LR_PATIENCE, LR_FACTOR, LR_MIN, GRAD_CLIP,
    CHECKPOINT, TARGET_NAME, TARGET_UNIT
)
from dataset import load_qm9
from model import MPNN, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Per-epoch helpers ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        pred = model(data)                              # [B, 1] normalized
        loss = F.mse_loss(pred, data.y.float())        # MSE on normalized target
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, std: float) -> float:
    """Returns MAE in original (un-normalized) units."""
    model.eval()
    total_mae = 0.0
    for data in loader:
        data  = data.to(DEVICE)
        pred  = model(data)
        # Denormalize: multiply by std (mean cancels in absolute error)
        mae   = ((pred - data.y.float()).abs() * std).sum().item()
        total_mae += mae
    return total_mae / len(loader.dataset)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print(f"  GNN Molecular Property Prediction")
    print(f"  Target : {TARGET_NAME}")
    print(f"  Device : {DEVICE}")
    print(f"{'='*55}\n")

    # Data
    print("Loading QM9 dataset …")
    train_loader, val_loader, test_loader, mean, std, node_dim, edge_dim = load_qm9()
    print(f"  Node features : {node_dim} | Edge features : {edge_dim}\n")

    # Model
    model = MPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Trainable parameters: {count_parameters(model):,}\n")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        patience=LR_PATIENCE, factor=LR_FACTOR, min_lr=LR_MIN
    )

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_val_mae    = float("inf")
    patience_counter = 0
    history = {"train_mse": [], "val_mae": []}

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer)
        val_mae   = evaluate(model, val_loader, std)
        scheduler.step(val_mae)

        history["train_mse"].append(train_mse)
        history["val_mae"].append(val_mae)

        # Checkpoint on improvement
        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            patience_counter += 1

        # Progress log
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train MSE: {train_mse:.5f} | "
                f"Val MAE: {val_mae:.5f} {TARGET_UNIT} | "
                f"LR: {lr:.2e} | "
                f"Patience: {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # ── Final evaluation ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(CHECKPOINT))
    test_mae = evaluate(model, test_loader, std)

    print(f"\n{'='*55}")
    print(f"  Best Val MAE : {best_val_mae:.5f} {TARGET_UNIT}")
    print(f"  Test MAE     : {test_mae:.5f} {TARGET_UNIT}")
    print(f"{'='*55}\n")

    # Save results + history
    results = {
        "target":         TARGET_NAME,
        "best_val_mae":   round(best_val_mae, 6),
        "test_mae":       round(test_mae, 6),
        "unit":           TARGET_UNIT,
        "epochs_trained": len(history["train_mse"]),
        "model_params":   count_parameters(model),
        "hidden_dim":     HIDDEN_DIM,
        "num_layers":     NUM_LAYERS,
    }
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("results/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Saved → results/results.json | results/history.json")
    print("Run evaluate.py to generate plots.")


if __name__ == "__main__":
    main()