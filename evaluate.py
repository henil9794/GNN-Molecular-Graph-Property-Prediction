import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from config import CHECKPOINT, TARGET_NAME, TARGET_UNIT, TARGET_IDX
from dataset import load_qm9
from model import MPNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def collect_predictions(model, loader, std, mean):
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data).squeeze()          # normalized
        y    = data.y.float().squeeze()       # normalized
        # Denormalize
        preds.extend((pred * std + mean).cpu().numpy())
        targets.extend((y   * std + mean).cpu().numpy())
    return np.array(preds), np.array(targets)


def plot_results(preds, targets, history, unit):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── 1. Training curves ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history["train_mse"], label="Train MSE (normalized)", color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["val_mae"], label=f"Val MAE ({unit})", color="darkorange")
    best = min(history["val_mae"])
    ax.axhline(best, linestyle="--", color="red", label=f"Best: {best:.5f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel(f"MAE ({unit})")
    ax.set_title("Validation MAE"); ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Predicted vs Actual ──────────────────────────────────────────
    ax = axes[2]
    ax.scatter(targets, preds, alpha=0.3, s=5, color="steelblue", label="Predictions")
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    mae = np.abs(preds - targets).mean()
    ax.set_xlabel(f"True {TARGET_NAME} ({unit})")
    ax.set_ylabel(f"Predicted {TARGET_NAME} ({unit})")
    ax.set_title(f"Predicted vs Actual  |  MAE={mae:.5f} {unit}")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("GNN Molecular Property Prediction — QM9", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("results/evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → results/evaluation_plots.png")


def main():
    print("Loading data …")
    _, _, test_loader, mean, std, node_dim, edge_dim = load_qm9()

    model = MPNN(node_dim=node_dim, edge_dim=edge_dim).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print("Checkpoint loaded.")

    preds, targets = collect_predictions(model, test_loader, std, mean)

    mae  = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    print(f"\nTest MAE  : {mae:.5f} {TARGET_UNIT}")
    print(f"Test RMSE : {rmse:.5f} {TARGET_UNIT}")

    with open("results/history.json") as f:
        history = json.load(f)

    plot_results(preds, targets, history, TARGET_UNIT)


if __name__ == "__main__":
    main()