import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from config import (
    TARGET_IDX, TRAIN_SIZE, VAL_SIZE, SEED,
    BATCH_SIZE, DATA_ROOT
)


def load_qm9():
    """
    Downloads QM9 (if needed), selects a single regression target,
    z-normalizes it, and returns DataLoaders + normalization stats.

    QM9 contains 130,831 small organic molecules with up to 9 heavy atoms
    (C, H, O, N, F). Each molecule has:
      - Node features (x):       11-dim  (atom type, hybridization, etc.)
      - Edge features (edge_attr): 4-dim  (bond type: single/double/triple/aromatic)
      - 19 quantum-chemical targets
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    dataset = QM9(root=DATA_ROOT)

    # ── Select & normalize target ──────────────────────────────────────────
    # dataset.data.y shape: [N_molecules, 19]
    y_all = dataset.data.y[:, TARGET_IDX].unsqueeze(1)   # [N, 1]
    mean  = y_all.mean().item()
    std   = y_all.std().item()
    dataset.data.y = (y_all - mean) / std                # in-place normalize

    # ── Reproducible split ─────────────────────────────────────────────────
    torch.manual_seed(SEED)
    perm = torch.randperm(len(dataset))

    train_idx = perm[:TRAIN_SIZE]
    val_idx   = perm[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_idx  = perm[TRAIN_SIZE + VAL_SIZE :]

    train_set = dataset[train_idx]
    val_set   = dataset[val_idx]
    test_set  = dataset[test_idx]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"  Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")
    print(f"  Target mean: {mean:.5f} {chr(10)}  Target std:  {std:.5f}")

    # Infer feature dims from first batch
    sample     = next(iter(train_loader))
    node_dim   = sample.x.shape[1]     # 11
    edge_dim   = sample.edge_attr.shape[1]  # 4

    return train_loader, val_loader, test_loader, mean, std, node_dim, edge_dim