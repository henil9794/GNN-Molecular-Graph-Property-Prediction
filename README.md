# MolGraph: Quantum-Chemical Property Prediction with GNNs

Predicts the **HOMO-LUMO gap** of small organic molecules using a
**Message Passing Neural Network (MPNN)** trained on the QM9 quantum-chemistry dataset.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.4%2B-red)

---

## Background

The **HOMO-LUMO gap** measures the energy difference between the highest occupied
and lowest unoccupied molecular orbitals — a key descriptor of a molecule's
electronic structure, reactivity, and optical properties. Accurate prediction of
this property is relevant in:

- 💊 **Drug discovery** — molecular stability and reactivity screening
- 🔋 **Materials science** — designing organic semiconductors and solar cells
- 🧬 **Molecular design** — guiding generative models toward stable candidates

---

## Model Architecture

```
Raw Atom Features (11-dim)
        │
        ▼
Linear Embedding (→ 64-dim)
        │
        ▼
┌───────────────────────────────────┐
│  NNConv Layer  ×3                 │
│  bond features (4-dim) condition  │
│  the convolution weight matrix    │
│  ── BatchNorm ── ReLU ── Dropout  │
└───────────────────────────────────┘
        │
        ▼
Global Mean Pooling → molecule vector (64-dim)
        │
        ▼
MLP Readout  (64 → 32 → 1)
        │
        ▼
Predicted Property (scalar)
```

**Why NNConv over plain GCN?**
Molecules have typed bonds (single, double, triple, aromatic). NNConv uses edge
features to *dynamically parameterize* the weight matrix at each message passing
step — allowing the model to learn different aggregation behavior per bond type.
A standard GCNConv would discard this structural information entirely.

---

## Dataset: QM9

| Property | Value |
|---|---|
| Total molecules | 130,831 |
| Atom types | C, H, N, O, F |
| Node features | 11 (atom type, hybridization, aromaticity, H-count, ring membership) |
| Edge features | 4 (bond type: single / double / triple / aromatic) |
| Target predicted | HOMO-LUMO gap (index 4, Hartree) |
| Train / Val / Test split | 110,000 / 10,000 / 10,831 |

---

## Project Structure

```
gnn-mol-property/
├── config.py        # All hyperparameters
├── dataset.py       # QM9 loading, splitting, normalization
├── model.py         # MPNN architecture (NNConv + global pooling)
├── train.py         # Training loop, early stopping, checkpointing
├── evaluate.py      # Test evaluation, scatter plots, error distribution
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/gnn-mol-property
cd gnn-mol-property

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train — QM9 (~1.3 GB) downloads automatically on first run
python train.py

# 4. Evaluate and generate plots
python evaluate.py
```

---

## Results

| Metric | Value |
|---|---|
| Test MAE | ~0.0065 Hartree |
| Test RMSE | ~0.0091 Hartree |
| Model parameters | ~47K |
| Training time (CPU) | ~3 hrs |
| Training time (GPU) | ~25 min |

---

## Key Concepts Demonstrated

- **Graph Neural Networks (GNNs)** for molecular property prediction
- **Edge-conditioned message passing** (NNConv) using typed bond features
- **Global mean pooling** to produce fixed-size representations from variable-length molecular graphs
- **Regression on quantum-chemical targets** with real-world life science relevance
- Strict **train/val/test splits** with zero data leakage
- **Early stopping**, ReduceLROnPlateau scheduling, and gradient clipping
- Target **z-normalization** with denormalized MAE reporting

---

## References

- Ramakrishnan et al. (2014) — [QM9 Dataset](https://www.nature.com/articles/sdata201422)
- Gilmer et al. (2017) — [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- PyTorch Geometric — [Documentation](https://pytorch-geometric.readthedocs.io/)

---