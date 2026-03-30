# config.py — All hyperparameters in one place

TARGET_IDX  = 4          # QM9 target: HOMO-LUMO gap (Hartree)
TARGET_NAME = "HOMO-LUMO Gap"
TARGET_UNIT = "Hartree"

# Data
TRAIN_SIZE  = 110_000
VAL_SIZE    = 10_000
# remainder (~10,831) → test
SEED        = 42

# Model
HIDDEN_DIM  = 64
NUM_LAYERS  = 3
DROPOUT     = 0.1

# Training
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
WEIGHT_DECAY= 1e-5
PATIENCE    = 15         # early stopping patience
LR_PATIENCE = 5          # scheduler patience
LR_FACTOR   = 0.5
LR_MIN      = 1e-6
GRAD_CLIP   = 1.0

# Paths
DATA_ROOT   = "data/qm9"
CHECKPOINT  = "checkpoints/best_model.pt"
RESULTS     = "results/results.json"
PLOT_PATH   = "results/training_curves.png"