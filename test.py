import subprocess
import os

# Path to your Python interpreter
PYTHON = "/usr/bin/python3.10"

# 🎯 Experiment parameters
MODEL = "googlenet"
DATASET = "cifar100"
SPARSITY = 0.20
PRUNE_STRATEGY = "taylor"
Q_FLOP_COEF = 0.00
Q_PARA_COEF = 0.00

# 📁 Directory and path setup
LOG_DIR = "log"
CKPT_BASE = "checkpoint"
PRETRAINED_MODEL_DIR = "pretrained_model"
COMPRESSED_MODEL_DIR = "compressed_model"

# 🧮 Format numerical values for path consistency
SPARSITY_STR = f"{SPARSITY:.2f}"
Q_FLOP_STR = f"{Q_FLOP_COEF:.2f}"
Q_PARA_STR = f"{Q_PARA_COEF:.2f}"

CHECKPOINT_DIR = f"{CKPT_BASE}/{MODEL}_{DATASET}_{SPARSITY_STR}_{Q_FLOP_STR}_{Q_PARA_STR}"
PRETRAINED_MODEL_PATH = f"{PRETRAINED_MODEL_DIR}/{MODEL}_{DATASET}_pretrained.pth"
COMPRESSED_MODEL_PATH = f"{COMPRESSED_MODEL_DIR}/{MODEL}_{DATASET}_{SPARSITY_STR}_{Q_FLOP_STR}_{Q_PARA_STR}.pth"

# 🔧 Make sure directories exist
for folder in [LOG_DIR, CHECKPOINT_DIR, PRETRAINED_MODEL_DIR, COMPRESSED_MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

def run_command(command_list, step_name=""):
    print(f"\n🔹 Running: {step_name}")
    print(" ".join(command_list))
    try:
        subprocess.run(command_list, check=True)
        print(f"✅ {step_name} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {step_name}:")
        print(e)
        exit(1)

def main():
    # ✅ Step 1: Train the model (or skip if already trained)
    run_command([
        PYTHON, "-m", "train",
        "--model", MODEL,
        "--dataset", DATASET,
        "--device", "cuda",
        "--output_dir", PRETRAINED_MODEL_DIR,
        "--log_dir", LOG_DIR,
        "--use_wandb"
    ], "Model Training")

    # ✅ Step 2: Compress the trained model
    run_command([
        PYTHON, "-m", "compress",
        "--model", MODEL,
        "--dataset", DATASET,
        "--device", "cuda",
        "--sparsity", SPARSITY_STR,
        "--prune_strategy", PRUNE_STRATEGY,
        "--ppo",
        "--Q_FLOP_coef", Q_FLOP_STR,
        "--Q_Para_coef", Q_PARA_STR,
        "--pretrained_pth", PRETRAINED_MODEL_PATH,
        "--compressed_dir", COMPRESSED_MODEL_DIR,
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--log_dir", LOG_DIR,
        "--use_wandb",
        "--save_model"
    ], "Model Compression")

    # ✅ Step 3: Evaluate the compression results
    run_command([
        PYTHON, "-m", "evaluate",
        "--model", MODEL,
        "--dataset", DATASET,
        "--device", "cuda",
        "--pretrained_pth", PRETRAINED_MODEL_PATH,
        "--compressed_pth", COMPRESSED_MODEL_PATH,
        "--log_dir", LOG_DIR
    ], "Model Evaluation")

if __name__ == "__main__":
    main()
