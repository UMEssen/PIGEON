"""
Stage 4: Fine-tune MedGemma-27B using Unsloth with LoRA adapters.

This script fine-tunes a medical language model for structured information
extraction from German medical texts. It uses:
- Unsloth for optimized 4-bit quantization and LoRA training
- SFTTrainer from TRL for supervised fine-tuning
- train_on_responses_only to only compute loss on model outputs

The training data is expected in JSON format with a 'text' column containing
Gemma chat-template formatted examples (produced by Stage 3).

Usage:
    python finetune_gemma.py

To resume from a checkpoint, set LOAD_FROM to the checkpoint path below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only


# =====================================================================
# Configuration
# =====================================================================

# Base model to fine-tune. Use a HuggingFace model ID for fresh training,
# or a local checkpoint path to resume training.
BASE_MODEL = "google/medgemma-27b-text-it"

# Set to a checkpoint directory to resume training (e.g., FINETUNED_DIR / "checkpoint-960").
# When loading from a checkpoint, LoRA adapters are already present and
# get_peft_model() is skipped automatically.
LOAD_FROM = None  # Example: FINETUNED_DIR / "checkpoint-960"

# LoRA hyperparameters
LORA_R = 16           # LoRA rank -- higher = more capacity but risk of overfitting
LORA_ALPHA = 16       # Scaling factor -- recommended to match r
LORA_DROPOUT = 0      # Dropout on LoRA layers (0 = no dropout)

# Training hyperparameters
BATCH_SIZE = 4                 # Per-device batch size
GRADIENT_ACCUMULATION = 16     # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION = 64
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 8192          # Maximum sequence length for training
NUM_EPOCHS = 1
SAVE_STEPS = 50                # Save a checkpoint every N steps
WARMUP_STEPS = 5
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.01

# Paths (derived from config.py)
TRAIN_JSON = TRAINING_READY_DIR / "train.json"   # Update to your actual filename
OUTPUT_DIR = FINETUNED_DIR / "medgemma-27b"


# =====================================================================
# Model loading
# =====================================================================

def load_model_and_tokenizer():
    """Load the base model (or checkpoint) with 4-bit quantization via Unsloth.

    Returns:
        Tuple of (model, tokenizer).
    """
    model_path = str(LOAD_FROM) if LOAD_FROM else BASE_MODEL
    is_checkpoint = LOAD_FROM is not None and "checkpoint" in str(LOAD_FROM)

    print(f"Loading model from: {model_path}")
    print(f"  4-bit quantization: enabled")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Loading from checkpoint: {is_checkpoint}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=HF_TOKEN if HF_TOKEN else None,
        attn_implementation="eager",
        cache_dir=str(HF_CACHE_DIR),
    )

    # Only add LoRA adapters when starting from a base model.
    # Checkpoints already have adapters baked in.
    if not is_checkpoint:
        print("Adding LoRA adapters to base model...")
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,      # Text-only model
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            random_state=RANDOM_SEED,
        )
    else:
        print("Checkpoint detected -- LoRA adapters already present, skipping get_peft_model()")

    return model, tokenizer


# =====================================================================
# Dataset loading
# =====================================================================

def load_training_data(train_path: Path):
    """Load the training dataset from a JSON file.

    The JSON file should have one JSON object per line with a 'text' field
    containing the Gemma chat-template formatted training example.

    Args:
        train_path: Path to the training JSON file.

    Returns:
        HuggingFace Dataset.
    """
    print(f"Loading training data from: {train_path}")
    dataset = load_dataset("json", data_files=str(train_path), split="train")
    print(f"Training examples: {len(dataset):,}")
    return dataset


# =====================================================================
# Training
# =====================================================================

def create_trainer(model, tokenizer, dataset):
    """Set up the SFTTrainer with response-only loss masking.

    The trainer uses Gemma chat markers to identify which tokens are
    model responses (and should contribute to the loss) vs. user
    instructions (which are masked out).

    Args:
        model: The LoRA-adapted model.
        tokenizer: The model tokenizer.
        dataset: HuggingFace Dataset with 'text' column.

    Returns:
        Configured SFTTrainer.
    """
    output_dir = str(OUTPUT_DIR)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            max_grad_norm=MAX_GRAD_NORM,
            save_steps=SAVE_STEPS,
            save_strategy="steps",
            save_total_limit=30,
            optim="adamw_8bit",
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="constant",
            seed=RANDOM_SEED,
            report_to="none",  # Set to "wandb" to enable Weights & Biases logging
        ),
    )

    # Mask instruction tokens so the loss is computed only on model responses.
    # This uses Gemma's turn markers to find the boundary.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    return trainer


def print_gpu_stats():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("No CUDA device available.")
        return

    props = torch.cuda.get_device_properties(0)
    reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    total = round(props.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {props.name}")
    print(f"  Total memory:    {total} GB")
    print(f"  Reserved memory: {reserved} GB")


def print_trainable_parameters(model):
    """Print the number of trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


# =====================================================================
# Main
# =====================================================================

def main():
    """Run the full fine-tuning pipeline."""
    ensure_dirs()

    # 1. Load model
    print("=" * 60)
    print("STEP 1: Loading model and tokenizer")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer()
    print_trainable_parameters(model)

    # 2. Load dataset
    print("\n" + "=" * 60)
    print("STEP 2: Loading training data")
    print("=" * 60)
    dataset = load_training_data(TRAIN_JSON)

    # 3. Create trainer
    print("\n" + "=" * 60)
    print("STEP 3: Setting up trainer")
    print("=" * 60)
    trainer = create_trainer(model, tokenizer, dataset)

    # Quick sanity check: decode a training example to verify formatting
    if len(dataset) > 0:
        sample_idx = min(100, len(dataset) - 1)
        print(f"\nSample decoded (index {sample_idx}):")
        print(tokenizer.decode(trainer.train_dataset[sample_idx]["input_ids"])[:500])

    # 4. Print GPU stats before training
    print("\n" + "=" * 60)
    print("GPU stats before training:")
    print_gpu_stats()

    # 5. Train
    print("\n" + "=" * 60)
    print("STEP 4: Training")
    print("=" * 60)

    # Resume from checkpoint if LOAD_FROM is set and is a checkpoint directory
    resume_path = None
    if LOAD_FROM and Path(LOAD_FROM).exists() and "checkpoint" in str(LOAD_FROM):
        resume_path = str(LOAD_FROM)
        print(f"Resuming from checkpoint: {resume_path}")

    if resume_path:
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    # 6. Save final model
    print("\n" + "=" * 60)
    print("STEP 5: Saving model")
    print("=" * 60)
    trainer.save_model()
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved to: {OUTPUT_DIR}")

    # 7. Final GPU stats
    print("\nGPU stats after training:")
    print_gpu_stats()

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
