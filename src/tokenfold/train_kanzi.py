#!/usr/bin/env python3
"""
Training script for Kanzi-based structure prediction.

Predicts Kanzi tokens (1000 vocab) instead of Foldseek 3Di (20 vocab).
Kanzi tokens can be decoded to 3D C-alpha coordinates for RMSD validation.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from .dataset import (
    AA_START,
    CONTACTS_START,
    KANZI_START,
    KANZI_TOKEN_PREFIX,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_CONTACTS,
    KanziStructureDataset,
    add_kanzi_tokens,
)
from .kanzi_tokenizer import KanziTokenizer

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_X = "X"  # Unknown/invalid amino acid

# Standard amino acid 3-letter codes
AA_3LETTER = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    'X': 'UNK',
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def coords_to_pdb(coords: np.ndarray, aa_seq: str, chain_id: str = "A") -> str:
    """Convert C-alpha coordinates to PDB format string.

    Args:
        coords: C-alpha coordinates in Angstroms, shape (L, 3).
        aa_seq: Amino acid sequence (1-letter codes).
        chain_id: Chain identifier.

    Returns:
        PDB format string.
    """
    lines = []
    for i, (coord, aa) in enumerate(zip(coords, aa_seq)):
        res_name = AA_3LETTER.get(aa.upper(), 'UNK')
        res_num = i + 1
        x, y, z = coord
        # PDB ATOM record format
        line = (
            f"ATOM  {i+1:5d}  CA  {res_name} {chain_id}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


def create_minimal_tokenizer():
    """Create a minimal tokenizer with only amino acids and Kanzi tokens.

    Vocabulary:
    - Special tokens: <PAD>, <EOS>, <BOS>, <UNK>
    - Natural language delimiters: "Protein", "sequence:", "Structure:", "Contacts:"
    - Amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, X
    - Kanzi tokens: <K0>, <K1>, ..., <K999>
    - Contact tokens: numbers 1-999, hyphen for "i-j" format

    Total: ~2030 tokens
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    # Build vocabulary
    vocab = {}
    idx = 0

    # Special tokens (must be first for compatibility)
    special_tokens = ["<PAD>", "<EOS>", "<BOS>", "<UNK>"]
    for token in special_tokens:
        vocab[token] = idx
        idx += 1

    # Natural language delimiters (split by whitespace since tokenizer uses whitespace)
    # "Protein sequence:" -> "Protein", "sequence:"
    # "Structure:" -> "Structure:"
    # "Contacts:" -> "Contacts:"
    delimiter_tokens = ["Protein", "sequence:", "Structure:", "Contacts:"]
    for token in delimiter_tokens:
        vocab[token] = idx
        idx += 1

    # Amino acids (as individual characters, since input is space-separated)
    for aa in AMINO_ACIDS + AMINO_ACID_X:
        vocab[aa] = idx
        idx += 1

    # Numbers for contact indices (1-999 covers protein lengths up to 999)
    for num in range(1, 1000):
        vocab[str(num)] = idx
        idx += 1

    # Hyphen for contact format "i-j"
    vocab["-"] = idx
    idx += 1

    # Kanzi tokens
    for i in range(1000):
        vocab[f"{KANZI_TOKEN_PREFIX}{i}>"] = idx
        idx += 1

    logger.info(f"Created minimal vocabulary with {len(vocab)} tokens")

    # Create tokenizer using WordLevel model (exact token matching)
    tokenizer_model = models.WordLevel(vocab=vocab, unk_token="<UNK>")
    tokenizer = Tokenizer(tokenizer_model)

    # Use whitespace pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Convert to HuggingFace tokenizer
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<PAD>",
        eos_token="<EOS>",
        bos_token="<BOS>",
    )

    return hf_tokenizer


def create_model_and_tokenizer(
    model_name: str,
    kanzi_checkpoint: str,
    use_flash_attn: bool = True,
    gradient_checkpointing: bool = True,
    from_scratch: bool = False,
):
    """Create model and tokenizer with Kanzi vocabulary.

    Args:
        model_name: HuggingFace model name (used for config if from_scratch=True).
        kanzi_checkpoint: Path to Kanzi model checkpoint.
        use_flash_attn: Whether to use Flash Attention 2.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        from_scratch: If True, use minimal vocabulary and random initialization.
    """
    from transformers import AutoConfig

    if from_scratch:
        # Create minimal tokenizer with only amino acids + Kanzi tokens
        tokenizer = create_minimal_tokenizer()
        logger.info(f"Created minimal tokenizer with {len(tokenizer)} tokens")

        # Load config from model_name but initialize weights randomly
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.vocab_size = len(tokenizer)

        if use_flash_attn:
            config._attn_implementation = "flash_attention_2"

        # Initialize model with random weights
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        logger.info(f"Initialized {model_name} from scratch with vocab size {config.vocab_size}")
    else:
        # Load pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Add Kanzi tokens (1000 structure tokens + special tokens)
        num_added = add_kanzi_tokens(tokenizer)
        logger.info(f"Added {num_added} Kanzi tokens to vocabulary")

        # Load pretrained model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Resize embeddings for new tokens
        model.resize_token_embeddings(len(tokenizer))

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load Kanzi tokenizer
    kanzi_tokenizer = KanziTokenizer(checkpoint_path=kanzi_checkpoint)
    logger.info(f"Loaded Kanzi tokenizer from {kanzi_checkpoint}")

    return model, tokenizer, kanzi_tokenizer


def create_dataloaders(
    tokenizer,
    kanzi_tokenizer,
    db_path: str,
    batch_size: int = 8,
    max_length: int = 1024,
    max_protein_length: int = 400,
    min_protein_length: int = 100,
    use_contacts: bool = False,
    max_contacts: int = 50,
    contact_prob: float = 1.0,
    use_system_prompt: bool = True,
    num_workers: int = 0,  # Use 0 for GPU-based Kanzi encoding
):
    """Create train and validation dataloaders."""

    train_dataset = KanziStructureDataset(
        db_path=db_path,
        tokenizer=tokenizer,
        kanzi_tokenizer=kanzi_tokenizer,
        max_length=max_length,
        max_protein_length=max_protein_length,
        min_protein_length=min_protein_length,
        use_contacts=use_contacts,
        max_contacts=max_contacts,
        contact_prob=contact_prob,
        use_system_prompt=use_system_prompt,
        split="train",
    )

    val_dataset = KanziStructureDataset(
        db_path=db_path,
        tokenizer=tokenizer,
        kanzi_tokenizer=kanzi_tokenizer,
        max_length=max_length,
        max_protein_length=max_protein_length,
        min_protein_length=min_protein_length,
        use_contacts=use_contacts,
        max_contacts=max_contacts,
        contact_prob=1.0,  # Always use all contacts for validation
        use_system_prompt=use_system_prompt,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_rmsd_eval(
    model,
    tokenizer,
    kanzi_tokenizer,
    eval_samples: list[tuple[str, np.ndarray]],
    device,
    max_new_tokens: int = 512,
    use_system_prompt: bool = True,
) -> tuple[dict, list[dict]]:
    """Run RMSD-based evaluation on generated structures.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer with Kanzi tokens.
        kanzi_tokenizer: KanziTokenizer for decoding.
        eval_samples: List of (aa_seq, gt_ca_coords) tuples.
        device: Device to run on.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (metrics_dict, examples_list).
        examples_list contains dicts with best, median, worst examples.
    """
    from kanzi import kabsch_rmsd

    model.eval()

    # Store individual results for selecting examples
    results = []
    token_accuracies = []
    valid_predictions = 0

    for aa_seq, gt_coords in eval_samples:
        # Format input prompt (with optional system prompt)
        aa_spaced = " ".join(aa_seq)
        prefix = SYSTEM_PROMPT_NO_CONTACTS if use_system_prompt else ""
        prompt = f"{prefix}{AA_START} {aa_spaced} {KANZI_START}"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=len(aa_seq) + 10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode and extract Kanzi tokens
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Parse Kanzi tokens from output
        if KANZI_START in decoded:
            kanzi_part = decoded.split(KANZI_START)[1]
            # Extract <K###> tokens
            pred_tokens = []
            for token in kanzi_part.split():
                if token.startswith(KANZI_TOKEN_PREFIX) and token.endswith(">"):
                    try:
                        token_id = int(token[2:-1])  # Extract number from <K###>
                        if 0 <= token_id < 1000:
                            pred_tokens.append(token_id)
                    except ValueError:
                        continue
                if len(pred_tokens) >= len(aa_seq):
                    break
        else:
            pred_tokens = []

        if len(pred_tokens) == 0:
            continue

        valid_predictions += 1

        # Truncate to match lengths
        min_len = min(len(pred_tokens), len(gt_coords))
        pred_tokens = pred_tokens[:min_len]
        gt_coords_trimmed = gt_coords[:min_len]
        aa_seq_trimmed = aa_seq[:min_len]

        # Decode predicted tokens to coordinates
        try:
            pred_coords = kanzi_tokenizer.decode(pred_tokens)

            # Compute RMSD
            rmsd = kabsch_rmsd(
                pred_coords / 10.0,  # Convert to nm
                gt_coords_trimmed / 10.0,
            )
            rmsd_angstrom = float(rmsd * 10.0)  # Back to Angstroms

            # Compute ground truth tokens for token accuracy
            gt_tokens = kanzi_tokenizer.encode(gt_coords_trimmed)
            if len(gt_tokens) == len(pred_tokens):
                correct = sum(p == g for p, g in zip(pred_tokens, gt_tokens))
                token_acc = correct / len(gt_tokens)
                token_accuracies.append(token_acc)
            else:
                gt_tokens = []
                token_acc = 0.0

            # Store result for example selection
            results.append({
                "rmsd": rmsd_angstrom,
                "aa_seq": aa_seq_trimmed,
                "pred_tokens": pred_tokens,
                "gt_tokens": gt_tokens,
                "pred_coords": pred_coords,
                "gt_coords": gt_coords_trimmed,
                "token_accuracy": token_acc,
            })
        except Exception:
            pass

    n = len(eval_samples)
    rmsds = [r["rmsd"] for r in results]

    # Compute fraction of samples below RMSD thresholds (denominator is total samples)
    rmsd_lt_2 = sum(r < 2.0 for r in rmsds) / n if n > 0 else 0.0
    rmsd_lt_4 = sum(r < 4.0 for r in rmsds) / n if n > 0 else 0.0

    # Compute token diversity metrics across all predictions
    all_pred_tokens = []
    per_sample_unique_ratios = []
    for r in results:
        tokens = r["pred_tokens"]
        all_pred_tokens.extend(tokens)
        if len(tokens) > 0:
            unique_ratio = len(set(tokens)) / len(tokens)
            per_sample_unique_ratios.append(unique_ratio)

    # Global diversity: unique tokens across all samples
    unique_tokens_global = len(set(all_pred_tokens)) if all_pred_tokens else 0
    total_tokens = len(all_pred_tokens)

    # Per-sample diversity: average ratio of unique tokens per sequence
    avg_unique_ratio = np.mean(per_sample_unique_ratios) if per_sample_unique_ratios else 0.0

    # Token entropy (higher = more diverse)
    if all_pred_tokens:
        from collections import Counter
        token_counts = Counter(all_pred_tokens)
        probs = np.array(list(token_counts.values())) / total_tokens
        token_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        # Max entropy for reference (if all 1000 tokens equally likely)
        max_entropy = np.log2(1000)
        normalized_entropy = token_entropy / max_entropy
        # Most common token
        most_common_token, most_common_count = token_counts.most_common(1)[0]
        most_common_freq = most_common_count / total_tokens
    else:
        token_entropy = 0.0
        normalized_entropy = 0.0
        most_common_token = -1
        most_common_freq = 0.0

    metrics = {
        "rmsd_mean": np.mean(rmsds) if rmsds else float("nan"),
        "rmsd_median": np.median(rmsds) if rmsds else float("nan"),
        "rmsd_std": np.std(rmsds) if rmsds else float("nan"),
        "rmsd_lt_2A": rmsd_lt_2,
        "rmsd_lt_4A": rmsd_lt_4,
        "token_accuracy": np.mean(token_accuracies) if token_accuracies else 0.0,
        "valid_predictions": valid_predictions / n if n > 0 else 0.0,
        "num_samples": n,
        # Diversity metrics
        "token_diversity/unique_tokens": unique_tokens_global,
        "token_diversity/unique_ratio_per_seq": avg_unique_ratio,
        "token_diversity/entropy": token_entropy,
        "token_diversity/normalized_entropy": normalized_entropy,
        "token_diversity/most_common_token": most_common_token,
        "token_diversity/most_common_freq": most_common_freq,
    }

    # Select best, median, worst examples
    examples = []
    if results:
        sorted_results = sorted(results, key=lambda x: x["rmsd"])
        # Best (lowest RMSD)
        examples.append({"type": "best", **sorted_results[0]})
        # Worst (highest RMSD)
        examples.append({"type": "worst", **sorted_results[-1]})
        # Median
        median_idx = len(sorted_results) // 2
        examples.append({"type": "median", **sorted_results[median_idx]})

    return metrics, examples


def get_eval_samples(
    db_path: str,
    kanzi_tokenizer,
    num_samples: int = 50,
    seed: int = 42,
    max_seq_len: int = 200,
    min_seq_len: int = 100,
):
    """Get fixed eval samples for RMSD evaluation."""
    import random

    from .foldseek_db import PairedFoldseekDB

    rng = random.Random(seed)

    samples = []
    with PairedFoldseekDB(db_path, include_ca=True) as db:
        # Get validation indices (same split as dataset)
        total = len(db)
        all_indices = list(range(total))
        rng_split = random.Random(42)
        rng_split.shuffle(all_indices)
        val_size = int(total * 0.001)
        val_indices = all_indices[:val_size]

        rng.shuffle(val_indices)

        for idx in val_indices:
            if len(samples) >= num_samples:
                break
            aa_seq, _, ca_coords = db.get_triplet(idx)
            if min_seq_len <= len(aa_seq) <= max_seq_len and len(aa_seq) == len(ca_coords):
                samples.append((aa_seq, ca_coords))

    return samples


def train(
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    kanzi_checkpoint: str = "checkpoints/cleaned_model.pt",
    db_path: str = "data/foldseek/afdb50/afdb50",
    output_dir: str = "outputs/kanzi_predictor",
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    max_length: int = 1024,
    max_protein_length: int = 400,
    min_protein_length: int = 100,
    warmup_ratio: float = 0.03,
    warmup_steps: int | None = None,
    use_flash_attn: bool = True,
    log_interval: int = 10,
    save_interval: int = 1000,
    eval_interval: int = 500,
    rmsd_eval_interval: int = 500,
    rmsd_eval_samples: int = 25,
    max_steps: int = -1,
    from_scratch: bool = False,
    use_contacts: bool = False,
    max_contacts: int = 50,
    contact_prob: float = 1.0,
    freeze_base_steps: int = 0,
    resume_from: str | None = None,
):
    """Main training function for Kanzi structure prediction.

    Args:
        from_scratch: If True, train with minimal vocabulary (amino acids + Kanzi tokens only)
                     and random weight initialization.
        freeze_base_steps: Number of steps to freeze base model and only train embeddings.
                          Helps new Kanzi token embeddings learn before full fine-tuning.
        resume_from: Path to checkpoint directory to resume training from.
    """

    # Check if wandb is available (env var or logged in)
    wandb_enabled = os.environ.get("WANDB_API_KEY") or wandb.api.api_key
    if not wandb_enabled:
        raise RuntimeError(
            "Wandb is required. Either set WANDB_API_KEY or run 'wandb login'"
        )

    # Initialize accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using {accelerator.num_processes} GPUs")

    # Initialize Wandb tracking
    if wandb_enabled and accelerator.is_main_process:
        # Get git commit SHA
        import subprocess
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            git_sha = "unknown"

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "structure-prediction"),
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_length": max_length,
                "max_protein_length": max_protein_length,
                "min_protein_length": min_protein_length,
                "warmup_ratio": warmup_ratio,
                "structure_type": "kanzi",
                "from_scratch": from_scratch,
                "use_contacts": use_contacts,
                "max_contacts": max_contacts,
                "contact_prob": contact_prob,
                "freeze_base_steps": freeze_base_steps,
                "git_sha": git_sha,
                "command": " ".join(sys.argv),
            },
        )

    # Create model and tokenizers
    logger.info(f"Loading model: {model_name} (from_scratch={from_scratch})")
    model, tokenizer, kanzi_tokenizer = create_model_and_tokenizer(
        model_name=model_name,
        kanzi_checkpoint=kanzi_checkpoint,
        use_flash_attn=use_flash_attn,
        from_scratch=from_scratch,
    )

    # Create dataloaders
    # Disable system prompt for from_scratch mode (model won't understand natural language)
    use_system_prompt = not from_scratch
    logger.info(f"Loading data from: {db_path} (use_contacts={use_contacts}, use_system_prompt={use_system_prompt})")
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        kanzi_tokenizer=kanzi_tokenizer,
        db_path=db_path,
        batch_size=batch_size,
        max_length=max_length,
        max_protein_length=max_protein_length,
        min_protein_length=min_protein_length,
        use_contacts=use_contacts,
        max_contacts=max_contacts,
        contact_prob=contact_prob,
        use_system_prompt=use_system_prompt,
    )

    # Calculate training steps
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    if max_steps > 0:
        num_training_steps = min(num_training_steps, max_steps)
    num_warmup_steps = warmup_steps if warmup_steps is not None else int(num_training_steps * warmup_ratio)

    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator (scheduler managed separately for resume support)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Resume from checkpoint if specified
    starting_step = 0
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        # Extract step number from checkpoint path (e.g., checkpoint-1000 -> 1000)
        try:
            starting_step = int(Path(resume_from).name.split("-")[-1])
            logger.info(f"Resuming from step {starting_step}")
        except ValueError:
            logger.warning("Could not parse step from checkpoint path, starting from 0")

        # Load only model weights (not optimizer/scheduler to allow LR changes)
        import safetensors.torch
        model_path = Path(resume_from) / "model.safetensors"
        if model_path.exists():
            state_dict = safetensors.torch.load_file(str(model_path))
            unwrapped_model = accelerator.unwrap_model(model)
            # Use strict=False to handle tied weights (lm_head shares embed_tokens)
            missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)
            if missing:
                # Filter out expected missing keys (tied weights)
                truly_missing = [k for k in missing if "lm_head" not in k]
                if truly_missing:
                    logger.warning(f"Missing keys in checkpoint: {truly_missing}")
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using accelerator.load_state()")
            accelerator.load_state(resume_from)

        # Create fresh scheduler for remaining steps with specified LR
        remaining_steps = num_training_steps - starting_step
        remaining_warmup = max(0, num_warmup_steps - starting_step)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=remaining_warmup,
            num_training_steps=remaining_steps,
        )
        logger.info(f"Created new scheduler: LR={learning_rate}, remaining_steps={remaining_steps}, warmup={remaining_warmup}")

    # Freeze base model if doing embedding warmup
    base_frozen = False
    if freeze_base_steps > 0 and not from_scratch:
        logger.info(f"Freezing base model for first {freeze_base_steps} steps (embedding warmup)")
        unwrapped = accelerator.unwrap_model(model)
        # Freeze everything except embeddings
        for name, param in unwrapped.named_parameters():
            if "embed" not in name.lower():
                param.requires_grad = False
        base_frozen = True
        # Log trainable params
        trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        total = sum(p.numel() for p in unwrapped.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})")

    # Get fixed eval samples for RMSD evaluation
    eval_samples = None
    if rmsd_eval_interval > 0 and accelerator.is_main_process:
        logger.info(f"Loading {rmsd_eval_samples} eval samples for RMSD evaluation...")
        eval_samples = get_eval_samples(
            db_path=db_path,
            kanzi_tokenizer=kanzi_tokenizer,
            num_samples=rmsd_eval_samples,
            max_seq_len=max_protein_length,
            min_seq_len=min_protein_length,
        )
        logger.info(f"Loaded {len(eval_samples)} eval samples")

    # Training loop
    model.train()
    global_step = starting_step
    running_loss = 0.0

    progress_bar = tqdm(
        range(num_training_steps),
        desc="Training",
        disable=not accelerator.is_main_process,
        initial=starting_step,
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                # Unfreeze base model after warmup period
                if base_frozen and global_step >= freeze_base_steps:
                    logger.info(f"Step {global_step}: Unfreezing base model (embedding warmup complete)")
                    unwrapped = accelerator.unwrap_model(model)
                    for param in unwrapped.parameters():
                        param.requires_grad = True
                    base_frozen = False
                    trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
                    logger.info(f"Trainable parameters: {trainable:,}")

                # Logging
                if global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    lr = scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}/{num_training_steps} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                        )
                        wandb.log(
                            {"train_loss": float(avg_loss), "learning_rate": float(lr)},
                            step=global_step,
                        )
                    running_loss = 0.0

                # Log example input/output every 100 steps
                if global_step % 100 == 0 and accelerator.is_main_process:
                    model.eval()
                    with torch.no_grad():
                        # Get first sample from current batch
                        sample_input = batch["input_ids"][0:1]
                        sample_mask = batch["attention_mask"][0:1]

                        # Find the first Kanzi token (<K0> through <K999>) to determine prompt end
                        kanzi_token_ids = set(
                            tokenizer.convert_tokens_to_ids(f"{KANZI_TOKEN_PREFIX}{i}>")
                            for i in range(1000)
                        )
                        kanzi_token_ids.discard(None)
                        kanzi_token_ids.discard(tokenizer.unk_token_id)

                        input_ids_list = sample_input[0].tolist()
                        kanzi_pos = None
                        for pos, tid in enumerate(input_ids_list):
                            if tid in kanzi_token_ids:
                                kanzi_pos = pos
                                break

                        try:
                            if kanzi_pos is not None:
                                # Prompt includes everything up to (not including) first Kanzi token
                                prompt_ids = sample_input[:, :kanzi_pos]
                                prompt_mask = sample_mask[:, :kanzi_pos]

                                # Decode just the prompt (not the full training example)
                                prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

                                # Generate from the prompt
                                unwrapped = accelerator.unwrap_model(model)
                                generated = unwrapped.generate(
                                    input_ids=prompt_ids,
                                    attention_mask=prompt_mask,
                                    max_new_tokens=200,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                )
                                pred_text = tokenizer.decode(generated[0], skip_special_tokens=False)

                                # Log to wandb as a table
                                example_table = wandb.Table(
                                    columns=["step", "prompt", "prediction"],
                                    data=[[global_step, prompt_text[:2000], pred_text[:2000]]]
                                )
                                wandb.log({"examples": example_table}, step=global_step)

                                # Also log to console (truncated)
                                logger.info(f"Example prompt: {prompt_text[:2000]}...")
                                logger.info(f"Example pred:   {pred_text[:2000]}...")
                        except Exception:
                            pass  # Skip if generation fails
                    model.train()

                # Evaluation
                if eval_interval > 0 and global_step % eval_interval == 0:
                    model.eval()
                    eval_loss = 0.0
                    eval_steps = 0
                    max_eval_steps = 50

                    with torch.no_grad():
                        eval_iter = iter(val_loader)
                        for _ in range(max_eval_steps):
                            try:
                                eval_batch = next(eval_iter)
                            except StopIteration:
                                eval_iter = iter(val_loader)
                                eval_batch = next(eval_iter)

                            outputs = model(
                                input_ids=eval_batch["input_ids"],
                                attention_mask=eval_batch["attention_mask"],
                                labels=eval_batch["labels"],
                            )
                            eval_loss += outputs.loss.detach().float()
                            eval_steps += 1

                    eval_loss_tensor = torch.tensor([eval_loss], device=accelerator.device)
                    gathered_losses = accelerator.gather(eval_loss_tensor)
                    avg_eval_loss = gathered_losses.mean() / eval_steps

                    if accelerator.is_main_process:
                        logger.info(f"Step {global_step} | Eval Loss: {avg_eval_loss:.4f}")
                        wandb.log({"eval_loss": float(avg_eval_loss)}, step=global_step)
                    model.train()

                # RMSD evaluation (independent from standard eval)
                if (
                    rmsd_eval_interval > 0
                    and global_step % rmsd_eval_interval == 0
                    and eval_samples
                    and accelerator.is_main_process
                ):
                    model.eval()
                    logger.info(f"Running RMSD evaluation ({len(eval_samples)} samples)...")
                    unwrapped_model = accelerator.unwrap_model(model)
                    rmsd_metrics, examples = run_rmsd_eval(
                        model=unwrapped_model,
                        tokenizer=tokenizer,
                        kanzi_tokenizer=kanzi_tokenizer,
                        eval_samples=eval_samples,
                        device=accelerator.device,
                        use_system_prompt=use_system_prompt,
                    )
                    logger.info(
                        f"Step {global_step} | RMSD: {rmsd_metrics['rmsd_mean']:.2f} Å, "
                        f"Token Acc: {rmsd_metrics['token_accuracy']:.4f}, "
                        f"Unique tokens: {rmsd_metrics['token_diversity/unique_tokens']}, "
                        f"Most common: <K{rmsd_metrics['token_diversity/most_common_token']}> ({rmsd_metrics['token_diversity/most_common_freq']:.1%})"
                    )
                    wandb.log(rmsd_metrics, step=global_step)

                    # Log example structures to wandb
                    if examples:
                        temp_files = []  # Track temp files to clean up after logging
                        try:
                            # Create wandb table with examples
                            columns = [
                                "type", "rmsd", "token_acc", "length",
                                "sequence", "pred_tokens", "gt_tokens",
                                "pred_structure", "gt_structure"
                            ]
                            data = []

                            for ex in examples:
                                # Create PDB strings
                                pred_pdb = coords_to_pdb(
                                    ex["pred_coords"], ex["aa_seq"]
                                )
                                gt_pdb = coords_to_pdb(
                                    ex["gt_coords"], ex["aa_seq"]
                                )

                                # Write to temp files for wandb.Molecule
                                with tempfile.NamedTemporaryFile(
                                    mode='w', suffix='.pdb', delete=False
                                ) as f:
                                    f.write(pred_pdb)
                                    pred_pdb_path = f.name
                                    temp_files.append(pred_pdb_path)

                                with tempfile.NamedTemporaryFile(
                                    mode='w', suffix='.pdb', delete=False
                                ) as f:
                                    f.write(gt_pdb)
                                    gt_pdb_path = f.name
                                    temp_files.append(gt_pdb_path)

                                # Format tokens as string (first 20 tokens)
                                pred_tok_str = " ".join(
                                    str(t) for t in ex["pred_tokens"][:20]
                                )
                                if len(ex["pred_tokens"]) > 20:
                                    pred_tok_str += "..."
                                gt_tok_str = " ".join(
                                    str(t) for t in ex["gt_tokens"][:20]
                                )
                                if len(ex["gt_tokens"]) > 20:
                                    gt_tok_str += "..."

                                data.append([
                                    ex["type"],
                                    f"{ex['rmsd']:.2f}",
                                    f"{ex['token_accuracy']:.3f}",
                                    len(ex["aa_seq"]),
                                    ex["aa_seq"],
                                    pred_tok_str,
                                    gt_tok_str,
                                    wandb.Molecule(pred_pdb_path),
                                    wandb.Molecule(gt_pdb_path),
                                ])

                            table = wandb.Table(columns=columns, data=data)
                            wandb.log(
                                {f"rmsd_examples_step_{global_step}": table},
                                step=global_step
                            )
                            logger.info(f"Logged {len(examples)} example structures to wandb")
                        except Exception as e:
                            logger.warning(f"Failed to log examples to wandb: {e}")
                        finally:
                            # Clean up temp files after wandb has read them
                            for f in temp_files:
                                try:
                                    os.unlink(f)
                                except OSError:
                                    pass

                    model.train()

                # Save checkpoint
                if save_interval > 0 and global_step % save_interval == 0:
                    save_path = Path(output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {save_path}")

                if max_steps > 0 and global_step >= max_steps:
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # Save final model
    final_path = Path(output_dir) / "final"
    accelerator.save_state(final_path)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_path / "model")
        tokenizer.save_pretrained(final_path / "tokenizer")
        logger.info(f"Saved final model to {final_path}")
        wandb.finish()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Kanzi structure prediction model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="Base model from Hugging Face",
    )
    parser.add_argument(
        "--kanzi-checkpoint",
        type=str,
        default="checkpoints/cleaned_model.pt",
        help="Path to Kanzi model checkpoint",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/foldseek/afdb50/afdb50",
        help="Path to Foldseek database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/kanzi_predictor",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument(
        "--max-protein-length", type=int, default=400, help="Max protein length"
    )
    parser.add_argument(
        "--min-protein-length", type=int, default=100, help="Min protein length"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps (overrides warmup-ratio)")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval interval")
    parser.add_argument(
        "--rmsd-eval-interval", type=int, default=500, help="RMSD eval interval"
    )
    parser.add_argument(
        "--rmsd-eval-samples", type=int, default=25, help="Number of RMSD eval samples"
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch with minimal vocabulary (amino acids + Kanzi tokens only)",
    )
    parser.add_argument(
        "--use-contacts",
        action="store_true",
        help="Include ground truth contacts as input hints",
    )
    parser.add_argument(
        "--max-contacts",
        type=int,
        default=50,
        help="Maximum number of contacts to include as hints",
    )
    parser.add_argument(
        "--contact-prob",
        type=float,
        default=1.0,
        help="Probability of including each contact (0-1, for dropout)",
    )
    parser.add_argument(
        "--freeze-base-steps",
        type=int,
        default=0,
        help="Steps to freeze base model and only train embeddings (embedding warmup)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        kanzi_checkpoint=args.kanzi_checkpoint,
        db_path=args.db_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        max_protein_length=args.max_protein_length,
        min_protein_length=args.min_protein_length,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        use_flash_attn=not args.no_flash_attn,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        rmsd_eval_interval=args.rmsd_eval_interval,
        rmsd_eval_samples=args.rmsd_eval_samples,
        max_steps=args.max_steps,
        from_scratch=args.from_scratch,
        use_contacts=args.use_contacts,
        max_contacts=args.max_contacts,
        contact_prob=args.contact_prob,
        freeze_base_steps=args.freeze_base_steps,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
