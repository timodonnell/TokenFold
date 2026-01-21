# Training Runs

This file documents training runs for the structure prediction model.

---

## Run 16: Kanzi + Contacts + Fixed Resume (2026-01-21) 🔄 IN PROGRESS

### Wandb Link
**https://wandb.ai/timodonnell/tokenfold/runs/diafb1mk**

### Status
Resumed from checkpoint-56000 with fixed LR (2.5e-5), 500 warmup steps, no flash attention.

### Output Directory
`outputs/kanzi_20260121_202532/`

### Command
```bash
HF_TOKEN=<token> WANDB_PROJECT=tokenfold \
uv run python -m tokenfold.train_kanzi \
  --model-name meta-llama/Llama-3.2-1B \
  --resume-from outputs/kanzi_20260121_150343/checkpoint-56000 \
  --use-contacts \
  --learning-rate 2.5e-5 \
  --warmup-steps 500 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-protein-length 256 \
  --rmsd-eval-samples 50 \
  --rmsd-eval-interval 250 \
  --output-dir outputs/kanzi_20260121_202532 \
  --no-flash-attn
```

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | Llama-3.2-1B (pretrained) |
| Resume from | checkpoint-56000 |
| Learning rate | 2.5e-5 |
| Warmup steps | 500 |
| Flash attention | Disabled |

### Key Fix
- Load only model weights (not optimizer/scheduler) when resuming to allow LR changes
- Use `strict=False` for state_dict loading to handle tied weights (lm_head)

---

## Run 15: Kanzi + Contacts + LR Fix Attempts (2026-01-21) ❌ FAILED

### Wandb Links
- https://wandb.ai/timodonnell/tokenfold/runs/30uqz63y
- https://wandb.ai/timodonnell/tokenfold/runs/k4q4xo55

### Status
Multiple failed attempts to resume with correct LR. Flash attention issues.

### Notes
- Run 30uqz63y: Flash attention enabled but caused issues
- Run k4q4xo55: Another attempt
- Fixed by disabling flash attention

---

## Run 14: Kanzi + Contacts Resume Attempt (2026-01-21) ❌ FAILED

### Status
Multiple failed attempts to resume with higher LR. Issues with scheduler state being restored.

### Notes
- First attempt: LR stayed at ~7e-6 despite specifying 2.5e-4
- Problem: `accelerator.load_state()` restores optimizer/scheduler state
- Fixed by loading only model weights via safetensors directly

---

## Run 13: Kanzi + Contacts + Embedding Warmup (2026-01-21) 🔄 COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/tokenfold/runs/quc3m5d7**

### Status
Ran from step 49000 to 56000 before stopping to fix LR warmup.

### Output Directory
`outputs/kanzi_20260121_150343/`

### Command
```bash
HF_TOKEN=<token> WANDB_PROJECT=tokenfold \
uv run python -m tokenfold.train_kanzi \
  --model-name meta-llama/Llama-3.2-1B \
  --resume-from outputs/kanzi_20260120_222137/checkpoint-49000 \
  --use-contacts \
  --learning-rate 2.5e-5 \
  --batch-size 16 \
  --max-protein-length 256 \
  --rmsd-eval-samples 50 \
  --rmsd-eval-interval 250 \
  --no-flash-attn
```

### Notes
- LR warmup was taking forever (261k steps default)
- Added `--warmup-steps` option to fix this

---

## Run 12: Kanzi + Contacts + Natural Language Format (2026-01-20) ✅ COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/tokenfold/runs/ybmd4h7t**

### Status
Completed 49000 steps. Model generating numbers instead of Kanzi tokens early on.

### Output Directory
`outputs/kanzi_20260120_222137/`

### Command
```bash
HF_TOKEN=<token> WANDB_PROJECT=tokenfold \
uv run python -m tokenfold.train_kanzi \
  --model-name meta-llama/Llama-3.2-1B \
  --freeze-base-steps 500 \
  --use-contacts \
  --learning-rate 5e-5 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-protein-length 256 \
  --rmsd-eval-samples 50 \
  --rmsd-eval-interval 250 \
  --no-flash-attn
```

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | Llama-3.2-1B (pretrained) |
| Format | Natural language (Protein sequence:, Structure:, Contacts:) |
| System prompt | Document-style explanation |
| Freeze base steps | 500 (embedding warmup) |
| Use contacts | Yes |
| Max protein length | 256 |
| Batch size | 16 |
| Learning rate | 5e-5 |

### New Features Tested
1. **Natural language format** - `Protein sequence: M K T ... Contacts: 5-20 ... Structure: <K100> ...`
2. **Document-style system prompt** - Explains task to leverage pretrained knowledge
3. **Embedding warmup** - Freeze base model for first 500 steps
4. **Token diversity metrics** - Track unique tokens, entropy, most common token

### Observations
- Model initially generates `1 2 3 4 5...` instead of `<K100> <K200>...`
- This is expected early in training - model defaults to familiar patterns
- Token diversity metrics show collapse (`unique_tokens: 0-3`)

---

## Run 11: Kanzi Old Format (2026-01-19 to 2026-01-20) ✅ COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/tokenfold** (multiple runs)

### Status
Completed ~37000 steps with old special token format.

### Output Directory
`outputs/kanzi_predictor/`

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | Llama-3.2-1B or TinyLlama |
| Format | Special tokens (`<AA>`, `<SEP>`, `<KANZI>`, `<CONTACTS>`) |
| Use contacts | Yes |
| Contact dropout | `--contact-prob` option added |

### Notes
- Used old special token format before switching to natural language
- Added contact-guided training (Phase 1)
- Added example logging every 100 steps
- Fixed wandb logging issues

---

## Run 10: Kanzi Initial Experiments (2026-01-18 to 2026-01-19)

### Status
Various experiments with Kanzi token prediction.

### Key Developments
1. Created `train_kanzi.py` for Kanzi structure prediction
2. Added RMSD evaluation using Kabsch alignment
3. Added C-alpha coordinate decoding via Kanzi tokenizer
4. Implemented contact extraction from coordinates

---

## Run 9: TinyLlama Full Fine-tune (2026-01-17) ✅ COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/structure-search/runs/9c49x0fd**

### Status
Completed 1000 steps of full fine-tuning on TinyLlama 1.1B.

### Command
```bash
WANDB_API_KEY="<key>" WANDB_PROJECT="structure-search" \
uv run python -m structure_search.train \
    --mode tinyllama-full \
    --output-dir outputs/tinyllama-run2 \
    --max-steps 1000 \
    --no-flash-attn
```

### Results

| Metric | Initial (Step 0) | Final (Step 1000) |
|--------|------------------|-------------------|
| Eval Loss | 2.1456 | 1.3841 |
| Gen Token Accuracy | 4.6% | **23.2%** |
| Length Match Rate | 2% | **18%** |
| Valid Chars Rate | 12% | **100%** |

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | TinyLlama-1.1B-intermediate-step-1431k-3T |
| Mode | Full fine-tuning (no LoRA) |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Max steps | 1000 |
| GPUs | 1x H100 80GB |
| Duration | ~34 minutes |

### Notes
- Model learned to produce 100% valid 3Di characters
- Token accuracy improved from 4.6% → 23.2%
- Saved to `outputs/tinyllama-run2/final/`

---

## Run 8: TinyLlama Full Fine-tune (2026-01-17) ✅ COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/structure-search/runs/6uwnq1nt**

### Status
Completed 1000 steps of full fine-tuning on TinyLlama 1.1B.

### Results

| Metric | Initial (Step 0) | Final (Step 1000) |
|--------|------------------|-------------------|
| Eval Loss | 2.1457 | 1.3848 |
| Gen Token Accuracy | 4.6% | **22.2%** |
| Length Match Rate | 2% | 4% |
| Valid Chars Rate | 12% | **100%** |

### Configuration
Same as Run 9. Saved to `outputs/tinyllama-run/final/`.

### Notes
- First successful TinyLlama training run
- Duration: ~39 minutes

---

## Run 7: ProstT5 Baseline Only (2026-01-16) ✅ CURRENT

### Wandb Link
**(pending - check wandb for latest run)**

### Status
Training with simplified ProstT5 evaluation (baseline only, no model generation).

### Command
```bash
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    -m structure_search.train \
    --mode llama-8b-lora \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor_v18 \
    --batch-size 24 \
    --gradient-accumulation-steps 1 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500 \
    --prostt5-eval-interval 1000 \
    --prostt5-eval-samples 50
```

### Key Fix
**Removed model generation from ProstT5 evaluation**: The previous crash was caused by calling `model.generate()` on only rank 0, which triggered NCCL collective operations that other ranks weren't participating in. Now we only evaluate ProstT5 baseline vs ground truth during training. Full model evaluation will be done separately.

### Configuration
Same as Run 6, with simplified ProstT5 evaluation.

---

## Run 6: Multi-Model Support + ProstT5 Fix (2026-01-16) - CRASHED

### Wandb Link
**(check wandb for run)**

### Status
Crashed at step 1000 during ProstT5 comparison - NCCL timeout due to model.generate() on single rank.

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 500 | 1.55 | ~1.52 |
| 1000 | 1.53 | 1.52 |

### Crash Details
- NCCL timeout at step 1000 after eval completed
- Root cause: `model.generate()` triggers NCCL collective ops even when only called on rank 0
- Rank 0 had 320 more NCCL operations enqueued than other ranks
- Fixed in Run 7 by removing model generation from ProstT5 eval

### Features Added
1. **Multi-model support**: New `--mode` argument with presets:
   - `llama-8b-lora`: LoRA fine-tuning on Llama 3.1 8B (default)
   - `tinyllama-full`: Full fine-tuning on TinyLlama 1.1B

2. **ProstT5 comparison** (barrier fix was insufficient)

---

## Run 5: ProstT5 Comparison + Validity Metrics (2026-01-16) - CRASHED

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/yuaqtvnq**

### Status
Crashed at step 1000 during ProstT5 comparison due to NCCL timeout.

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 500 | 1.6210 | 1.5777 |
| 1000 | 1.4777 | 1.5272 |

### Crash Details
- NCCL collective timeout at step 1000 when ProstT5 comparison started
- Cause: Only main process ran ProstT5 eval while other processes continued and timed out waiting for collective operations
- Fixed in Run 6 by adding `accelerator.wait_for_everyone()` barrier

---

## Run 4: Stable Training with Eval Fix (2026-01-16) - STOPPED

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/kr4u1yod**

### Status
Stopped at step ~1420 to restart with ProstT5 eval enabled.

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 500 | 1.4827 | 1.5777 |
| 1000 | 1.4695 | 1.5221 |

### Command
```bash
export WANDB_API_KEY="<your-key>"
export WANDB_PROJECT="structure-prediction"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor_v15 \
    --batch-size 24 \
    --gradient-accumulation-steps 1 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 2000 \
    --eval-interval 500
```

### Key Fixes
1. **Distributed evaluation fix**: Fixed NCCL timeout during evaluation
   - Use fixed iteration count (50 steps) with iterator reset
   - Proper gathering of losses across all GPUs with `accelerator.gather()`
   - Call `accelerator.save_state()` on all processes

2. **Gradient checkpointing**: Reduces memory for larger batch sizes

---

## Run 3: Optimized Batch Size (2026-01-16) - CRASHED

### Notes
- Crashed during evaluation due to NCCL collective timeout
- Fixed in Run 4
- Optimized batch size from 4 to 48 per GPU after OOM testing
- No gradient accumulation needed with larger batch size

---

## Run 2: Fixed Tokenization (2026-01-16)

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/5lv9i3dc**

### Command
```bash
export WANDB_API_KEY="<your-key>"
export WANDB_PROJECT="structure-prediction"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor_v2 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500
```

### Fix Applied
**Space-separated tokenization** to ensure 1:1 alignment between amino acids and 3Di characters.

Before (broken):
```
<AA>MKTLKDLLK  →  tokens: ['MK', 'TL', 'K', 'DLL', 'K']  (merged)
```

After (fixed):
```
<AA> M K T L K D L L K  →  tokens: ['<AA>', 'ĠM', 'ĠK', 'ĠT', 'ĠL', ...]  (1:1)
```

### Configuration
Same as Run 1, but with corrected tokenization.

---

## Run 1: Initial Training (2026-01-16) ❌ CANCELLED - Tokenization Bug

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/nm9f8ymf**

### Command
```bash
export WANDB_API_KEY="<your-key>"
export WANDB_PROJECT="structure-prediction"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.1-8B` |
| Dataset | afdb50 (66.7M proteins) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Batch size (per GPU) | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 256 (4 × 8 GPUs × 8 accum) |
| Learning rate | 2e-4 |
| Warmup ratio | 3% |
| Max sequence length | 1024 tokens |
| Precision | bfloat16 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight decay | 0.01 |

### Hardware
- 8× NVIDIA H100 80GB HBM3
- DeepSpeed ZeRO-3 (no offloading)
- Flash Attention 2

### Description

First training run to fine-tune Llama 3.1 8B for sequence-to-structure prediction. The model learns to translate amino acid sequences to Foldseek's 3Di structural alphabet.

**Training format:**
```
<AA>MKTLKDLLKEKQNLIK...<SEP><3Di>DDPLVVVLVVLVVVLVV...
```

Loss is computed only on the structure tokens (after `<SEP>`), so the model learns to predict 3Di given the amino acid sequence.

### Notes
- **CANCELLED** after ~50 steps due to tokenization bug
- BPE tokenizer was merging amino acids (e.g., "MK", "DLL") breaking 1:1 alignment
- Loss was ~30 which is abnormally high
- See Run 2 for the corrected version
