# JUMP

Minimal code bundle for the JUMP attack on diffusion language models.

Overview:
- [overview.pdf](./figures/overview.pdf)
- [![JUMP overview](./figures/overview.png)](./figures/overview.pdf)

This folder contains only the core code used to:

- train a PRISM selector on top of an LLaDA backbone,
- compute clean-token selector signals,
- run the JUMP multi-mask membership-inference attack,
- share the small model/loading utilities those scripts depend on.

It intentionally excludes paper figures, Slurm launchers, unrelated ablations, and defense-side analysis code.

## Layout

- `eval_jump.py`
  - Short CLI entrypoint for attack evaluation.

- `train_prism.py`
  - Short CLI entrypoint for selector training.

- `core/prism_utils.py`
  - PRISM model definition and checkpoint utilities.
  - Includes the LoRA-wrapped selector head and checkpoint loading/saving helpers.

- `core/attack/`
  - Attack-time modules.
  - `args.py`: attack CLI arguments
  - `score_utils.py`: clipping and robust aggregation helpers
  - `masking.py`: joint multi-mask query construction and score extraction
  - `hierarchy.py`: grouped/hierarchical refinement helpers
  - `features.py`: feature construction from target/reference scores
  - `runner.py`: main attack pipeline

- `core/selection/`
  - Clean-sequence selector analysis modules.
  - `signals.py`: PRISM clean forward signals and target one-hole readout
  - `features.py`: selector-based feature construction
  - `runner.py`: selector diagnostic pipeline

- `core/training/`
  - PRISM selector training modules.
  - `args.py`: training CLI arguments
  - `data.py`: local dataset loading and batching
  - `corruption.py`: corrupted-input construction and supervision sampling
  - `evaluation.py`: validation logic
  - `runner.py`: main training loop

- `core/shared/`
  - Shared lightweight utilities:
    - metrics
    - manifest loading
    - model loading

- `figures/`
  - Overview assets used in the README preview.

## Environment

The code expects the same Python environment used in the original repository:

- `torch`
- `transformers`
- `numpy`
- `scikit-learn`
- `accelerate` for `train_prism.py`
- See [requirements.txt](./requirements.txt) for the minimal pip package list.

LLaDA checkpoints and tokenizers are not bundled here.

## Attack workflow

### 1. Train or load a PRISM selector

If you already have a selector checkpoint, you can skip training.

Example:

```bash
python jump/train_prism.py \
  --base-model GSAI-ML/LLaDA-8B-Base \
  --tokenizer-path external/llada_8b_base_tokenizer \
  --train-text-path path/to/train.jsonl \
  --out-dir outputs/prism_c4 \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 2 \
  --max-length 256 \
  --lr 1e-4 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0.1
```

### 2. Run the JUMP attack

Example:

```bash
python jump/eval_jump.py \
  --model_path GSAI-ML/LLaDA-8B-Base \
  --tokenizer_path external/llada_8b_base_tokenizer \
  --target_model_path path/to/target/checkpoint-80 \
  --prism_checkpoint path/to/prism_checkpoint.pt \
  --target_backbone path/to/target/checkpoint-80 \
  --reference_backbone none \
  --eval_data path/to/eval_manifest.json \
  --output_dir results/jump_eval \
  --selected_k 64 \
  --selection_mode quality_bot \
  --batch_size 8 \
  --bf16 \
  --fixed_huber_clip_values 0.4054651081081644
```

Typical retained main setting:

- selector: `quality_bot`
- selected-token budget: `K = 64`
- aggregation: clipped mean target/reference gap
- clipping threshold: `c = log(1.5)`

## Inputs

The evaluator expects an evaluation manifest in the repository's existing `sample_groups` format, with member and non-member examples already defined.

## Outputs

The attack evaluator writes:

- `summary.json`
- per-feature metrics
- optional per-sample diagnostics when enabled

under the directory passed to `--output_dir`.

## Notes

- The top-level files are intentionally short wrappers.
- The implementation is split into `attack/`, `selection/`, `training/`, and `shared/` modules so the code is easier to browse.
- These scripts are lightly cleaned copies of the repository's main attack code.
- Paths to checkpoints, manifests, and tokenizers still need to be supplied by the user.
