# FuRA: Full-Rank Adaptation of LLMs via Lossless Block Tensor-Train Factorization

This repository contains the reference implementation for **FuRA**, the
parameter-efficient fine-tuning method introduced in our submission. It
provides everything needed to reproduce the four main empirical tracks of the
paper:

1. **Commonsense reasoning SFT** (Llama-3-8B).
2. **Math reasoning RL** (GRPO on Qwen3-1.7B).
3. **Visual instruction tuning** (LLaVA-v1.5-7B).
4. **QFuRA — quantized FuRA** for commonsense reasoning.

---

## 1. What is FuRA?

LoRA-style adapters write the update as a low-rank product `BA`. They are
parameter efficient, but they constrain the update to lie inside a *low-rank*
subspace — a constraint that limits expressivity and forces practitioners to
guess at a rank. Full fine-tuning, on the other hand, has full flexibility but
no inductive bias toward the pre-trained feature space, and on small / noisy
fine-tuning sets it tends to over-fit.

FuRA reparameterizes each weight matrix `W ∈ R^{m×n}` with a **lossless block
tensor-train factorization**: `W = L · S · R`, where `L`, `R` are blocks of an
orthogonal Kronecker-structured factorization initialized from the SVD of `W`,
and `S` carries the singular values. We freeze the larger core (so the on-device
parameter count stays comparable to LoRA at rank ≈ 64), and train only the
smaller core together with `S`.

This gives us two simultaneous wins:

- **Full-rank update** — there is no rank cap; the effective update can change
  every singular direction.
- **Spectral regularization** — because the update is parameterized through the
  pre-trained subspace, gradient steps are pre-conditioned by the pre-trained
  spectrum, which provides a strong inductive bias toward the directions that
  matter and damps drift in irrelevant directions.

The default configuration is `rank=full`, `decomp_mode=output_one_block`,
`train_position=small`, `s_merged_to=keep_trainable`. These defaults appear in
the launch scripts and are the configuration evaluated in the paper.

---

## 2. Repository layout

```
FuRA/
├── fura/                    # The shared FuRA layer implementations.
│   ├── btt_layer.py         #   BTTLayer + Linear→BTT conversion + QBTT (NF4)
│   ├── svd_layer.py         #   SVDLayer (full-rank SVD reparameterization)
│   └── math_utils.py        #   Math RL reward helpers
├── commonsense/             # Track 1: LIFT-style commonsense SFT (Llama-3-8B)
├── rl/                      # Track 2: GRPO RL on math (Qwen3-1.7B)
├── vlm/                     # Track 3: LLaVA-v1.5 visual instruction tuning
├── qfura_commonsense/       # Track 4: QFuRA commonsense (NF4-quantized FuRA)
│   └── qdora_baseline/      #   QDoRA baseline (separate environment)
├── pyproject.toml           # Shared env for tracks 1, 2, 4
└── README.md
```

All four tracks share `fura/` as the source of truth for the BlockTT and SVD
adapters; each track only adds the trainer / data-loader / launcher glue
specific to its setting.

---

## 3. User-population directories (TODO before running)

The repository ships **without** large datasets or pretrained weights. Each
of the directories listed below contains a `TODO_POPULATE.md` marker. Before
you run any reproduction script, fetch the corresponding artifact from its
public source and drop it into the indicated path.

| Path                                                                     | What goes here                          | Source                                                               |
| ------------------------------------------------------------------------ | --------------------------------------- | -------------------------------------------------------------------- |
| `commonsense/LLM-Adapters/ft-training_set/commonsense_170k.json`       | Commonsense SFT mixture (~115 MB)       | LLM-Adapters public repo (see `commonsense/LLM-Adapters/SETUP.md`) |
| `commonsense/LLM-Adapters/dataset/<task>/test.json`                    | 8-task commonsense eval suite (~115 MB) | LLM-Adapters public repo                                             |
| `qfura_commonsense/LLM-Adapters/ft-training_set/commonsense_170k.json` | Same as above (re-used by QFuRA track)  | LLM-Adapters public repo                                             |
| `qfura_commonsense/LLM-Adapters/dataset/<task>/test.json`              | Same as above                           | LLM-Adapters public repo                                             |
| `vlm/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin`              | LLaVA-v1.5 pretrained projector         | LLaVA public repo (see `vlm/README.md`)                            |
| `vlm/playground/data/llava_v1_5_mix665k.json`                          | LLaVA instruction-tuning mixture        | LLaVA public repo                                                    |
| `vlm/playground/data/{coco,gqa,ocr_vqa,textvqa,vg}/...`                | Image folders for the mixture           | COCO / GQA / OCR-VQA / TextVQA / VG public sources                   |

The `rl/` track loads its training and evaluation data directly from
HuggingFace Hub at runtime (`qwedsacf/competition_math`,
`HuggingFaceH4/MATH-500`, AIME, AMC23) — no manual download needed.

---

## 4. Environments

The repository uses **three** Python environments — they're separated because
LLaVA pins an older PyTorch / transformers stack and QDoRA expects a slightly
different PEFT.

| Environment                                                             | Used by                                           | Setup                                                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------- |
| **Root** (`pyproject.toml`)                                     | `commonsense/`, `rl/`, `qfura_commonsense/` | `pip install -e .` (or `uv sync`)                                      |
| **VLM** (`vlm/requirements.txt`)                                | `vlm/`                                          | `cd vlm && pip install -r requirements.txt && pip install -e peft`       |
| **QDoRA** (`qfura_commonsense/qdora_baseline/requirements.txt`) | `qfura_commonsense/qdora_baseline/`             | `cd qfura_commonsense/qdora_baseline && pip install -r requirements.txt` |

Sample setup with `uv`:

```bash
# Root environment (commonsense + rl + qfura)
uv venv .venv && source .venv/bin/activate
uv pip install -e .
# To also enable the RL track's vLLM-based rollouts:
uv pip install -e ".[rl]"
```

For non-uv users, `pip install -e .` in a fresh venv works too.

---

## 5. Reproducing paper results

Each track has a top-level `reproduce.sh` that picks sane defaults from the
paper. Replace the variant argument to switch baselines.

### 5.1 Commonsense reasoning (Track 1)

```bash
cd commonsense
# Populate commonsense/LLM-Adapters/ first (see Section 3 / SETUP.md)
bash reproduce.sh fura          # FuRA (paper main result)
bash reproduce.sh full          # Full fine-tuning baseline
bash reproduce.sh lora          # LoRA baseline (rank 32)
bash reproduce.sh svd           # SVD reparameterization baseline
```

Hardware: 1× 80GB GPU is sufficient for Llama-3-8B with the default
batch settings (per-device BS 8, grad-acc 2 → effective BS 16). The training
script saves the **last-step** checkpoint and runs the eight-task LLM-Adapters
evaluation suite (BoolQ / PIQA / SIQA / HellaSwag / WinoGrande / ARC-Easy /
ARC-Challenge / OBQA) automatically.

### 5.2 Math reasoning RL (Track 2)

```bash
cd rl
bash reproduce.sh fura          # FuRA (paper main result)
bash reproduce.sh full          # Full FT baseline
bash reproduce.sh lora          # LoRA r=32 baseline
bash reproduce.sh svd           # SVD reparameterization baseline
```

This trains Qwen3-1.7B with GRPO on the math reasoning prompt template at
`rl/boxed.prompt`. By default it uses an in-process vLLM rollout. To use a
separate vLLM server (faster for LoRA), in another terminal run

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B --enable-lora --max-lora-rank 64
```

For the larger Qwen3-7B configuration described in the paper, see `run_rl_7B.sh`.

### 5.3 Visual instruction tuning (Track 3)

```bash
cd vlm
# Activate the dedicated VLM env, then:
bash reproduce.sh fura          # FuRA on LLaVA-v1.5-7B
bash reproduce.sh dora          # DoRA baseline
```

Prereqs (see `vlm/README.md` and Section 3 above):

- Pretrained MLP projector at `./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin`.
- Instruction-tuning data `./playground/data/llava_v1_5_mix665k.json` and
  associated images at `./playground/data/`.

### 5.4 QFuRA commonsense (Track 4)

```bash
cd qfura_commonsense
# Populate qfura_commonsense/LLM-Adapters/ first
bash reproduce.sh qfura         # QFuRA (paper main result)
bash reproduce.sh qlora         # QLoRA baseline

# QDoRA needs its own env:
cd qdora_baseline
python -m venv .venv-qdora && source .venv-qdora/bin/activate
pip install -r requirements.txt
bash reproduce.sh
```

QFuRA quantizes the frozen large core to NF4 (via `bitsandbytes`) while keeping
the trainable small core and `S` in bf16. The default `output_one_block` /
`keep_trainable` configuration was selected from a quantization-error sweep
documented in the paper.

---

## 6. Customizing FuRA in your own training script

The shared `fura` package exposes a small API:

```python
from fura import (
    convert_linear_to_btt,
    configure_blocktt_trainability,
)

# 1) After loading a HuggingFace causal-LM, convert its linear layers in-place.
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
convert_linear_to_btt(
    model,
    target_modules=target_modules,
    decomp_mode="output_one_block",   # FuRA default
    rank="full",
    s_merged_to="keep_trainable",
)

# 2) Freeze everything except the small core + S.
configure_blocktt_trainability(model, train_position="small")

# 3) Train as usual.
```

The four config knobs to remember are `decomp_mode`, `train_position`,
`s_merged_to`, and `rank`. Their semantics, gradient analysis, and design
rationale live in §3–§4 of the paper.

---

## 7. Notes on this code release

- **Source of truth for FuRA layer code is `fura/`.** Each track imports from
  there; do not edit a copy inside a track directory.
- **Anonymity.** Author names, institutional emails, WandB usernames, personal
  paths, and personal HuggingFace Hub IDs have been removed from this release.
  The third-party `LICENSE` files preserved inside `vlm/`, `vlm/peft/`, and
  `commonsense/LLM-Adapters/` carry the **upstream** authors' attribution as
  required by Apache-2.0 / MIT. They identify the libraries we built on, not
  the authors of this submission.
- **Datasets.** The LLM-Adapters commonsense data and the LLaVA-v1.5 mixture
  are not vendored in this repository (~250 MB and several GB respectively).
  Follow the `SETUP.md` files and Section 3 above.

---

## 8. License

The new code introduced by this release is provided under the Apache-2.0
license. Vendored third-party components (LLaVA, the DoRA-patched PEFT fork,
the LLM-Adapters evaluation harness) retain the licenses shipped in their
upstream repositories — see the `LICENSE` files inside each subdirectory.
