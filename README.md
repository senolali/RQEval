# RQEval: [Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework](https://arxiv.org/abs/2605.24661)

[![PyPI](https://img.shields.io/pypi/v/rqeval)](https://pypi.org/project/rqeval/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.24661-b31b1b.svg)](https://arxiv.org/abs/2605.24661)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **RQEval** (Reasoning Quality Evaluation) -- a config-driven, multi-dimensional framework for evaluating reasoning quality in Large Language Models, beyond simple answer correctness.

**6 metrics . 7 models (5 API + 2 local) . 4 benchmark datasets . CLI + no-code web interface . no code changes needed to add models or datasets**

RQEval measures six theoretically grounded behavioral dimensions -- Correctness (CQ), Consistency (CS), Robustness (RS), Local Logical Coherence (LS), Efficiency (ES), and Stability (SS) -- and organizes them into deployment-aware aggregation scores, so you can pick the right model for the right context rather than just the highest leaderboard score.

*(Paper under revision at MDPI Big Data and Cognitive Computing; this repository tracks the corrected, camera-ready protocol described in the Experimental Setup / Reproducibility note of the manuscript.)*

---

## Table of Contents

- [Overview](#overview)
- [Metrics](#metrics)
- [Models](#models)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Interface (No Code Required)](#web-interface-no-code-required)
- [Custom Evaluation: Your Own Dataset & Weights](#custom-evaluation-your-own-dataset--weights)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Configuration](#configuration)
- [Checkpoint / Resume & LaTeX Export](#checkpoint--resume--latex-export)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Adding New Models](#adding-new-models)
- [Adding New Datasets](#adding-new-datasets)
- [Known Issues & Platform Notes](#known-issues--platform-notes)
- [Citation](#citation)

---

## Overview

Standard LLM evaluation asks: *"Is the answer correct?"*

This framework asks: *"How well does the model reason?"*

It evaluates models across **6 complementary dimensions** of reasoning quality, producing a composite score that captures correctness, behavioral stability, robustness, logical integrity, and efficiency simultaneously.

```
Q = f(CQ, CS, RS, LS, ES, SS)
```

The framework is fully **config-driven** — models, datasets, metrics, and aggregation weights are all controlled from a single YAML file. No code changes are needed for common use cases.

---

## Metrics

| Symbol | Name              | What It Measures                                       |
| ------ | ----------------- | ------------------------------------------------------ |
| **CQ** | Correctness       | Fraction of correct final answers                      |
| **CS** | Consistency       | Same answer across K independent runs?                 |
| **RS** | Robustness        | Same answer on semantically equivalent rephrases?      |
| **LS** | Local Logical Coherence | No contradictions between consecutive reasoning steps? |
| **ES** | Efficiency        | Correct **and** concise? (harmonic mean of CQ and inverse normalized token count) |
| **SS** | Stability         | Same reasoning *process* across K runs? (BERTScore over traces) |

Formal definitions of all six metrics are given in Section 3 of the paper.

### Key design decisions

**CQ — Multi-strategy matching pipeline:** Raw model outputs are often verbose (e.g. *"John has 8 apples."* instead of *"8"*). The correctness metric applies 7 sequential matching strategies before marking an answer wrong: exact match → normalized → number extraction → yes/no extraction → A/B/C/D extraction → substring match → numeric tolerance. This prevents local models from being penalized purely for output format.

**RS — Conditioned on correctness:** Robustness is only counted for questions the model originally answered correctly. A model that gets everything wrong would trivially get RS = 1.0 otherwise.

**LS — NLI-based contradiction detection:** Uses `cross-encoder/nli-deberta-v3-small` to detect contradictions between consecutive reasoning steps. Single-sentence responses receive LS = 1.0 by convention (a single atomic step admits no internal contradiction). Falls back gracefully to LS = 1.0 if the NLI model is unavailable.

**ES — Harmonic mean:** Prevents rewarding short-but-wrong or long-but-correct responses equally. Both correctness and conciseness must be high for ES to be high.

**SS — BERTScore similarity:** Measures semantic similarity between reasoning traces across runs, not just whether the final answer matches. Falls back to Jaccard similarity if `bert-score` is not installed.

**CS/SS and temperature:** Running with `deterministic: true` (temperature = 0) produces CS = SS = 1.0 for all models — this is a mathematical artifact, not a real measurement. Set `temperature: 0.7` per model in config to get meaningful CS/SS scores. All paper results were obtained at temperature = 0.7.

### Aggregation strategies

Seven built-in weighting schemes are computed for every experiment. All appear as separate columns in the Excel output.

| Strategy            | CQ   | CS   | RS   | LS   | ES   | SS   | Use case                         |
| ------------------- | ---- | ---- | ---- | ---- | ---- | ---- | -------------------------------- |
| Balanced            | 1/6  | 1/6  | 1/6  | 1/6  | 1/6  | 1/6  | General comparison               |
| Safety Priority     | 0.30 | 0.05 | 0.30 | 0.25 | 0.05 | 0.05 | High-stakes deployment           |
| Accuracy Priority   | 0.50 | 0.10 | 0.15 | 0.15 | 0.05 | 0.05 | Accuracy-critical tasks          |
| Efficiency Priority | 0.20 | 0.15 | 0.15 | 0.10 | 0.30 | 0.10 | Resource-constrained deployment  |
| Medical Triage      | 0.40 | 0.05 | 0.30 | 0.20 | 0.03 | 0.02 | Clinical decision support        |
| Legal/Compliance    | 0.15 | 0.25 | 0.20 | 0.35 | 0.03 | 0.02 | Audit-sensitive applications     |
| Edge Device/IoT     | 0.30 | 0.03 | 0.10 | 0.05 | 0.50 | 0.02 | Resource-limited edge deployment |

These weight vectors are theoretically motivated illustrative defaults; practitioners should calibrate them against their own operational requirements. Custom strategies can be added directly in `config.yaml` — no code changes needed.

---

## Models

The seven models evaluated in the paper:

| # | Model               | Provider     | Type                    | Parameters | Access            |
| --- | ------------------- | ------------ | ----------------------- | ---------- | ----------------- |
| 1 | GPT-4o-mini         | OpenAI       | API                     | —          | OpenAI API        |
| 2 | Claude Haiku 4.5    | Anthropic    | API                     | —          | Anthropic API     |
| 3 | DeepSeek-V3         | DeepSeek AI  | API                     | —          | DeepSeek API      |
| 4 | Gemini 2.5 Flash    | Google       | API                     | —          | Google API        |
| 5 | LLaMA-3-70B         | Meta         | API (OpenAI-compatible) | 70B        | OpenRouter        |
| 6 | Qwen2.5-1.5B-Instruct | Alibaba    | Local (HF)              | 1.5B       | HuggingFace, float16 |
| 7 | Phi-2               | Microsoft    | Local (HF)              | 2.7B       | HuggingFace, float16 |

The framework additionally supports **any OpenAI-compatible endpoint** (e.g., Groq) and **any HuggingFace causal LM** (e.g., Mistral-7B-Instruct-v0.3, LLaMA-3-8B-Instruct with 4-bit quantization) via config only — see [Adding New Models](#adding-new-models). These additional models are supported by the framework but were not part of the paper's evaluation.

Local models are **loaded one at a time** and released from RAM before the next model loads — allowing evaluation on machines without enough RAM to hold all models simultaneously. HuggingFace models are **downloaded automatically** on first run and cached in `~/.cache/huggingface/`.

---

## Datasets

The 975-item evaluation suite used in the paper:

| Dataset    | Type                    | Size (paper) | Answer format | Source             |
| ---------- | ----------------------- | ------------ | ------------- | ------------------ |
| GSM8K      | Math word problems      | 250          | Numerical     | `openai/gsm8k`     |
| MMLU       | 9 reasoning subjects    | 225          | A / B / C / D | `cais/mmlu`        |
| StrategyQA | Commonsense reasoning   | 250          | Yes / No      | `ChilleD/StrategyQA` (`test`) |
| Synthetic  | Built-in generator      | 250          | Mixed         | This repository    |

**MMLU subjects (9):** logical fallacies, formal logic, abstract algebra, elementary mathematics, high school mathematics, college mathematics, high school statistics, conceptual physics, and philosophy. The paper configuration samples 25 items from each subject, yielding 225 items.

**Synthetic dataset (250 items):** 100 arithmetic word problems with numerical variation, 75 distinct adversarial instances embedding deliberate logical contradictions, and 75 distinct robustness probes with three surface-level perturbations. Sizes are configurable; the values above reproduce the paper's N=975 and P=3 protocol.

All sampling uses a fixed random seed (`seed: 42`) for reproducibility. Custom JSON datasets can also be added — see [Adding New Datasets](#adding-new-datasets).

---

## Installation

### Option A — Install from PyPI (recommended for using the framework)

```bash
pip install rqeval
```

Then scaffold a ready-to-run project in any directory:

```bash
mkdir my-rqeval && cd my-rqeval
rqeval setup                    # copies main.py, app.py, config/ into the current directory
rqeval --config config/config_test.yaml   # quick smoke test, no API keys needed
```

CLI commands:

| Command | Description |
| ------- | ----------- |
| `rqeval setup` | Set up project files in the current directory (`--dir` for a different target) |
| `rqeval --config <file>` | Run an evaluation with the given YAML config |
| `rqeval --version` | Show installed version |

### Option B — Install from source (recommended for reproducing the paper / development)

#### Prerequisites

- Python 3.11 (recommended)
- Miniconda or Anaconda

The robustness pipeline uses `nltk`, `spacy`, `lemminflect`, `tiktoken`,
`transformers`, and `sentencepiece` (all declared in the package metadata).
After installing the package dependencies, download these linguistic
resources once:

```bash
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_md
```

> ⚠️ **PyTorch must be installed first and separately** — platform-specific instructions below. Do not run `pip install -r requirements.txt` before PyTorch is installed.

#### Windows + NVIDIA GPU (tested & recommended)

**Tested configuration:** GTX 1650 4 GB · CUDA 12.1 · Python 3.11 · PyTorch 2.4.0 · bitsandbytes 0.44.0

> ⚠️ Use `conda install` for PyTorch on Windows — **not** `pip install torch --index-url`. The pip CUDA wheels cause `fbgemm.dll` or `cusparse64_11.dll` errors on many Windows systems. Conda resolves all DLL dependencies automatically.

```bash
# Step 1 — Create environment
conda create -n llm_eval_gpu python=3.11 -y
conda activate llm_eval_gpu

# Step 2 — Install PyTorch via conda (CUDA 12.1)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Step 3 — Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Step 4 — Install bitsandbytes (pinned version)
pip install bitsandbytes==0.44.0

# Step 5 — Install remaining dependencies
pip install -r requirements.txt
pip install transformers -U
```

#### Windows CPU-only

```bash
conda create -n llm_eval python=3.11 -y
conda activate llm_eval

# PyTorch CPU wheel (max 2.3.x — 2.4+ causes fbgemm.dll errors on Windows CPU)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install "transformers==4.45.2"
pip install -r requirements.txt
```

#### Linux / macOS

```bash
conda create -n llm_eval_gpu python=3.11 -y
conda activate llm_eval_gpu

# GPU (CUDA 12.1 — tested):
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes==0.44.0

# CPU:
pip install torch

pip install -r requirements.txt
pip install transformers -U
```

---

## Quick Start

### 1. Set API keys

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY     = "sk-..."
$env:ANTHROPIC_API_KEY  = "sk-ant-..."
$env:GOOGLE_API_KEY     = "AIza..."
$env:DEEPSEEK_API_KEY   = "sk-..."
$env:OPENROUTER_API_KEY = "sk-or-..."
```

**Linux / macOS:**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

> Models with missing API keys are **automatically skipped** — you don't need all keys to run the framework.

### 2. Run full evaluation

```bash
python main.py --config config/config.yaml
```

Or specify a custom config:

```bash
python main.py --config config/my_experiment.yaml
```

For a quick smoke test without API keys (~5 min):

```bash
python main.py --config config/config_test.yaml
```

---

## Web Interface (No Code Required)

For users who prefer a graphical interface — researchers, clinicians, or domain experts without a coding background — the framework includes a **Streamlit** web app. Everything is point-and-click: no YAML, no terminal commands, no code.

### Launch

```bash
pip install streamlit
cd my-rqeval          # your project directory created by 'rqeval setup'
streamlit run app.py
```

A browser window opens automatically.

### What you can do in the browser

**Step 1 — Dataset**
- Upload your own dataset (**JSON or CSV**), with a live preview tab, or
- Pick a built-in benchmark (GSM8K, StrategyQA, MMLU, synthetic) and set the number of items with a slider

> ⚠️ CSV uploads map only `question` and `answer` columns — the RS (robustness) metric requires `perturbations` and is skipped for CSV datasets. Use the JSON format (see [Custom Evaluation](#custom-evaluation-your-own-dataset--weights)) to evaluate all six dimensions.

**Step 2 — Models**
- Enable/disable any model with a checkbox
- Add new models with **+ Add New Model** — choose the provider (OpenAI, Anthropic, Gemini, DeepSeek, local HuggingFace, mock), enter the model ID and API key directly in the browser
- Edit temperature, max tokens, device, quantization, and base URL per model
- A **Mock Model** is available for trying the interface without any API key

**Step 3 — Aggregation Strategy**
- Choose a preset (Balanced, Safety Priority, Clinical/Medical, Legal/Compliance, Accuracy Priority, Efficiency Priority, Edge Device/IoT), or
- Build a **Custom** strategy with one slider per dimension (weights auto-normalized to 1.0)

**Step 4 — Run**
- Click **Run Evaluation** and watch live progress with streaming logs
- Inspect the generated YAML in the **Config Preview** tab
- View results as per-model score cards and full tables (raw metrics + aggregated scores)
- Download the complete Excel report with one click

### Example: a clinician comparing models on their own cases

1. `streamlit run app.py`
2. Upload a JSON file of clinical questions and reference answers
3. Enable GPT-4o-mini and Claude, paste API keys
4. Select the **Clinical/Medical** preset
5. Click **Run Evaluation** and download the Excel report

---

## Custom Evaluation: Your Own Dataset & Weights

### Step 1 — Prepare your dataset

Create a JSON file (e.g. `my_dataset.json`):

```json
[
  {
    "id": "q001",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "type": "reasoning",
    "perturbations": [
      "Name the capital city of France.",
      "Which city serves as France's capital?",
      "What city is the capital of France?"
    ]
  }
]
```

> **Rules:**
> - `answer` must be a string (matched by the CQ pipeline)
> - `perturbations` — rephrased versions of the same question, used for RS (omit to skip RS for that item)
> - `type` — any label you choose, used for grouping in output

### Step 2 — Define your custom weights

Add a strategy to your config (auto-normalized if weights don't sum to 1.0):

```yaml
aggregation:
  strategies:
    my_strategy:
      correctness:       0.50
      robustness:        0.30
      logical_coherence: 0.20
      consistency:       0.00
      efficiency:        0.00
      stability:         0.00
```

### Step 3 — Run

```bash
rqeval --config config/config_custom.yaml
```

Or do all of the above with zero code in the [web interface](#web-interface-no-code-required).

---

## Reproducing the Paper Results

The configuration below reproduces the experimental setup reported in the paper (975 items, 7 models, temperature = 0.7, `max_new_tokens` = 256 unless noted otherwise, seed = 42). This is exactly what `config/config.yaml` already contains, so in most cases you can simply run:

```bash
rqeval --config config/config.yaml
```

The relevant settings, for reference:

```yaml
experiment:
  name: "paper_reproduction"
  seed: 42
  deterministic: false       # per-model temperature is used instead
  output_dir: "outputs"

metrics:
  consistency_runs: 3          # K = 3
  robustness_perturbations: 3  # P = 3
  stability_runs: 3
  nli_model: "cross-encoder/nli-deberta-v3-small"
  bertscore_model: "distilbert-base-uncased"

datasets:
  - { name: "gsm8k",      params: { num_samples: 250 } }
  - { name: "mmlu",       params: { num_samples: 225 } }   # 25 x 9 subjects
  - { name: "strategyqa", params: { num_samples: 250 } }
  - { name: "synthetic",  params: { num_reasoning: 100, num_adversarial: 75, num_robustness: 75 } }
```

Every model entry must set `temperature: 0.7` and `max_new_tokens: 256` (API models: `max_tokens: 256`) — **with one exception**: Gemini-2.5-Flash needs `max_tokens: 1024`. Its internal "thinking" tokens are drawn from the same budget as the visible response, and at 256 tokens generation was systematically truncated before an answer was produced (see the paper's Experimental Setup / Reproducibility note). This does not bias ES, which is computed via min–max normalization over each model's own observed token counts, not an absolute scale.

> ⚠️ **`max_new_tokens` matters for LS and SS.** Final-answer extraction (CQ) works even with `max_new_tokens: 64`, but LS and SS are computed over the full reasoning trace. Truncating generation below 256 tokens shortens or removes traces, inflating LS via the single-step convention and distorting SS. **Use 256 (1024 for Gemini) to reproduce the paper.** Lower values are acceptable only for quick correctness-oriented smoke tests.

**Runtime note for local models:** Each item requires `1 + consistency_runs + robustness_perturbations` inference calls (7 with defaults). At ~7–8 s/call on a GTX 1650 (float16), 975 items take approximately **12–15 hours per local model**. Reducing to `consistency_runs: 2` and `robustness_perturbations: 2` brings this to ~8–10 hours, at the cost of deviating from the paper setup.

### Optional: reproducing via the ASU CreateAI gateway

Four of the seven models in the paper (GPT-4o-mini, Claude-Haiku-4.5, Gemini-2.5-Flash, LLaMA-3.1-70B) were originally queried through Arizona State University's institutional CreateAI gateway rather than each provider's public API. `config/config.yaml` reproduces the same panel via public APIs and your own keys, which is what most users should use. If your institution provides CreateAI access and you want to reproduce that exact routing, use `config/config_asu_createai.yaml` instead:

```bash
rqeval --config config/config_asu_createai.yaml
```

This requires `ASU_CREATEAI_TOKEN` (see `.env.example`). Rate limits (750k tokens/min, service token) are handled by proactive pacing (`request_delay`) plus exponential backoff on HTTP 429. Search/RAG, history, and prompt enhancement are disabled per request so results match direct API calls. DeepSeek and the two local models are queried the same way as in `config.yaml`.

---

## Configuration

Everything is controlled from a single YAML file. The default is `config/config.yaml`.

### Experiment settings

```yaml
experiment:
  name: "my_experiment"     # Used as prefix for output folder name
  seed: 42                  # Random seed for reproducibility
  deterministic: true       # true = greedy decoding (temperature=0)
  output_dir: "outputs"     # Where results are saved

max_workers: 1              # Always set to 1 for local models — parallel loading
                            # causes meta tensor errors and CUDA OOM
```

### Adding an API model

```yaml
models:
  - name: "GPT-4o"          # Display name (appears in Excel / radar chart)
    type: "openai"           # openai | anthropic | gemini | deepseek | local | mock
    params:
      model_id: "gpt-4o"
      api_key_env: "OPENAI_API_KEY"   # Environment variable name
      max_tokens: 256
      temperature: 0.7       # Optional — overrides deterministic setting for CS/SS
      max_retries: 3
      timeout: 60
```

### Adding a local HuggingFace model

```yaml
models:
  - name: "Qwen2.5-1.5B"
    type: "local"
    params:
      model_id: "Qwen/Qwen2.5-1.5B-Instruct"
      device: "cuda"
      use_4bit: true                   # Attempts 4-bit; falls back to float16 if unsupported
      max_new_tokens: 256              # Use 256 for full-trace metrics (LS/SS); see note above
      temperature: 0.7
```

**RAM guide for local models:**

| Model size | `use_4bit` | VRAM needed                                 |
| ---------- | ---------- | ------------------------------------------- |
| 1.5B–2.7B  | `false`    | ~4–6 GB (float32)                           |
| 1.5B–2.7B  | `true`     | ~1.5–2 GB (4-bit, may fall back to float16) |
| 7B–8B      | `true`     | ~5–6 GB (4-bit, required)                   |

> **Note on 4-bit fallback:** For small models (Qwen2.5-1.5B, Phi-2) on some hardware/driver configurations, 4-bit loading may fail with a meta tensor error. The framework catches this automatically and falls back to float16 CUDA. The `copying from a non-meta parameter` warnings in the log are expected in this case and do not affect results.

### Adding a custom aggregation strategy

```yaml
aggregation:
  strategies:
    my_strategy:
      correctness:       0.50
      robustness:        0.30
      logical_coherence: 0.20
      consistency:       0.00
      efficiency:        0.00
      stability:         0.00
```

Weights are **auto-normalized** if they don't sum exactly to 1.0.

### Temperature and CS/SS measurement

By default, `deterministic: true` sets temperature = 0. This causes CS = SS = 1.0 for all models (deterministic models always produce the same output — a mathematical artifact, not a meaningful measurement).

To get meaningful CS/SS scores, add `temperature: 0.7` per model:

```yaml
experiment:
  deterministic: true    # Keep this — only the temperature param overrides it

models:
  - name: "GPT-4o-mini"
    type: "openai"
    params:
      model_id: "gpt-4o-mini"
      temperature: 0.7    # ← This overrides deterministic for this model only
```

---

## Checkpoint / Resume & LaTeX Export

**Checkpoint (item-level, on by default).** Every completed item
(prediction + K consistency runs + perturbation responses) is written
immediately to `outputs/checkpoints/<experiment_name>/<model>.jsonl`.
If a run crashes at item 700/975, the next run restores the completed
items and only executes the missing ones — the log shows e.g.
`[Checkpoint] 700/975 items restored — running remaining 275`.
The checkpoint carries a protocol/config fingerprint (model parameters,
token budget, seed, datasets, sample counts, metric settings, K/P settings,
and aggregation weights); if any of these change, the old file is renamed
`*.stale` and a fresh run starts. To force a fresh run, delete the
checkpoint folder or set `experiment: { checkpoint: false }`.

**Fixed-denominator failure policy.** Failed primary generations remain in
N as incorrect observations. Failed or unavailable consistency/stability
runs remain in K, and failed or unavailable perturbations remain in P, with
zero contribution. The framework never improves a score by silently
dropping a failed API call, and never substitutes the original question for
a missing perturbation.

**LaTeX tables.** After each model completes, `reasoning_quality_tables.tex`
is (re)written next to the Excel file: a booktabs overall table
(CQ–SS + all aggregation strategies, column-best in bold) and a
per-dataset breakdown table, ready to `\input` or paste into the
manuscript. Requires `\usepackage{booktabs}` (already loaded by MDPI).

## Outputs

All results are saved to `outputs/<experiment_name>_<timestamp>/`:

| File                             | Description                                                                            |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `reasoning_quality_results.xlsx` | Full results: raw metrics, all aggregation strategies, per-dataset breakdown, metadata |
| `radar_plot.png`                 | Multi-dimensional radar chart — one polygon per model                                  |
| `summary.json`                   | Complete results in machine-readable JSON                                              |
| `<ModelName>_result.json`        | Per-model detailed results                                                             |

### Excel structure

- **Overall Raw Metrics** — one row per model, columns: CQ, CS, RS, LS, ES, SS
- **Aggregated Scores** — composite Q scores per model × all seven aggregation strategies
- Additional sheets — per-dataset breakdown and experiment metadata (config parameters, timestamps, dataset sizes)

---

## Project Structure

```
RQEval/
│
├── config/
│   ├── config.yaml              ← Main config: add models/datasets/strategies here
│   ├── config_asu_createai.yaml ← Optional: institutional ASU CreateAI gateway setup
│   └── config_test.yaml         ← Quick test (mock + Phi-2 + synthetic, ~5 min)
│
├── models/
│   ├── base_model.py           ← Abstract base class (cache, interface)
│   │                             Cache disabled for stochastic models (temperature>0)
│   ├── openai_model.py         ← GPT-4o-mini, GPT-4o, any OpenAI-compatible API
│   ├── anthropic_model.py      ← Claude models
│   ├── gemini_model.py         ← Gemini models
│   ├── deepseek_model.py       ← DeepSeek (OpenAI-compatible endpoint)
│   ├── asu_model.py            ← ASU CreateAI gateway (institutional access only)
│   ├── local_model.py          ← HuggingFace local models
│   │                             4-bit quantization with float16 fallback
│   │                             Sequential RAM management (one model at a time)
│   │                             Pre-loading before evaluation loop (no per-item reload)
│   │                             Prompt templates per model family
│   └── mock_model.py           ← Deterministic mock for testing without APIs
│
├── llm_datasets/
│   ├── base_dataset.py         ← Abstract base + JSON file loader
│   ├── synthetic_dataset.py    ← Auto-generated reasoning/adversarial/robustness items
│   ├── gsm8k_dataset.py        ← GSM8K math word problems
│   ├── mmlu_dataset.py         ← MMLU multi-subject multiple choice
│   │                             Skips missing subjects gracefully
│   ├── strategyqa_dataset.py   ← StrategyQA commonsense yes/no
│   └── multi_dataset.py        ← Combines multiple datasets, tracks source per item
│
├── metrics/
│   ├── answer_extraction.py    ← Shared type-aware canonical answer matching,
│   │                             used by CQ, RS, and CS (see paper's
│   │                             Reproducibility note)
│   ├── accuracy.py             ← CQ — delegates to answer_extraction.py
│   ├── consistency.py          ← CS — pairwise agreement across K runs,
│   │                             computed over canonicalized answers
│   ├── robustness.py           ← RS — perturbation matching (conditioned on CQ,
│   │                             delegates to accuracy.py)
│   ├── logical_consistency.py  ← LS — NLI contradiction detection
│   ├── efficiency.py           ← ES — harmonic mean of CQ and inverse token count
│   ├── explainability.py       ← SS — BERTScore across reasoning traces
│   └── aggregation.py          ← Weighted composite Q score, 7 built-in strategies
│
├── evaluation/
│   └── evaluator.py            ← Main pipeline: load → generate → 6 metrics → export
│
├── visualization/
│   └── radar_plot.py           ← Radar chart + grouped bar chart
│
├── utils/
│   ├── logger.py               ← Structured logging
│   ├── reproducibility.py      ← Seed setting across Python / NumPy / PyTorch
│   ├── experiment_tracker.py   ← JSON + Excel export, result aggregation
│   ├── checkpoint.py           ← Item-level checkpoint / resume
│   └── latex_export.py         ← Auto-generated LaTeX result tables
│
├── outputs/                    ← Auto-created; all results (and checkpoints/) go here
├── rqeval/                     ← Installed-package glue (CLI, `rqeval setup`)
├── app.py                      ← Streamlit web interface (streamlit run app.py)
├── requirements.txt
└── main.py                     ← Entry point; config parsing + model/dataset registration

(Installed via PyPI, the package additionally provides the `rqeval` CLI:
`rqeval setup` copies main.py, app.py and config/ into your working directory.)
```

---

## Adding New Models

### Option A — Config only (API models)

For any OpenAI-compatible API (OpenRouter, Groq, etc.):

```yaml
- name: "My-Model"
  type: "openai"
  params:
    model_id: "my-model-id"
    api_key_env: "MY_API_KEY"
    max_tokens: 256
```

For HuggingFace local models:

```yaml
- name: "My-Local-Model"
  type: "local"
  params:
    model_id: "org/model-name"
    device: "cuda"
    use_4bit: true            # Falls back to float16 if unsupported
    max_new_tokens: 256
    temperature: 0.7
```

### Option B — Custom model class

1. Create `models/my_model.py` extending `BaseModel`
2. Implement `generate(prompt)` and `generate_with_trace(prompt)`
3. Add a `_build_mytype()` function in `main.py`
4. Register `"mytype": lambda n, p, mc, det, s: _build_mytype(n, p, mc, det)` in the dict returned by `_get_model_registry()` in `main.py`
5. Use `type: "mytype"` in config

---

## Adding New Datasets

### Option A — JSON file (no code needed)

Prepare a JSON file with this structure:

```json
[
  {
    "id": "q001",
    "question": "What is 2 + 2?",
    "answer": "4",
    "type": "reasoning",
    "perturbations": [
      "What does 2 plus 2 equal?",
      "Calculate 2 + 2",
      "Find the sum of 2 and 2"
    ]
  }
]
```

Then add to config:

```yaml
datasets:
  - name: "my_dataset"
    type: "json"
    params:
      path: "data/my_questions.json"
      num_samples: 100
```

The `perturbations` field is used for the RS (robustness) metric. If omitted, robustness is skipped for that item.

### Option B — HuggingFace dataset class

1. Create `llm_datasets/my_dataset.py` extending `BaseDataset`
2. Implement the `load()` method to populate `self._data`
3. Register the type in `main.py`

---

## Known Issues & Platform Notes

### Windows GPU — DLL errors (fbgemm.dll / cusparse64_11.dll)

Both errors share the same cause: pip CUDA wheels have DLL dependency issues on many Windows systems.

**Fix — use `conda install` instead of `pip install`:**

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

> ⚠️ **Do NOT install torch 2.10.x.** It breaks torchvision/torchaudio compatibility and reintroduces DLL errors. If you accidentally upgrade, restore with:
>
> ```bash
> pip uninstall torch torchvision torchaudio bitsandbytes -y
> pip cache purge
> conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
> pip install bitsandbytes==0.44.0
> ```

### bitsandbytes version compatibility

| bitsandbytes | torch             | Status                              |
| ------------ | ----------------- | ----------------------------------- |
| 0.44.0       | 2.4.0 + CUDA 12.1 | ✅ Tested, works                     |
| 0.49.x       | 2.4.0             | ❌ Incompatible — causes CUDA errors |
| any          | 2.10.x            | ❌ Do not use torch 2.10.x           |

### Windows CPU — PyTorch version

PyTorch 2.4+ causes `fbgemm.dll` errors with Windows CPU pip wheels. Use 2.3.x for CPU-only:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install "transformers==4.45.2"
```

### transformers version (CPU-only systems)

`transformers >= 4.46` requires `torch >= 2.4`. On Windows CPU where 2.4 cannot be installed, pin transformers to 4.45.2. On GPU systems with PyTorch 2.4+, install the latest transformers freely.

### Local model — meta tensor / copying from non-meta parameter warnings

During 4-bit model loading you may see many warnings like:

```
UserWarning: for model.layers.X...: copying from a non-meta parameter in the
checkpoint to a meta parameter in the current model, which is a no-op.
```

This is **expected and harmless**. It means the 4-bit loading path was attempted but fell back to float16 CUDA. The model loads correctly in float16 and inference proceeds normally.

### Local model — 4-bit fallback to float16

For small models (Qwen2.5-1.5B, Phi-2) 4-bit quantization may fail on some hardware with `Cannot copy out of meta tensor; no data!`. The framework catches this and automatically falls back to float16 CUDA. This is not an error — evaluation continues normally. float16 uses slightly more VRAM (~3 GB for 1.5B) but works reliably on GTX 1650.

### Local model — workers must be 1

The evaluator automatically detects local models and forces `workers=1` regardless of the `max_workers` config setting. Running multiple local model workers causes repeated HuggingFace downloads, CUDA OOM, and meta tensor errors. This is by design.

### Flash attention warning

`Torch was not compiled with flash attention.` — harmless on GTX 1650 (Turing architecture). The model uses standard scaled dot-product attention instead. Flash attention requires Ampere or newer (RTX 3000+).

### MMLU — missing subjects

Subjects not present in `cais/mmlu` are logged and skipped automatically; MMLU loads 225 items from the 9 available reasoning subjects. This is expected and matches the paper.

### Mistral / LLaMA tokenizer error

`Cannot instantiate this tokenizer from a slow version... sentencepiece` — fix with:

```bash
pip install sentencepiece
```

### CS = SS = 1.0 for all models

This happens when `deterministic: true` and no `temperature` is set per model. The cache returns the same response for all K runs. Fix: add `temperature: 0.7` to each model in config. See [Temperature and CS/SS measurement](#temperature-and-csss-measurement).

---

## Citation

If you use RQEval in your research, please cite:

```bibtex
@article{senol2026reasoning,
  title         = {Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework (RQEval)},
  author        = {Şenol, Ali and Agrawal, Garima and Liu, Huan},
  year          = {2026},
  eprint        = {2605.24661},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.24661}
}
```

*This citation refers to the arXiv preprint. The paper is currently under revision at MDPI Big Data and Cognitive Computing; once published, please prefer the journal version if available.*

---

## License

MIT License — see `LICENSE` for details.
