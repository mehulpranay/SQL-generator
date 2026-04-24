
# NL2SQL: Matching GPT-3.5 at 1/10th the Cost

Fine-tuning Llama 3.1 8B on the Spider benchmark using QLoRA — achieving **~70% Execution Accuracy** on par with GPT-3.5 zero-shot, trained entirely on free Kaggle T4 GPUs.

> **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/Rickster1995/nl2sql-demo) | **Model Adapter:** [Rickster1995/nl2sql-llama3-qlora](https://huggingface.co/Rickster1995/nl2sql-llama3-qlora)

---

## The Problem

Every analyst who can't write SQL is blocked. Every BI tool that requires hand-written queries has a ceiling. NL2SQL removes that ceiling — but the difference between a 70% and 50% accurate system is the difference between a tool people trust and one they abandon.

SQL generation punishes sloppy models fast. You're not generating prose — you're generating code that gets executed against a real database. Either the result set matches, or it doesn't. There's no partial credit.

---

## Results

| Model | Method | Spider Execution Accuracy |
|---|---|---|
| GPT-3.5-turbo | Zero-shot prompting | ~70–75% |
| **Llama 3.1 8B (this project)** | **QLoRA fine-tuning** | **~70%** |
| Llama 3.1 8B | No fine-tuning | ~40–50% |

Matching GPT-3.5 with an 8B model fine-tuned on consumer hardware is the headline result.

---

## The Key Insight: Schema Representation

Most NL2SQL tutorials dump raw `CREATE TABLE` strings into the prompt. This project doesn't.

Spider ships with structured PK/FK annotations. A custom formatter converts these into explicit, human-readable representations:

```
# Table: concert
## Columns:
- concert_ID (number, PK)
- Stadium_ID (text, FK -> stadium.Stadium_ID)
- Year (text)
```

By annotating join paths explicitly, the model sees the relational structure directly — instead of inferring it from column names. **This single change is what pushes accuracy from ~50% to 70%.**

---

## Architecture
<img width="1637" height="4249" alt="Architecture" src="https://github.com/user-attachments/assets/73a3b276-a4e1-48c0-8172-8045d1f327b4" />

```
Input: Natural Language Question + Database Schema
        ↓
Schema Formatter (PK/FK annotations)
        ↓
Prompt Builder (Llama 3.1 chat format)
        ↓
Llama 3.1 8B Instruct [4-bit QLoRA]
  - Base model: 4-bit quantized (~5.6GB VRAM)
  - LoRA adapters: r=16, targeting all projection layers
  - Trainable params: ~1-2% of total
        ↓
Generated SQL
        ↓
Execution Accuracy Evaluator (SQLite)
```

---

## Project Structure

```
├── question-to-query.ipynb           # Full training pipeline: EDA, schema parsing,
│                                     # prompt engineering, QLoRA setup, training, eval
├── inference-notebook-for-          
│   question-to-sql-query.ipynb       # Inference pipeline for running the trained adapter
└── README.md
```

---

## Training Setup

**Dataset:** Spider NL2SQL benchmark — 7,000 training examples across 140+ databases. Train/val split is at the database level, so the model must generalize to entirely unseen schemas.

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`

**Hardware:** Kaggle dual T4 GPUs (2x 16GB VRAM)

**Key config decisions:**
- `gradient_accumulation_steps=8` — effective batch size of 8 without 8x VRAM
- `paged_adamw_8bit` — offloads optimizer states to CPU, essential for T4s
- `train_on_responses_only` — computes loss only on SQL output tokens, not schema/question tokens. Without this, ~60-70% of gradient signal is wasted on tokens the model was just told
- `cosine` LR schedule from 5e-5, 3 epochs
- Filtered 82 training examples exceeding 2048 tokens to prevent silent truncation

---

## Evaluation: Execution Accuracy

BLEU and exact string match are meaningless for SQL — `SELECT COUNT(*)` and `SELECT count(*)` are identical queries that score 0 on exact match.

This project uses **Execution Accuracy**: run both predicted and ground-truth SQL against the actual SQLite database, compare result sets order-insensitively using `Counter` equality. Invalid SQL counts as wrong.

Evaluation is parallelized with `ThreadPoolExecutor` since SQLite reads are I/O-bound.

---

## Error Analysis

| Failure Mode | Example | Root Cause |
|---|---|---|
| Schema linking | Maps "name of song" → `name` column instead of `song_name` | Column disambiguation without extra metadata |
| Over-aggregation | Wraps already-aggregate column in `AVG()` | Semantic misunderstanding of column names |
| GROUP BY key | Groups by descriptive string instead of PK | Doesn't consistently prefer ID columns for grouping |
| Set operations | `EXCEPT` drops duplicates unintentionally | Shallow understanding of set-theory semantics |

The hard/extra-hard Spider categories (~30% of validation) heavily feature nested queries and set operations — consistent with where the model struggles.

---

## What's Next

- **DPO fine-tuning** using execution success as the reward signal — treating correct SQL execution as positive examples and failures as negative, without needing human preference labels
- **RAG over schema** — for databases with 50+ tables, an embedding-based table retrieval step to prune irrelevant schema before the prompt
- **Schema pruning** — select only tables relevant to the question, reducing prompt length and improving focus on complex schemas
- **MLOps** — W&B experiment tracking, FastAPI inference wrapper, Docker containerization

---

## Key Learnings

1. **Schema representation is load-bearing.** Explicit PK/FK annotations in the prompt are not a nice-to-have — they're what makes multi-table queries possible at all.
2. **`train_on_responses_only` matters more than it looks.** Computing loss on schema tokens wastes ~60-70% of gradient signal per example.
3. **Execution Accuracy is the only honest metric for SQL.** Implement it early — string-match proxies hide real model failures.
4. **Kaggle's environment has sharp edges.** HuggingFace cache corruption (EOF errors) fixed via `snapshot_download` with `local_dir_use_symlinks=False`.

---

## Resources

- **Full write-up:** [Blog post](<!-- add link -->)
- **Dataset:** [Spider](https://huggingface.co/datasets/spider) | [Spider Schema](https://huggingface.co/datasets/richardr1126/spider-schema)
- **Base model:** [Unsloth Llama 3.1 8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)
- **Unsloth:** [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
