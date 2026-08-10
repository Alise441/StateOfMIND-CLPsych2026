# StateOfMIND: Self-State Identification with Retrieved In-Context Examples and Open-Weight LLMs

Code for the **StateOfMIND** team's submission to Task 1 of the [CLPsych 2026 shared task](https://clpsych.org/), which asks systems to identify adaptive and maladaptive psychological self-states in social media posts under the MIND (ABCD) framework.

> Alina Ponomareva, Nina Stekacheva Sancho, Karina Litvinova. **Self-State Identification with Retrieved In-Context Examples and Open-Weight LLMs.** In *Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026)*, pages 521–530. [aclanthology.org/2026.clpsych-1.42](https://aclanthology.org/2026.clpsych-1.42/)

No fine-tuning: a retrieval-augmented, in-context-learning ensemble of two open-weight LLMs (Qwen3.5-27B and Mistral-Small-3.2-24B-Instruct). Ranked **2nd of 17 teams** on Task 1.1 (subelement classification, 0.0004 behind the winner) and **5th of 17** on Task 1.2 (presence rating), with the best maladaptive-state subelement F1 of any submitted system.

## Architecture

For each post and each of the two models:

```
Retrieval:  post → hybrid similarity (text cosine + label-aware codebook)
                 → top-20 pool → MMR diversity re-rank (α=0.6) → 8 in-context examples

Prompting:  P1 (Unified)          — all elements + both presence scores, one pass
            P2 (Adaptive-focused) — adaptive elements only, conditioned on P1's maladaptive output
            P3 (Affect-focused)   — Affect element only, with Affect-specific retrieval

Merge:      Stage 1 (per-model):  maladaptive elements + presence ← P1
                                   adaptive elements               ← P2
                                   Affect element (either valence)  ← P3, overrides P1/P2
                                   no elements in a valence → presence reset to 1

            Stage 2 (cross-model, subtask-specific):
              Task 1.1 (Macro F1, sensitive to false positives) → Qwen's elements only
              Task 1.2 (RMSE, sensitive to missed elements)     → union of both models,
                                                                    Mistral wins conflicts
              Presence (both tasks): average of both models, {1,2} → 2
```

See the paper for the full method description, ablations, and discussion (retrieval is the dominant driver of performance; no single merge strategy is best for both subtasks; the system is substantially weaker on adaptive than maladaptive states).

## Repository structure

```
taxonomy.py                  MIND ABCD taxonomy: elements, subelements, valences, presence scale
data_loader.py                Reads timeline JSON files into Post objects
rag_index.py                  Retrieval: gte-multilingual-base embeddings, codebook scoring, MMR
prompt_builder_rag.py         Builds P1/P2/P3 prompts
output_parser.py              Parses structured LLM output into predictions
evaluate.py                   Task 1.1 (subelement Macro F1) and Task 1.2 (RMSE) evaluation
submission_formatter.py       Predictions → task1_pred.json (Codabench submission format)
distribution_analysis.py      CLI: compare train-gold vs. predicted label distributions
compare_submissions.py        CLI: diff two submissions' scoring results (TP/FP/FN)

official_eval/                 Shared-task organizers' evaluation and validation scripts
tests/test_all.py              Unit tests for taxonomy, prompting, parsing, evaluation

notebooks/
  01_submission1_train.ipynb           Submission 1 (solo Qwen): train-set inference + eval
  02_submission1_test.ipynb            Submission 1: test-set inference + submission file
  03_submission2_ensemble_union.ipynb  Submission 2 (ensemble, union merge)
  04_submission3_ensemble_qwen_elements.ipynb  Submission 3 (ensemble, Qwen-elements merge)
  05_ablation_prompting_and_merge.ipynb        Ablations 1 (prompting) and 3 (ensemble merge)
  06_ablation_retrieval.ipynb                  Ablation 2 (retrieval strategy)
```

Each notebook's first cell moves the working directory to the repo root, so open them from wherever Jupyter is running — no manual `cd` needed.

## Setup

```bash
pip install -r requirements.txt
```

Inference uses [vLLM](https://github.com/vllm-project/vllm) to serve Qwen3.5-27B and Mistral-Small-3.2-24B-Instruct locally, in FP16, with no fine-tuning — you'll need a GPU with enough VRAM to hold one ~25–27B model at a time (the notebooks free VRAM between models rather than loading both simultaneously).

### Data

The CLPsych 2026 dataset is not included in this repository. It requires signing the shared task's Data Access Agreement; obtain it directly from the [CLPsych 2026 organizers](https://clpsych.org/). Once obtained, place it at the repo root as:

```
train_tasks12/*.json            30 training timelines
test_tasks12nolabels/*.json     10 test timelines (no labels)
```

`01_submission1_train.ipynb` and `02_submission1_test.ipynb` (Submission 1 only) additionally expect an `augmented_data/*.json` directory of synthetic posts for rare Affect subelements, also not included. Our ablations found this augmentation helps the solo-model system but hurts the ensemble, so **Submissions 2 and 3 (the best-performing systems) don't use it at all**. To run the Submission 1 notebooks without it, drop the `aug_posts = load_all_timelines('augmented_data')` line and set `rag_index_aug = rag_index_real`.

## Results

| Submission | Description | Task 1.1 Sub F1 ↑ | Task 1.2 RMSE ↓ |
|---|---|---|---|
| 1 | Solo Qwen3.5-27B | 0.440 | 1.094 |
| 2 | Ensemble, union merge (Mistral wins conflicts) | 0.433 | **0.994** |
| 3 | Ensemble, Qwen-elements merge | **0.441** | 1.021 |

Leaderboard placement (17 teams): **2nd** on Task 1.1 (Submission 3), **5th** on Task 1.2 (Submission 2), **1st** on maladaptive-valence subelement F1. Full ablations, per-element breakdowns, and discussion are in the paper.

## Citation

```bibtex
@inproceedings{ponomareva-etal-2026-self,
    title = "Self-State Identification with Retrieved In-Context Examples and Open-Weight {LLM}s",
    author = "Ponomareva, Alina  and
      Stekacheva Sancho, Nina  and
      Litvinova, Karina",
    booktitle = "Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.clpsych-1.42/",
    doi = "10.18653/v1/2026.clpsych-1.42",
    pages = "521--530",
}
```

## Ethics and data access

The CLPsych 2026 dataset is an extended, de-identified subset of the CLPsych 2022 "Reddit-New" corpus. Per the shared task's Data Access Agreement, the dataset (and anything derived from it, including the synthetic `augmented_data`) must not be redistributed — this repository contains only code. See the paper's Ethics section and the [CLPsych 2026 guidelines](https://clpsych.org/) for details.

## License

MIT — see [LICENSE](LICENSE).
