# Stage 1 — reproduction of known poisoning attacks (checkpoint 1 artifacts)

Protocol: official PoisonedRAG code (sleeeeper/PoisonedRAG, MIT, USENIX
Security 2025), white-box `LM_targeted` attack against the Contriever
retriever, BEIR datasets, precomputed clean rankings from the official repo,
M=10 target questions x repeat_times=10 (=100 questions per dataset),
adv_per_query=5, seed=12 — the repository defaults of the paper's protocol.

Changes to the official code for this environment (documented for
reproducibility; all in the local run, none of them semantic):
- CPU adaptation: `torch.cuda.set_device` and `.cuda()` -> device-aware
  (no GPU available in this environment);
- `PYTHONUTF8=1` for console encoding on Windows;
- LLM provider: NASK GLM-5.2 via the internal OpenAI-compatible API
  (openai-python reads `OPENAI_BASE_URL`/`OPENAI_API_KEY`; model config
  `glm52_config.json`, provider "gpt", model name "glm-5-2",
  temperature 0.1, max_output_tokens 300);
- PaLM2/Vicuna/Llama model classes disabled (imports unavailable) — the
  "gpt" provider path is the only one exercised;
- clean EM/F1 scored by `clean_eval.py` (same 100 NQ questions, clean
  top-5, string EM vs the dataset's correct answer + token F1), because
  the official clean path has `topk_contents` unset (upstream bug).

## Results (`reproduction_summary.csv`)

| dataset | ASR mean (iters) | Ret P/R/F1 | clean EM / F1 | duration |
|---|---|---|---|---|
| NQ       | **0.95** (0.9–1.0) | 0.96 / 0.96 / 0.96 | 0.69 / 0.146 | ~13 min |
| HotpotQA | **0.98** (0.9–1.0) | 1.0 / 1.0 / 1.0 | — | ~17 min |

Acceptance (EXPERIMENT_PLAN §5): PoisonedRAG reports ~90% ASR with 5
injected texts per question (abstract); the reproduction on both datasets
lands at/above that range with GLM-5.2 as the answering LLM, and the
retrieval corruption (Ret F1 0.96–1.0, i.e., the poisoned texts enter the
top-5 almost always) matches the published mechanism. **Criterion met**;
KEPo has no public code (plan §2) — Stage 2 proceeds with PoisonedRAG as
the reproduced attack; KEPo's finding (PoisonedRAG-style attacks largely
ineffective against *GraphRAG*) is a Stage-2 hypothesis to check, not
assumed.

Cost: ~0.3M LLM tokens per full attack run + ~0.25M for the clean baseline
(internal NASK budget); retrieval attack on CPU ~7 s/question.

## Raw artifacts

- `reproduction/nask_full_nq.json` — official-format per-question results
  (NQ, attack): prompts, outputs, injected adv texts, per-question ASR.
- `reproduction/nask_full_hotpotqa.json` — same for HotpotQA.
- `reproduction/clean_eval_results.json` — clean top-5 baseline (NQ).

Environment: Windows, CPU-only (no GPU), Python 3.11, torch 2.15+cpu,
sentence-transformers 6.3.0, openai 2.44.0; LLM: NASK GLM-5.2
(api.ai.nask.pl, internal).
