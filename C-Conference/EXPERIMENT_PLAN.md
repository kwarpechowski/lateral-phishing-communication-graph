# EXPERIMENT_PLAN.md — Paper C: independent corroboration as a defense of GraphRAG against knowledge poisoning

Pre-registered plan (rule 2 of the rebuild brief). **No experiment in the
pre-registered grids below is run before this plan is committed.** Changes to
this plan are appended to §7 with a dated justification.

## 1. Research questions

- **RQ1 (corroboration gate).** Does an independent-corroboration gate — trust
  a knowledge fragment only if ≥m *independent* sources confirm it (hard mode:
  reject; soft mode: down-rank) — reduce the attack success rate (ASR) of
  known poisoning attacks in real GraphRAG pipelines, and at what price in
  clean-question accuracy and rejected/degraded true facts? Grid: m ∈ {1,2,3,4}
  × {hard, soft} × ≥2 pipelines.
- **RQ2 (adaptive attacker).** An attacker who knows the gate plants m
  mutually-confirming fabricated sources ("different domains", different
  styles). Measure: (a) attack cost (number of sources × tokens) to restore
  ASR ≥ baseline, as a function of m; (b) detectability of planted
  confirmation clusters by temporal co-arrival, stylistic similarity, source
  freshness (AUC, sens@FPR1%, and the cost on genuinely independent true
  sources).
- **RQ3 (fabrication on a real KG).** On Wikidata vandalism corpora
  (real labels), do relational-consistency features match the
  literature-standard vandalism features under the same tuning budget, with a
  temporal train/test split (AUC, PR-AUC, sens@FPR1%, 95% bootstrap CI)?
  FB15k-237 is an appendix-only sanity check *after* the known asymmetry is
  fixed: features computed on a graph with ALL evaluation triples (true and
  fabricated) removed; negatives = random, type-constrained, degree-matched,
  and path-based degree-matched; baselines = CN, Adamic-Adar, RA, and an
  embedding model (RotatE/TransE) from a maintained library.

## 2. Pipelines, datasets, attacks (availability to be verified in Stage 1)

| Component | Candidate | Status |
|---|---|---|
| Pipelines (≥2 of) | Microsoft GraphRAG, LightRAG, HippoRAG 2 | verified public + MIT: HKUDS/LightRAG (EMNLP 2025), OSU-NLP-Group/HippoRAG (NeurIPS'24); `graphrag` on PyPI 3.3.0; local install pending LLM budget |
| LLM | open 7–8B class (e.g. via ollama/vLLM) or fixed-budget API | **blocked in this environment**: no GPU, no ollama, no API key set (checked 2026-11-13); decision needed: API budget or GPU host |
| Attack 1 | PoisonedRAG (official code if available) | verified: sleeeeper/PoisonedRAG, MIT, USENIX Security 2025; BEIR datasets auto-download; eval needs API LLM (PaLM2/GPT/Llama-2 token) |
| Attack 2 | KEPo (official code if available) | **no public code found** (GitHub search 2026-11-13); reimplementation from the paper would be needed; venue verified WWW'26 (DOI 10.1145/3774904.3792547) |
| Question sets | as used by PoisonedRAG/KEPo (MuSiQue, HotpotQA, MultiHop-RAG) | PoisonedRAG: BEIR (NQ, HotpotQA) via `prepare_dataset.py` (auto-download); KEPo's exact splits to be checked in the paper text |
| RQ3 corpora | WSDM Cup 2017 vandalism corpus, Wikidated 1.0 | repos verified public: heindorf/wsdmcup17-wdvd-classification (+ cikm16 feature extraction, www19 debiasing code); license to inspect in Stage 2 |

## 3. Metrics

- ASR: fraction of questions answered with the attacker's target answer
  (string-match where the target is fixed, as in PoisonedRAG's protocol;
  LLM-judge variant reported if budget allows).
- Clean accuracy: EM / F1 on unattacked questions.
- Retrieval corruption: fraction of top-k retrieved chunks that are poisoned.
- Defense price: fraction of true facts rejected/degraded by the gate
  (hard/soft), and clean-accuracy delta.
- RQ3: AUC, PR-AUC, sens@FPR1%, 95% bootstrap CIs.

## 4. Grids, seeds, tuning budget

- Corroboration threshold m ∈ {1,2,3,4}, modes {hard, soft} (RQ1);
  adaptive escalation m ∈ {1,2,3,4} planted sources (RQ2).
- Seeds: ≥5 for LLM-in-the-loop experiments (more if budget allows), 10 for
  non-LLM (RQ3). Bootstrap 95% CI (percentile, 2000 resamples); paired tests
  (Wilcoxon signed-rank) for headline contrasts.
- **Tuning budget parity:** the gate and every baseline get the same number
  of tuning trials on the validation split (≤8 grid points each), chosen
  before the test run; the test set is run once.

## 5. Acceptance criteria (Stage 1 / Checkpoint 1)

- Reproduced ASR of PoisonedRAG-style attacks on the chosen pipelines falls
  within a reasonable range of the published numbers for the corresponding
  setting (documented in the checkpoint report with the exact numbers and
  cost). If it does not, we stop and report; we do not proceed to Stage 2.
- KEPo's reported finding (PoisonedRAG-style attacks largely ineffective
  against GraphRAG) is treated as a hypothesis to *check*, not assume.

## 6. Reporting rules

- All variants are reported, including failed ones. If a result weakens the
  thesis, the text changes, not the experiment.
- All numbers in the text come from CSVs via generated macros.
- The synthetic organizational multiplex (the old Track 1) is demoted to a
  motivational appendix without empirical claims.

## 7. Plan changes (append-only)

- (none yet)
