"""EXP — provenance on the REAL Enron graph, content-mimicry/BEC, multi-seed + CI.

Addresses reviewer [1] (no real data) and [5] (single split, no CI) for the provenance
claim. Uses Enron's ACTUAL per-user communication graph and ACTUAL mail:

  * Each user's contact graph is built from the senders in their TRAIN history; the
    TEST set therefore contains genuinely NEW correspondents -> realistic provenance
    false positives (not a synthetic perfect-separability artifact).
  * Benign (label 0) = real Enron message with its real sender.
  * Content-mimicry attack (label 1) = the SAME real benign text with the sender
    replaced by an off-graph address (realistic BEC / display-spoof). Body is real and
    benign, so a content detector has no signal; only the (real-graph) provenance does.

We repeat over K random splits and report mean AUC with a 95% CI across seeds.

Output: results/exp_provenance_enron.csv.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis.metrics import auc
from config import DATA_DIR, RESULTS_DIR

SEEDS = list(range(10))
MAX_USERS = 150   # full Enron maildir, matched to the N=150 synthetic twin population
MIN_MSGS = 30


def _rows_for_seed(graph: dict, seed: int) -> pd.DataFrame:
    """Per user: split mail; contacts from TRAIN senders; build benign/attack rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for user, d in graph.items():
        emails = d["emails"]
        if len(emails) < MIN_MSGS:
            continue
        idx = rng.permutation(len(emails))
        ntr = max(5, int(0.6 * len(emails)))
        tr_idx, te_idx = set(idx[:ntr].tolist()), set(idx[ntr:].tolist())
        contacts_train = {emails[i][0] for i in tr_idx}      # real graph from history
        for split, ids in (("train", tr_idx), ("test", te_idx)):
            for i in ids:
                sender, text = emails[i]
                # deterministic parity: half benign (real sender), half BEC (off-graph)
                if (i % 2) == 0:
                    label, s_addr = 0, sender
                else:
                    label, s_addr = 1, f"attacker{i}@external-mail.com"  # off-graph spoof
                rows.append({"user": user, "split": split, "text": text, "label": label,
                             "sender": s_addr, "contacts": contacts_train})
    return pd.DataFrame(rows)


def _prov(df: pd.DataFrame, victim_contacts) -> np.ndarray:
    """in_graph feature: sender in (victim's) train contact set."""
    return np.asarray([[1.0 if r.sender in victim_contacts(r.user, r.contacts) else 0.0]
                       for r in df.itertuples()], dtype=np.float32)


def _eval(df: pd.DataFrame, users: list[str]):
    train, test = df[df.split == "train"], df[df.split == "test"]
    yte = test["label"].to_numpy()
    real_c = lambda u, c: c
    wrong = {users[i]: users[(i + 1) % len(users)] for i in range(len(users))}
    # shuffled: evaluate sender against a WRONG user's contacts
    contacts_by_user = {u: g for u, g in zip(df["user"], df["contacts"])}
    wrong_c = lambda u, c: contacts_by_user.get(wrong.get(u, u), set())

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000).fit(train["text"])
    ytr = train["label"].to_numpy()

    def fit(extr_tr, extr_te, text):
        parts_tr = ([vec.transform(train["text"])] if text else []) + ([csr_matrix(extr_tr)] if extr_tr is not None else [])
        parts_te = ([vec.transform(test["text"])] if text else []) + ([csr_matrix(extr_te)] if extr_te is not None else [])
        Xtr = hstack(parts_tr).tocsr() if len(parts_tr) > 1 else parts_tr[0]
        Xte = hstack(parts_te).tocsr() if len(parts_te) > 1 else parts_te[0]
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1).fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]

    pr_tr, pr_te = _prov(train, real_c), _prov(test, real_c)
    sh_te = _prov(test, wrong_c)
    a_content = auc(yte, fit(None, None, True))
    a_prov = auc(yte, fit(pr_tr, pr_te, False))
    a_comb = auc(yte, fit(pr_tr, pr_te, True))
    a_shuf = auc(yte, fit(_prov(train, wrong_c), sh_te, False))
    # provenance FPR = benign whose (real) sender is NOT in the train graph (new senders)
    fpr = float(np.mean([pr_te[j][0] == 0.0 for j in range(len(yte)) if yte[j] == 0]))
    return a_content, a_prov, a_comb, a_shuf, fpr


def _ci(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals)
    m, s = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    half = 1.96 * s / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, half


def main() -> None:
    from data.load_enron_graph import parse_user_graph

    print("[enron-prov] parsing real Enron graph (may take a few minutes)...", flush=True)
    graph = parse_user_graph(DATA_DIR / "enron", min_msgs=MIN_MSGS, max_users=MAX_USERS)
    users = sorted(graph)
    print(f"[enron-prov] {len(users)} users | seeds={len(SEEDS)}", flush=True)

    acc = {k: [] for k in ["content", "provenance", "combined", "shuffled", "prov_fpr"]}
    for seed in SEEDS:
        df = _rows_for_seed(graph, seed)
        c, p, cm, sh, fpr = _eval(df, users)
        for k, v in zip(["content", "provenance", "combined", "shuffled", "prov_fpr"], [c, p, cm, sh, fpr]):
            acc[k].append(v)
        print(f"  seed={seed}: content={c:.3f} prov={p:.3f} comb={cm:.3f} shuf={sh:.3f} fpr={fpr:.3f}", flush=True)

    out = RESULTS_DIR / "exp_provenance_enron.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "ci95_halfwidth", "min", "max", "n_seeds"])
        for k, vals in acc.items():
            m, h = _ci(vals)
            w.writerow([k, round(m, 4), round(h, 4), round(min(vals), 4), round(max(vals), 4), len(vals)])
    print(f"\n[enron-prov] wrote {out}")
    for k, vals in acc.items():
        m, h = _ci(vals)
        print(f"  {k:11s}: {m:.3f} ± {h:.3f} (95% CI, {len(vals)} seeds)")


if __name__ == "__main__":
    main()
