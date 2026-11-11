"""EXP — provenance vs REALISTIC attacker variants on the real Enron graph ([A], [D]).

The earlier real-Enron test used a single straw-man sender (a hardcoded external
address). A reviewer rightly notes that real BEC uses look-alike domains, compromised
in-graph accounts, or display-spoofs. Here we evaluate provenance against three honest
attacker types, with richer provenance features and the operationally relevant
Recall@FPR=1% (not only AUC), multi-seed with CIs:

  * external_freemail : sender from a generic external freemail address (off-graph) --- easy.
  * lookalike_domain  : a known contact's local-part at a TYPO-SQUATTED version of their
    domain (off-graph domain, not freemail) --- needs domain-level reasoning.
  * compromised_account: a REAL in-graph contact address (account takeover) --- provenance
    CANNOT catch this (sender is legitimately in the graph).

Provenance features (per email, vs the victim's TRAIN graph): address in graph; sender
DOMAIN among the victim's known domains; sender is a free-mail provider.

Output: results/exp_provenance_attackers.csv (attacker x {AUC, Recall@FPR=1%} mean±CI).
"""

from __future__ import annotations

import csv

import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix

from analysis.metrics import auc, recall_at_fpr
from config import DATA_DIR, RESULTS_DIR

SEEDS = list(range(8))
MAX_USERS = 150   # full Enron maildir, matched to the N=150 synthetic twin population
MIN_MSGS = 30
FREEMAIL = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "protonmail.com", "external-mail.com", "mail.com", "gmx.com"}
TYPES = ["external_freemail", "lookalike_domain", "compromised_account"]


def _dom(a: str) -> str:
    return a.split("@")[-1] if "@" in a else ""


def _lookalike(domain: str) -> str:
    """A typo-squatted variant of a domain (off-graph but visually similar)."""
    if len(domain) < 4:
        return domain + "x.com"
    i = len(domain) // 2
    return domain[:i] + domain[i] + domain[i:]  # duplicate a middle char


def _feats(sender: str, addrs: set, domains: set) -> list[float]:
    d = _dom(sender)
    return [1.0 if sender in addrs else 0.0,
            1.0 if d in domains else 0.0,
            1.0 if d in FREEMAIL else 0.0]


def _build(graph: dict, seed: int, atk_type: str):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for user, d in graph.items():
        emails = d["emails"]
        if len(emails) < MIN_MSGS:
            continue
        idx = rng.permutation(len(emails))
        ntr = max(5, int(0.6 * len(emails)))
        tr_ids, te_ids = set(idx[:ntr].tolist()), set(idx[ntr:].tolist())
        addrs = {emails[i][0] for i in tr_ids}
        domains = {_dom(emails[i][0]) for i in tr_ids}
        contacts = sorted(addrs)
        for bucket, ids in ((train, tr_ids), (test, te_ids)):
            for i in ids:
                sender, text = emails[i]
                if i % 2 == 0:                       # benign: real sender
                    bucket.append((_feats(sender, addrs, domains), 0))
                else:                                 # attack of the requested type
                    base = contacts[i % len(contacts)] if contacts else "x@x.com"
                    if atk_type == "external_freemail":
                        s = "attacker@external-mail.com"
                    elif atk_type == "lookalike_domain":
                        s = base.split("@")[0] + "@" + _lookalike(_dom(base))
                    else:                             # compromised in-graph account
                        s = base
                    bucket.append((_feats(s, addrs, domains), 1))
    Xtr = csr_matrix(np.array([f for f, _ in train], dtype=np.float32))
    ytr = np.array([y for _, y in train])
    Xte = csr_matrix(np.array([f for f, _ in test], dtype=np.float32))
    yte = np.array([y for _, y in test])
    return Xtr, ytr, Xte, yte


def _ci(vals):
    a = np.asarray(vals, dtype=float)
    m = float(a.mean())
    h = 1.96 * float(a.std(ddof=1)) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, h


def main() -> None:
    from data.load_enron_graph import parse_user_graph

    print("[atk] parsing real Enron graph...", flush=True)
    graph = parse_user_graph(DATA_DIR / "enron", min_msgs=MIN_MSGS, max_users=MAX_USERS)
    print(f"[atk] {len(graph)} users | seeds={len(SEEDS)} | types={TYPES}", flush=True)

    rows = []
    for t in TYPES:
        aucs, recs = [], []
        for seed in SEEDS:
            Xtr, ytr, Xte, yte = _build(graph, seed, t)
            clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(Xtr, ytr)
            s = clf.predict_proba(Xte)[:, 1]
            aucs.append(auc(yte, s))
            recs.append(recall_at_fpr(yte, s, fpr_target=0.01))
        am, ah = _ci(aucs); rm, rh = _ci(recs)
        rows.append({"attacker": t, "auc": round(am, 3), "auc_ci": round(ah, 3),
                     "recall_at_fpr1": round(rm, 3), "recall_ci": round(rh, 3)})
        print(f"  {t:20s} AUC={am:.3f}±{ah:.3f}  Recall@FPR1%={rm:.3f}±{rh:.3f}", flush=True)

    out = RESULTS_DIR / "exp_provenance_attackers.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["attacker", "auc", "auc_ci", "recall_at_fpr1", "recall_ci"])
        w.writeheader(); w.writerows(rows)
    print(f"[atk] wrote {out}")


if __name__ == "__main__":
    main()
