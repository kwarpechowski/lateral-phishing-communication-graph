"""EXP — provenance under content-mimicry, swept over the COMPROMISED-account fraction.

Hardens the headline positive (Tab.~results-rq5): the perfect provenance detection in
the content-mimicry regime is not a trivial artifact but a function of a realistic
attacker parameter --- how often the attacker sends from a COMPROMISED legitimate
account (which is in the victim's graph and therefore invisible to provenance).

For each fraction ``f`` of mimicry attacks sent from a compromised in-graph account
(the rest spoof an off-graph address), we report content AUC (always near chance,
because the body is benign) and provenance / combined AUC. As ``f`` rises from 0 to 1,
provenance degrades gracefully toward chance --- quantifying exactly when victim-graph
provenance helps. Output: results/exp_provenance_sweep.csv.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

from analysis.metrics import auc
from config import RANDOM_SEED, RESULTS_DIR
from data.build_provenance import _addr
from experiments.exp_provenance import _load_graph, _prov_feats

FRACTIONS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]


def _h(*p: str) -> int:
    return int(hashlib.sha256(":".join(p).encode()).hexdigest(), 16)


def _build(frac: float) -> pd.DataFrame:
    """Benign corpus relabeled: half become mimicry attacks; ``frac`` of those use a
    compromised in-graph sender (uncatchable by provenance), the rest spoof off-graph."""
    ben = pd.read_csv(RESULTS_DIR / "benign_corpus.csv").reset_index().rename(columns={"index": "idx"})
    graph = _load_graph()
    rows = []
    for r in ben.itertuples():
        g = graph.get(r.profile_id)
        if not g or not g["addrs"]:
            continue
        names, addrs = sorted(g["names"]), sorted(g["addrs"])
        h = _h(r.profile_id, str(r.idx))
        name = names[h % len(names)]
        if h % 2 == 0:  # mimicry attack
            compromised = (_h("comp", r.profile_id, str(r.idx)) % 1000) / 1000.0 < frac
            addr = addrs[h % len(addrs)] if compromised else _addr(name, "gmail.com")
            label = 1
        else:           # genuine benign
            addr = addrs[h % len(addrs)]
            label = 0
        rows.append({"profile_id": r.profile_id, "text": str(r.text), "label": label,
                     "sender_name": name, "sender_address": addr})
    return pd.DataFrame(rows)


def _auc_for(df, real_of):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_SEED)
    tr, te = next(gss.split(df, groups=df["profile_id"]))
    train, test = df.iloc[tr], df.iloc[te]
    y = test["label"].to_numpy()
    pr_tr, pr_te = _prov_feats(train, real_of), _prov_feats(test, real_of)

    def fit_score(extra_tr, extra_te, text):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000).fit(train["text"]) if text else None
        def feats(d, ex):
            parts = ([vec.transform(d["text"])] if vec is not None else []) + ([csr_matrix(ex)] if ex is not None else [])
            return hstack(parts).tocsr() if len(parts) > 1 else parts[0]
        clf = LGBMClassifier(n_jobs=1, random_state=RANDOM_SEED, n_estimators=300, verbose=-1).fit(feats(train, extra_tr), train["label"])
        return auc(y, clf.predict_proba(feats(test, extra_te))[:, 1])

    return (fit_score(None, None, True),            # content
            fit_score(pr_tr, pr_te, False),         # provenance only
            fit_score(pr_tr, pr_te, True))          # content+provenance


def main() -> None:
    graph = _load_graph()
    real_of = lambda p: graph.get(p, {"addrs": set(), "names": set(), "domain": ""})
    rows = []
    for f in FRACTIONS:
        c, p, cp = _auc_for(_build(f), real_of)
        rows.append({"compromised_frac": f, "content_auc": round(c, 4),
                     "provenance_auc": round(p, 4), "combined_auc": round(cp, 4)})
        print(f"  frac={f:.2f}: content={c:.3f}  provenance={p:.3f}  combined={cp:.3f}", flush=True)
    out = RESULTS_DIR / "exp_provenance_sweep.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["compromised_frac", "content_auc", "provenance_auc", "combined_auc"])
        w.writeheader(); w.writerows(rows)
    print(f"[provsweep] wrote {out}")


if __name__ == "__main__":
    main()
