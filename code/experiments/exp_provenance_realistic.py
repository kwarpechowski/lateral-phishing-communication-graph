"""EXP — provenance under REALISTIC two-sided graph noise (de-constructs RQ5, addresses M1).

The headline RQ5 result assumed perfect separability: benign always from an in-graph
sender, attacks always off-graph. Reality is noisier on BOTH sides:
  * some LEGITIMATE mail comes from senders NOT yet in the victim's graph (a new client,
    a first-time contact) -> provenance false-positives them;
  * some ATTACKS come from a COMPROMISED in-graph account -> provenance misses them.

We sweep the benign off-graph (new-sender) fraction at a fixed, realistic attack
compromise rate and report content vs provenance vs combined AUC, plus the provenance
false-positive rate. This shows provenance is NOT magic — graph noise costs it — yet it
still beats content in the content-mimicry regime, where content stays at chance.

Output: results/exp_provenance_realistic.csv.
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

BENIGN_OFFGRAPH = [0.0, 0.05, 0.1, 0.2, 0.3]
ATTACK_COMPROMISED = 0.3


def _h(*p: str) -> int:
    return int(hashlib.sha256(":".join(p).encode()).hexdigest(), 16)


def _build(benign_offgraph: float, attack_compromised: float) -> pd.DataFrame:
    """Mimicry corpus with two-sided graph noise (benign new-senders, attack compromise)."""
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
        if h % 2 == 0:  # content-mimicry attack
            label = 1
            compromised = (_h("c", r.profile_id, str(r.idx)) % 1000) / 1000.0 < attack_compromised
            addr = addrs[h % len(addrs)] if compromised else _addr(name, "gmail.com")
        else:           # genuine benign
            label = 0
            new_sender = (_h("n", r.profile_id, str(r.idx)) % 1000) / 1000.0 < benign_offgraph
            # New legitimate contact: a real-looking external address NOT in the graph.
            addr = _addr(f"new {name}", "partnerfirm.com") if new_sender else addrs[h % len(addrs)]
        rows.append({"profile_id": r.profile_id, "text": str(r.text), "label": label,
                     "sender_name": name, "sender_address": addr})
    return pd.DataFrame(rows)


def _aucs(df, real_of):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_SEED)
    tr, te = next(gss.split(df, groups=df["profile_id"]))
    train, test = df.iloc[tr], df.iloc[te]
    y = test["label"].to_numpy()
    pr_tr, pr_te = _prov_feats(train, real_of), _prov_feats(test, real_of)

    def fit(extra_tr, extra_te, text):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000).fit(train["text"]) if text else None
        def feats(d, ex):
            parts = ([vec.transform(d["text"])] if vec is not None else []) + ([csr_matrix(ex)] if ex is not None else [])
            return hstack(parts).tocsr() if len(parts) > 1 else parts[0]
        clf = LGBMClassifier(n_jobs=1, random_state=RANDOM_SEED, n_estimators=300, verbose=-1).fit(feats(train, extra_tr), train["label"])
        return clf.predict_proba(feats(test, extra_te))[:, 1]

    s_content = fit(None, None, True)
    s_prov = fit(pr_tr, pr_te, False)
    s_comb = fit(pr_tr, pr_te, True)
    # provenance FPR = benign flagged as off-graph (in_graph feature == 0)
    fp = float(np.mean([f[0] == 0.0 for f, lab in zip(pr_te, y) if lab == 0]))
    return auc(y, s_content), auc(y, s_prov), auc(y, s_comb), fp


def main() -> None:
    graph = _load_graph()
    real_of = lambda p: graph.get(p, {"addrs": set(), "names": set(), "domain": ""})
    rows = []
    for fb in BENIGN_OFFGRAPH:
        c, p, cp, fpr = _aucs(_build(fb, ATTACK_COMPROMISED), real_of)
        rows.append({"benign_offgraph": fb, "attack_compromised": ATTACK_COMPROMISED,
                     "content_auc": round(c, 4), "provenance_auc": round(p, 4),
                     "combined_auc": round(cp, 4), "provenance_fpr": round(fpr, 4)})
        print(f"  benign_offgraph={fb:.2f}: content={c:.3f} provenance={p:.3f} "
              f"combined={cp:.3f} prov_FPR={fpr:.3f}", flush=True)
    out = RESULTS_DIR / "exp_provenance_realistic.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["benign_offgraph", "attack_compromised", "content_auc",
                                          "provenance_auc", "combined_auc", "provenance_fpr"])
        w.writeheader(); w.writerows(rows)
    print(f"[provreal] wrote {out}  (attack_compromised={ATTACK_COMPROMISED})")


if __name__ == "__main__":
    main()
