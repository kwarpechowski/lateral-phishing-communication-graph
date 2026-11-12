"""EXP — provenance/BEC on the SYNTHETIC interconnected-twin organization graph.

Coherence fix: runs the content-mimicry / BEC provenance test on the TWINS' OWN
communication graph (``data/org_graph`` -> ``results/twin_network.csv``) instead of
borrowing Enron's. The same synthetic population thus drives both the
content-personalization experiments and the provenance one --- the twins genuinely
know each other (coworkers + cross-firm partners are other twins).

Design mirrors the realistic Enron experiment (``exp_provenance_realistic``) so the
result is honest, not a synthetic perfect-separability artifact:

  * benign (0): the twin's real benign text, sent by one of its KNOWN contacts
    (another twin). With probability ``NEW_SENDER_RATE`` it instead comes from a NEW
    legitimate external sender (off-graph but benign) --- a realistic false-positive
    source (first-time correspondents).
  * BEC attack (1): the SAME benign text with a spoofed sender. With probability
    ``COMPROMISED_RATE`` the sender is a COMPROMISED known contact (in-graph but
    malicious) --- a realistic false-negative source; otherwise an off-graph
    look-alike/external address.

The ONLY provenance feature is victim-specific: ``sender in THIS twin's contact set``.
A shuffled control evaluates each sender against the WRONG twin's contacts; if the
signal were a victim-independent artifact (e.g. free-mail flag) the shuffle would not
fall to chance. We report multi-seed AUC, the operational Recall@FPR=1\%, and the
shuffled control.

Output: results/exp_provenance_org.csv.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis.metrics import auc, recall_at_fpr
from config import RESULTS_DIR
from data.build_provenance import _addr

SEEDS = list(range(8))
NEW_SENDER_RATE = 0.20      # benign from a first-time (off-graph) legit sender
COMPROMISED_RATE = 0.25     # BEC from a compromised in-graph contact (blind to provenance)
LOOKALIKE = "external-partner-mail.com"


def _h(*p: str) -> int:
    return int(hashlib.sha256(":".join(p).encode()).hexdigest(), 16)


def _load_graph():
    import pandas as pd
    net = pd.read_csv(RESULTS_DIR / "twin_network.csv")
    g: dict[str, dict] = {}
    for tid, grp in net.groupby("twin_id"):
        g[tid] = {"addrs": list(grp["contact_address"]), "names": list(grp["contact_name"])}
    return g


def main() -> None:
    import pandas as pd

    if not (RESULTS_DIR / "twin_network.csv").exists():
        raise FileNotFoundError("twin_network.csv missing — run `python orchestrator.py org` first.")
    graph = _load_graph()
    ben = pd.read_csv(RESULTS_DIR / "benign_corpus.csv").reset_index().rename(columns={"index": "idx"})
    ben = ben[ben["profile_id"].isin(graph)].reset_index(drop=True)
    twins = sorted(graph)
    # Shuffled control maps each twin to the SAME role slot in a DIFFERENT firm
    # (offset by the 8-person org size): a proper victim-specificity test, since
    # colleagues in one firm share almost the same contact set (the signal is
    # org-level). A cross-firm victim has a disjoint graph, so a genuine
    # victim-specific signal must fall toward chance here.
    wrong = {twins[i]: twins[(i + 8) % len(twins)] for i in range(len(twins))}

    def build(seed: int):
        rows = []
        for r in ben.itertuples():
            t = r.profile_id
            cs = graph[t]
            if not cs["addrs"]:
                continue
            h = _h(t, str(r.idx), str(seed))
            ci = h % len(cs["addrs"])
            cname, caddr = cs["names"][ci], cs["addrs"][ci]
            if h % 2 == 0:                                       # benign
                if (h // 3) % 100 < int(NEW_SENDER_RATE * 100):  # new legit sender (off-graph)
                    sender = _addr(f"New Contact {h % 997}", LOOKALIKE)
                else:                                            # known contact (in-graph)
                    sender = caddr
                rows.append((str(r.text), 0, sender, t))
            else:                                                # BEC attack
                if (h // 5) % 100 < int(COMPROMISED_RATE * 100):  # compromised contact (in-graph!)
                    sender = caddr
                else:                                            # off-graph look-alike spoof
                    sender = _addr(cname, LOOKALIKE)
                rows.append((str(r.text), 1, sender, t))
        return rows

    def feat(sender: str, victim_addrs) -> float:
        return 1.0 if sender in victim_addrs else 0.0           # ONLY victim-specific signal

    acc = {k: [] for k in ["content_auc", "prov_auc", "comb_auc", "shuf_auc",
                           "prov_recall1", "prov_fpr"]}
    for seed in SEEDS:
        rows = build(seed)
        texts = [r[0] for r in rows]; y = np.array([r[1] for r in rows])
        senders = [r[2] for r in rows]; vics = [r[3] for r in rows]
        rng = np.random.default_rng(seed)
        te_twins = set(np.array(twins)[rng.permutation(len(twins))[:max(1, int(0.3 * len(twins)))]])
        te = np.array([v in te_twins for v in vics]); tr = ~te

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000).fit(
            [t for t, m in zip(texts, tr) if m])
        Xtr_txt, Xte_txt = vec.transform([t for t, m in zip(texts, tr) if m]), \
            vec.transform([t for t, m in zip(texts, te) if m])
        real = np.array([[feat(senders[i], graph[vics[i]]["addrs"])] for i in range(len(rows))], dtype=np.float32)
        shuf = np.array([[feat(senders[i], graph[wrong[vics[i]]]["addrs"])] for i in range(len(rows))], dtype=np.float32)
        ytr, yte = y[tr], y[te]

        def fit(Xtr, Xte):
            clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1).fit(Xtr, ytr)
            return clf.predict_proba(Xte)[:, 1]

        s_c = fit(Xtr_txt, Xte_txt)
        s_p = fit(csr_matrix(real[tr]), csr_matrix(real[te]))
        s_m = fit(hstack([Xtr_txt, csr_matrix(real[tr])]).tocsr(), hstack([Xte_txt, csr_matrix(real[te])]).tocsr())
        s_s = fit(csr_matrix(shuf[tr]), csr_matrix(shuf[te]))
        # provenance FPR: benign whose sender is off the (real) graph (new senders)
        fpr = float(np.mean([real[te][j][0] == 0.0 for j in range(len(yte)) if yte[j] == 0]))
        acc["content_auc"].append(auc(yte, s_c))
        acc["prov_auc"].append(auc(yte, s_p))
        acc["comb_auc"].append(auc(yte, s_m))
        acc["shuf_auc"].append(auc(yte, s_s))
        acc["prov_recall1"].append(recall_at_fpr(yte, s_p, fpr_target=0.01))
        acc["prov_fpr"].append(fpr)
        print(f"  seed={seed}: content={acc['content_auc'][-1]:.3f} prov={acc['prov_auc'][-1]:.3f} "
              f"shuf={acc['shuf_auc'][-1]:.3f} R@1%={acc['prov_recall1'][-1]:.3f} fpr={fpr:.3f}", flush=True)

    out = RESULTS_DIR / "exp_provenance_org.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "mean", "ci95", "n_seeds"])
        for k, v in acc.items():
            a = np.asarray(v); ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
            w.writerow([k, round(float(a.mean()), 4), round(float(ci), 4), len(a)])
    print(f"\n[org-prov] wrote {out}")
    for k, v in acc.items():
        a = np.asarray(v); ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        print(f"  {k:14s}: {a.mean():.3f} ± {ci:.3f}")


if __name__ == "__main__":
    main()
