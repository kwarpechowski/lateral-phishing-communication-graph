"""Waznosc warstw PER ZAGROZENIE (#GP-EXP-10): OSINT vs czas sa komplementarne.

Zagregowany AUC selekcji warstw maskuje, ze rozne warstwy adresuja rozne zagrozenia.
Tu rozdzielamy dwa zadania i liczymy leave-one-out waznosc warstw OSOBNO:
  * Zadanie A (nowi nadawcy / FP): benign = known/new_osint/new_cold, atak = bec_off.
    Tu OSINT odroznia legalnego nowego nadawce (na warstwie OSINT) od ataku off-graph;
    CZAS nie pomaga (nowy nadawca nie ma historii rytmu).
  * Zadanie B (przejete konta / FN): benign = known (w rytmie), atak = compromised (poza
    rytmem). Tu CZAS jest decydujacy; OSINT nie pomaga (przejety = znany kontakt na OSINT).

Pokazuje, ze ani OSINT, ani czas nie dominuja uniwersalnie -- sa KOMPLEMENTARNE.
Wyjscie: results/exp_layer_by_threat.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_osint_overlay import CATEGORIES, build_overlay      # noqa: E402
from graph.build_temporal_overlay import in_bucket, is_consistent, off_bucket  # noqa: E402

SEEDS = list(range(5))
OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
         "osint_event", "osint_pretext", "osint_platform", "interest_sim"]  # realne warstwy z c3-c8
TEMPORAL = "TEMPORAL"


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _adj(layers):
    out = {}
    for name, edges in layers.items():
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1]); nbr.setdefault(e[1], set()).add(e[0])
        out[name] = nbr
    return out


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prep(seed):
    base = build_layers()
    layers = {"contact": base["contact"], **{k: base[k] for k in OSINT}}
    adj = _adj(layers)
    cn = adj["contact"]
    ou = {}
    for L in OSINT:
        for t, nb in adj[L].items():
            ou.setdefault(t, set()).update(nb)
    twins = sorted(set(cn) | set(ou) | {r["twin_id"] for r in _self_rows()})
    return adj, twins, cn, ou


def _events(task, seed, twins, cn, ou):
    rows = []   # (sender, victim, label, bucket)
    for v in twins:
        c = sorted(cn.get(v, set()))
        o_only = sorted(ou.get(v, set()) - cn.get(v, set()) - {v})
        off = [t for t in twins if t != v and t not in cn.get(v, set()) and t not in ou.get(v, set())]
        if not c or not off:
            continue
        for k in range(20):
            h = _h(task, str(seed), v, str(k)); salt = f"{v}:{k}"
            if task == "A":                                  # nowi nadawcy (czas NIEINFORMATYWNY)
                if h % 2 == 0:                               # benign known/new_osint/new_cold
                    r = (h // 3) % 100
                    s = c[h % len(c)] if r < 40 else (o_only[h % len(o_only)] if (r < 70 and o_only) else off[h % len(off)])
                    rows.append((s, v, 0, in_bucket(s, salt, seed)))
                else:                                        # atak off-graph; ten SAM rozklad godzin co benign
                    s = off[h % len(off)]
                    rows.append((s, v, 1, in_bucket(s, salt, seed)))
            else:                                            # task B: przejecia
                s = c[h % len(c)]
                if h % 2 == 0:
                    rows.append((s, v, 0, in_bucket(s, salt, seed)))          # benign w rytmie
                else:
                    mimic = (h // 5) % 100 < 25
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    rows.append((s, v, 1, b))                                  # przejecie poza rytmem
    return rows


def _auc(adj, rows, twins, seed, graph_layers, temporal):
    y = np.array([r[2] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te
    X = []
    for s, v, _l, b in rows:
        per = [1.0 if s in adj[L].get(v, set()) else 0.0 for L in graph_layers]
        feat = per + [sum(per)]
        if temporal:
            feat.append(1.0 if is_consistent(s, b, seed) else 0.0)
        X.append(feat)
    X = np.array(X, np.float32)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=150, verbose=-1).fit(X[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])


def group_ablation(task):
    """AUC dla: full / bez OSINT / bez czasu / bez kontaktow (grupowo)."""
    full = ["contact"] + OSINT
    acc = {k: [] for k in ["full", "bez_OSINT", "bez_czasu", "bez_kontaktow", "tylko_OSINT", "tylko_czas"]}
    for seed in SEEDS:
        adj, twins, cn, ou = _prep(seed)
        rows = _events(task, seed, twins, cn, ou)
        acc["full"].append(_auc(adj, rows, twins, seed, full, True))
        acc["bez_OSINT"].append(_auc(adj, rows, twins, seed, ["contact"], True))
        acc["bez_czasu"].append(_auc(adj, rows, twins, seed, full, False))
        acc["bez_kontaktow"].append(_auc(adj, rows, twins, seed, OSINT, True))
        acc["tylko_OSINT"].append(_auc(adj, rows, twins, seed, OSINT, False))
        acc["tylko_czas"].append(_auc(adj, rows, twins, seed, [], True))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    out = RESULTS / "exp_layer_by_threat.csv"
    rowsout = []
    for task, lab in (("A", "nowi_nadawcy"), ("B", "przejecia")):
        a = group_ablation(task)
        print(f"\n[threat={lab}]")
        for k in ["full", "bez_OSINT", "bez_czasu", "tylko_OSINT", "tylko_czas"]:
            print(f"   {k:14s} AUC={a[k]:.3f}")
            rowsout.append({"task": lab, "config": k, "auc": round(a[k], 4)})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "config", "auc"]); w.writeheader(); w.writerows(rowsout)
    print(f"\n[layer-by-threat] wrote {out}")
    print("Oczekiwane: A(nowi)->'bez_OSINT' duzy spadek, 'bez_czasu' bez zmian; "
          "B(przejecia)->odwrotnie.")


if __name__ == "__main__":
    main()
