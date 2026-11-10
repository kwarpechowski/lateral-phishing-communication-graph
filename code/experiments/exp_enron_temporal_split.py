"""Enron z PODZIALEM CZASOWYM rytmu (#GP-EXP-29, recenzja P1 pkt 2 -- wyciek rytmu).

Zarzut: w exp_enron_temporal/multiplex rytm `active[s]` liczono na CALEJ historii nadawcy,
a te same wiadomosci stawaly sie zdarzeniami benign -> benign jest "w rytmie" Z KONSTRUKCJI
(wyciek etykiety, odrebny od audytowanego odsetka off-hours). Tu naprawiamy to wprost:

  * sortujemy wiadomosci kazdego nadawcy CHRONOLOGICZNIE (po realnej dacie),
  * rytm `active_early[s]` estymujemy WYLACZNIE na pierwszych SPLIT_FRAC wiadomosci,
  * zdarzenia ewaluacyjne losujemy WYLACZNIE z pozostalych (pozniejszych) wiadomosci.

Cecha tc = "czy bucket pozniejszej wiadomosci nalezy do rytmu z wczesniejszego okna" jest
teraz PROGNOZA poza proba, nie tautologia. Raportujemy obok siebie:
  * leaky : rytm z calej historii (jak w pracy) -- reprodukcja,
  * split : rytm z wczesniejszego okna (bez wycieku) -- uczciwy wynik.
Dla warstwy czasowej-samej i pelnego multipleksu strukturalno-czasowego.

Uruchom z katalogu code/. Wyjscie: results/exp_enron_temporal_split.csv
"""

from __future__ import annotations

import csv
import email
import hashlib
from collections import defaultdict
from email.utils import parsedate_to_datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ENRON = Path(
    "../../projects/personalized-phishing-defense/code/data/enron/maildir"
).resolve()
RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_USERS = 60
MAX_MSGS = 400
MIN_SENT = 25  # min. wiadomosci nadawcy (by podzial mial sens)
N_BUCKETS = 28
ACTIVE_MIN = 2
MAX_RECIP = 8
SPLIT_FRAC = (
    0.6  # pierwsze 60% (po czasie) -> estymacja rytmu; ostatnie 40% -> ewaluacja
)
SEEDS = list(range(20))
FPR = 0.01


def _bucket(dt):
    return (dt.weekday() * 4 + dt.hour // 6) % N_BUCKETS


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def parse():
    """sender -> chronologicznie posortowana lista (recipient, bucket, epoch).

    LEAK-AWARE (fix po recenzji zewnetrznej): zwrocone sa takze warstwy strukturalne
    policzone z Wczesnego okna (60%): contact_early, corecip_early. Pelnohistoryczne
    warstwy zawyzaly wynik definicyjnie: legalne zdarzenia ewaluacyjne to prawdziwe
    wiadomosci, ktore z definicji sa kontaktami nadawcy -> contacts-only mialo 0.75
    z samej konstrukcji. Domena zostaje statyczna (adres = tozsamosc, bez historii)."""
    sent = defaultdict(list)
    corecip = defaultdict(set)
    groups = []  # (frm, epoch, frozenset odbiorcow) dla wspolodbiorcow
    domain = {}
    users = sorted([d for d in ENRON.iterdir() if d.is_dir()])[:MAX_USERS]
    for ud in users:
        n = 0
        for f in ud.rglob("*"):
            if not f.is_file() or n >= MAX_MSGS:
                continue
            try:
                msg = email.message_from_bytes(f.read_bytes())
            except Exception:
                continue
            frm = (msg.get("From") or "").strip().lower()
            to = (msg.get("To") or "") + "," + (msg.get("Cc") or "")
            date = msg.get("Date")
            if not frm or "@" not in frm or not date:
                continue
            try:
                dt = parsedate_to_datetime(date)
                b = _bucket(dt)
                ep = dt.timestamp()
            except Exception:
                continue
            recips = [r.strip().lower() for r in to.split(",") if "@" in r]
            for r in recips:
                if r != frm:
                    sent[frm].append((r, b, ep))
            domain.setdefault(frm, frm.split("@")[-1])
            for r in recips:
                domain.setdefault(r, r.split("@")[-1])
            if 2 <= len(recips) <= MAX_RECIP:
                rset = frozenset(r for r in recips if r != frm)
                if rset:
                    groups.append((frm, ep, rset))
                for a, c in combinations(sorted(set(recips)), 2):
                    corecip[a].add(c)
                    corecip[c].add(a)
            n += 1
    sent = {
        s: sorted(v, key=lambda x: x[2]) for s, v in sent.items() if len(v) >= MIN_SENT
    }
    contact = {s: {r for r, _b, _e in v} for s, v in sent.items()}

    active_full, active_early, eval_msgs = {}, {}, {}
    contact_early, corecip_early = {}, defaultdict(set)
    early_ep = {}
    for s, v in sent.items():
        cut = int(len(v) * SPLIT_FRAC)
        early, late = v[:cut], v[cut:]
        early_ep[s] = early[-1][2] if early else -1.0
        contact_early[s] = {r for r, _b, _e in early}

        def buckets(msgs):
            cnt = defaultdict(int)
            for _r, b, _e in msgs:
                cnt[b] += 1
            return {b for b, c in cnt.items() if c >= ACTIVE_MIN}

        active_full[s] = buckets(v)
        active_early[s] = buckets(early)
        eval_msgs[s] = late
    for frm, ep, rset in groups:  # wspolodbiorcy tylko z wczesnych maili nadawcy
        if ep <= early_ep.get(frm, -1.0):
            for a, c in combinations(sorted(rset), 2):
                corecip_early[a].add(c)
                corecip_early[c].add(a)
    # tylko nadawcy z niepustym rytmem wczesnym i niepusta proba pozna
    keep = {s for s in sent if active_early.get(s) and eval_msgs.get(s)}
    sent = {s: sent[s] for s in keep}
    return (
        sent,
        eval_msgs,
        active_full,
        active_early,
        contact,
        corecip,
        domain,
        contact_early,
        corecip_early,
    )


def _recall_at_fpr(y, s, t=FPR):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _events(eval_msgs, active_rhythm, contact, seed, off_rate=0.0):
    """Zdarzenia ewaluacyjne WYLACZNIE z poznego okna; rytm z `active_rhythm` (early albo full).

    off_rate: frakcja benign przeniesiona POZA rytm (realny ruch off-hours) -- gdy >0, laczymy
    DRUGA kontrole wycieku (off-hours) z podzialem czasowym jednoczesnie."""
    senders = sorted(eval_msgs)
    rows = []
    for s in senders:
        act = sorted(active_rhythm.get(s, set())) or list(range(N_BUCKETS))
        off_bkts = sorted(set(range(N_BUCKETS)) - set(act)) or act
        late = eval_msgs[s]
        recips = [r for r, _b, _e in late]
        non_contacts = [x for x in senders if x != s and x not in contact.get(s, set())]
        for i, (r, b, _e) in enumerate(late):
            h = _h(str(seed), s, str(i))
            t = h % 4

            def benign_bucket(bb, hh):
                if off_rate and (hh // 13) % 100 < off_rate * 100:
                    return off_bkts[hh % len(off_bkts)]  # legalny ruch off-hours
                return bb

            if t == 0:  # benign: realna pozna wiadomosc
                rows.append((s, r, 0, benign_bucket(b, h)))
            elif t == 1:  # przejecie: znany odbiorca, 25% mimikry
                bb = (
                    off_bkts[h % len(off_bkts)]
                    if (h // 5) % 100 >= 25
                    else act[h % len(act)]
                )
                rows.append((s, r, 1, bb))
            elif t == 2 and non_contacts:  # off-graph
                rows.append((non_contacts[h % len(non_contacts)], r, 1, b))
            else:  # benign: inny realny odbiorca
                rows.append((s, recips[h % len(recips)], 0, benign_bucket(b, h)))
    return rows


def _feat(rows, contact, corecip, domain, active_rhythm, full):
    X = []
    for s, v, _l, b in rows:
        c = 1.0 if (v in contact.get(s, set()) or s in contact.get(v, set())) else 0.0
        if not full:
            X.append([1.0, 1.0 if b in active_rhythm.get(s, set()) else 0.0])
            continue
        cc = 1.0 if v in corecip.get(s, set()) else 0.0
        dm = 1.0 if domain.get(s) and domain.get(s) == domain.get(v) else 0.0
        tc = 1.0 if b in active_rhythm.get(s, set()) else 0.0
        X.append([c, cc, dm, tc, c + cc + dm])
    return np.array(X, np.float32)


def _eval(rows, contact, corecip, domain, active_rhythm, seed, contact_e=None, corecip_e=None):
    # LEAK-AWARE: w trybach split warstwy strukturalne pochodza z wczesnego okna
    contact_e = contact_e if contact_e is not None else contact
    corecip_e = corecip_e if corecip_e is not None else corecip
    y = np.array([r[2] for r in rows])
    snd = [r[0] for r in rows]
    order = sorted(set(snd))
    te_s = set(
        np.array(order)[
            np.random.default_rng(seed).permutation(len(order))[
                : max(1, len(order) // 3)
            ]
        ]
    )
    te = np.array([x in te_s for x in snd])
    tr = ~te
    out = {}
    for tag, full in (("temporal", False), ("full", True)):
        X = _feat(rows, contact_e if full else contact, corecip_e if full else corecip,
                            domain, active_rhythm, full)
        clf = LGBMClassifier(
            n_jobs=1, random_state=42, n_estimators=150, verbose=-1
        ).fit(X[tr], y[tr])
        sc = clf.predict_proba(X[te])[:, 1]
        out[f"{tag}_auc"] = roc_auc_score(y[te], sc)
        # czułości przy FPR 1%, 2% i 5% — binarna cecha tc daje funkcję schodkową:
        # przy rytmie z pełnej historii próg mieści się w FPR<=1%, przy podziale czasowym
        # żaden próg binarnego wyniku nie spełnia FPR<=1%, ale 2%/5% już tak (recenzja r2 B, pkt 3)
        for t in (0.01, 0.02, 0.05):
            out[f"{tag}_r{int(t * 100)}"] = _recall_at_fpr(y[te], sc, t)
    return out


def main():
    if not ENRON.exists():
        print(f"[split] brak maildira {ENRON}")
        return
    print("[split] parsowanie Enrona (chronologicznie)...", flush=True)
    (sent, eval_msgs, active_full, active_early, contact, corecip, domain,
     contact_early, corecip_early) = parse()
    print(
        f"[split] nadawcow z podzialem={len(sent)} (SPLIT_FRAC={SPLIT_FRAC})",
        flush=True,
    )

    OFF = 0.175  # 15-20% legalnej poczty off-hours (srodek przedzialu) dla kontroli laczonej
    acc = defaultdict(lambda: defaultdict(list))
    for seed in SEEDS:
        # leaky: rytm z calej historii (reprodukcja); split: rytm z wczesnego okna (bez wycieku);
        # split+oh: OBIE kontrole naraz (podzial czasowy + ~17.5% off-hours) -> najuczciwsza liczba.
        for mode, rhythm, orate, cl, cc in (
            ("leaky", active_full, 0.0, contact, corecip),            # pelne warstwy + rytm z calosci
            ("split", active_early, 0.0, contact_early, corecip_early),  # warstwy I rytm z wczesnych 60%
            ("split+oh", active_early, OFF, contact_early, corecip_early),
        ):
            rows = _events(eval_msgs, rhythm, contact, seed, orate)
            for k, v in _eval(rows, cl, cc, domain, rhythm, seed,
                              contact_e=cl, corecip_e=cc).items():
                acc[mode][k].append(v)
    out = RESULTS / "exp_enron_temporal_split.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "metric", "mean", "n_seeds"])
        for mode in ("leaky", "split", "split+oh"):
            for k, v in acc[mode].items():
                w.writerow([mode, k, round(float(np.mean(v)), 4), len(v)])
    print(
        f"[split] Enron, rytm z calej historii (leaky) vs podzial czasowy (split) vs +off-hours:"
    )
    for mode in ("leaky", "split", "split+oh"):
        a = {k: float(np.mean(v)) for k, v in acc[mode].items()}
        print(
            f"  [{mode:9s}] temporal AUC={a['temporal_auc']:.3f} R@1%={a['temporal_r1']:.3f} | "
            f"full AUC={a['full_auc']:.3f} R@1%={a['full_r1']:.3f}"
        )
    print(f"[split] wrote {out}")


if __name__ == "__main__":
    main()
