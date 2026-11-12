# cascadebench — leak-aware poligon adwersaryjny dla grafowych detektorów phishingu

Kontrolowane, syntetyczne **środowisko testowe** do oceny grafowych detektorów phishingu i lateral
movement wobec **adaptacyjnych atakujących**, z ground-truth etykietami i **leak-aware** ewaluacją.
Realne dane lateral phishingu są zastrzeżone i pełne konfundentów (wycieki rytmu/stopnia); poligon
daje sterowalność, ground-truth i kontrolę konfundentów, których realne korpusy nie zapewniają.

## Instalacja
Wymaga: `numpy`, `scikit-learn`, `lightgbm`, `torch` (CPU OK). Uruchamiać przez venv projektu.
```bash
./venv/Scripts/python -m cascadebench demo
```

## Szybki start (API)
```python
import cascadebench as cb

g  = cb.Graph.synthetic(800)                      # lub cb.load("email-Eu-core") / "enron"
sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=8, spread=2), seed=0)
res = cb.evaluate(sc, cb.get_panel(), seed=0)     # AUC + Recall@{0.1%,0.5%,1%,5%}
```

## Komponenty
| Moduł | Rola |
|------|------|
| `graph.py` | grafy: `Graph.synthetic(N)` (proceduralny), `Graph.from_edgelist` (SNAP), `Graph.enron` |
| `attack.py` | `CascadeStrategy` (fan-out=zasięg; spread/mimicry/fabrication=ewazja), `build_scenario` |
| `detect.py` | panel detektorów ze wspólnym interfejsem (1-hop, COMPA, GCN-stat, kontekst, temporalny) |
| `features.py` / `models.py` | cechy i lekkie modele grafowe (czysty torch) |
| `evaluate.py` | **leak-aware**: split po ofiarach/org, Recall@niski-FPR, kontrola shuffle, audyt off-hours |
| `experiments.py` | reprodukowalne: `pareto`, `adaptive`, `panel`, `leak_audit` |

## CLI
```bash
python -m cascadebench pareto    --graph synthetic:600 --seeds 3   # front ewazja↔zasięg
python -m cascadebench adaptive  --graph synthetic:600 --fanout 5  # adaptacyjny atak przy stałym zasięgu
python -m cascadebench panel     --graph email-Eu-core            # panel pod strategiami
python -m cascadebench leak-audit --graph enron                   # audyt wycieku rytmu
```

## Rozszerzanie
Nowy detektor = podklasa `Detector` z `fit_score(scenario, train_mask, test_mask, seed)`.
Nowa strategia ataku = pola w `CascadeStrategy` (lub własna funkcja generująca zdarzenia).
Nowy graf = `Graph.from_edgelist(path)` (format `src dst [timestamp]`).

## Leak-aware (czym różni się od istniejących benchmarków)
- **matched-design benign** — ten sam rozkład końców/stopni co atak (usuwa konfundent tożsamości/stopnia),
- **audyt off-hours** — sprawdza, czy wynik nie jedzie na założeniu „benign zawsze w rytmie",
- **kontrola shuffle** — przetasowanie czasu musi zniszczyć przewagę (dowód przyczynowości),
- **metryki operacyjne** — Recall przy niskim FPR, nie tylko AUC.

Cytować: (praca P3 — w przygotowaniu).
