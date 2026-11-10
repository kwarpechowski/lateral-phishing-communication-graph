# Lateral Phishing as a Violation of the Organizational Communication Graph

Repozytorium dwóch powiązanych artykułów o detekcji phishingu ukierunkowanego
na grafie komunikacji organizacji + pakiet reprodukcyjny eksperymentów.

| Katalog | Zawartość |
|---|---|
| `A-Journal/` | Artykuł A (journal, Elsevier CAS): kaskadowy lateral phishing, temporal GNN vs detektory wolumetryczne, kontrola time-shuffle, walidacja Enron + SNAP |
| `B-Conference/` | Artykuł B (konferencja IEEE): multiplex grafu wiedzy domenowej, cross-layer inconsistency, rhythm-leak pitfall na Enronie |
| `code/` | Pakiet reprodukcyjny: generator populacji, warstwy grafu, detektory, skrypty eksperymentów i **przeliczone wyniki CSV** (`code/results/`) |

Wszystkie liczby w artykułach pochodzą wprost z plików `code/results/*.csv`.

## Odkrycia i wyniki

### Artykuł A (JNCA) — lateral phishing jako naruszenie grafu komunikacji w czasie

- **Hierarchia detektorów** (kaskada, graf 150 kont): lokalne cechy 1-hop są bezużyteczne (AUC 0.70),
  statyczny GNN 0.76, ręczny kontekst 0.81, **temporal GNN 0.88** — jedyny detektor z realną czujnością
  operacyjną (0.121 przy FPR 1%). Tasowanie kolejności zdarzeń zabiera mu przewagę (→ 0.73), więc
  sygnał jest dynamiczny, nie topologiczny.
- **Atak rozłożony w czasie (stealth)**: detektor wolumetryczny kładzie się (0.51 przy K=1, czujność 0.00),
  temporal trzyma się powyżej losowości (0.59). Kluczowa statystyka: temporal vs volume **+0.105 AUC,
  p<0.001, δ=0.89**; temporal vs kontekst ręczny — **brak istotnej różnicy (p=0.62)**: sygnał odbiorczy
  czytają równie dobrze model i cechy ręczne.
- **Sam bit rytmu nie wnosi nic** w tym harnessie (rhythm-only 0.51; poczta poza rytmem: ataki 78% ≈
  legalna 79%) — sygnałem jest fala po stronie *odbiorców*, nie rytm nadawcy.
- **Granica modelu zagrożeń**: sygnał wymaga fan-outu ≥ 2 — atak chirurgiczny na jednego odbiorcę jest
  niewidoczny dla wszystkich detektorów (~0.50). Propagacja (kaskada) nie jest konieczna — wystarcza
  synchroniczna wysyłka do kilku odbiorców (volume 0.69–0.84, temporal 0.68–0.93 przy p_inf=0).
- **Walidacja na prawdziwych topologiach**: Enron 1990 kont (temporal 0.83, tasowanie → 0.75) i dwa grafy
  SNAP — przewaga temporal nad volume największa pod stealth (+0.16/+0.19 AUC); po uczciwej zasadzie
  liczenia cech volume nie wygrywa już nawet blast.
- **Granice operacyjne**: przy realnej częstości ataku (1:10⁴) na każdy wykryty atak przypada ~830 fałszywych
  alarmów → narzędzie triage dla analityków, nie autonomiczna obrona. Szum etykiet najszybciej degraduje
  model nauczony (temporal 0.73→0.60 przy 20% szumu).

### Artykuł B (IEEE) — multiplex grafu wiedzy i rhythm-leak pitfall

- **Multiplex + rytm**: AUC 0.70→0.87 i likwidacja 100% fałszywych alarmów na legalnych nowych nadawcach.
  Uczciwa kontrola (tasowanie **tylko** warstw OSINT) pokazuje jednak, że zysk niosą kontakty+rytm, a OSINT
  dokłada tylko ~0.03 AUC — wartość metody leży po stronie fałszywych alarmów, nie wykrywania podszywek
  (zimna impersonacja: czujność 0.04 przy FPR 1%).
- **Odkrycie metodyczne — rhythm-leak pitfall**: rytm nadawcy liczony na pełnej historii zawyża czujność
  detektora rytmu do 0.73 przy FPR 1%; uczciwy podział czasowy (rytm z wczesnych 60%) daje **0.00 nawet przy
  FPR 5%**, bo ponad 5% legalnej poczty wypada poza uczciwie policzonym rytmem. Pełny multiplex trzyma
  czujność 0.51 (AUC 0.84). Ostrzeżenie dla całej klasy detektorów behawioralnych.
- **Susceptibility (miara detektorowa, nie ludzka)**: ekspozycję ofiary napędza **kanał ataku, nie ranga** —
  zimna impersonacja zostawia strukturalną podłogę Susc≈0.96 na każdym stanowisku; w kanale lateral
  executive jest najbardziej narażony (0.924) — bogatsze osadzenie w grafie nie kupuje ochrony.
- **Fuzja graf+treść**: treść i graf zawodzą na różnych atakach (treść widzi atak z błędem treściowym 0.997,
  ale jest ślepa na mimikrię 0.50; graf odwrotnie 0.86), a fuzja trzyma się blisko lepszego z nich w każdym
  reżimie — argument za systemem hybrydowym.

## Uruchomienie eksperymentów

### Wymagania

- Python 3.12 (wystarczy CPU; GPU niewymagany)
- Pakiety: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `torch`, `torch-geometric`
  (wersje zbliżone do: lightgbm 4.6, scikit-learn 1.8, torch 2.x, torch-geometric 2.x)
- Korpus Enron (maildir) — publiczny, wypakowany poza repozytorium;
  domyślna ścieżka w skryptach: `../projects/personalized-phishing-defense/code/data/enron/maildir`
  (do zmiany w nagłówkach plików `code/experiments/exp_enron_*.py` oraz `code/experiments/exp_cascade_enron.py`)
- Grafy SNAP (`email-Eu-core`, `CollegeMsg`) — **dołączone** w `code/data/realgraphs/`

### Mapa: tabela w artykule → skrypt → wynikowy CSV

| Tabela / figura | Skrypt (`code/experiments/`) | CSV (`code/results/`) |
|---|---|---|
| A: hierarchia detektorów (Tab. 1) | `exp_cascade.py` | `exp_cascade.csv` |
| A: library TGN (wiersz Tab. 1) | `exp_cascade_tgn.py` | `exp_cascade_tgn.csv` |
| A: stealthy attacker (Tab. 2) + audyt rytmu | `exp_cascade_stealth.py` | `exp_cascade_stealth.csv`, `exp_cascade_stealth_tc.csv` |
| A: baselines (a)/(b) (Tab. 3) | `exp_cb_stealth.py [surgical]` | `cb_hopper_stealth.csv`, `cb_surgical_fanout.csv` |
| A: label noise (Tab. S4) | `exp_cb_stealth.py labelnoise` | `cb_labelnoise.csv` |
| A: sweep p_inf (Tab. S2) | `exp_cascade_sweep.py` | `exp_cascade_sweep.csv` |
| A: walidacja indukcyjna (Tab. S3) | `exp_cascade.py org [scaleN=…]` | `exp_cascade_org*.csv` |
| A: Enron (Tab. 4a) | `exp_cascade_enron.py` | `exp_cascade_enron.csv` |
| A: SNAP (Tab. 4b) | `exp_cascade_realgraphs.py` | `exp_cascade_realgraphs.csv` |
| A: kontrasty statystyczne | `cascadebench.experiments.headline_ci` | `cb_headline_ci.csv`, `cb_headline_contrasts.csv` |
| B: multiplex vs contacts (Tab. inv1) | `exp_multiplex.py` | `exp_multiplex.csv` |
| B: ablacja warstw | `exp_extended.py` | `exp_ablation.csv` |
| B: **kontrola shuffle tylko OSINT** | `exp_osint_shuffle.py` | `exp_osint_shuffle.csv` |
| B: rhythm-leak @FPR 1/2/5% (Tab. enronsplit) | `exp_enron_temporal_split.py` | `exp_enron_temporal_split.csv` |
| B: spójność jawna vs konkatenacja | `exp_consistency.py` | `exp_consistency.csv` |
| B: susceptibility (Tab. susc) | `exp_susceptibility.py` | `exp_susceptibility.csv` |
| B: fusion graf+treść | `exp_content_fusion.py` | `exp_content_fusion.csv` |

Pełna mapa: `code/README.md`. Figury: `code/analysis/plot_cascade.py` i `plot_figures.py`
(czytają `code/results/*.csv`, piszą PDF-y figur).

### Komendy

```bash
cd code

# pojedynczy eksperyment (np. stealth, 20 seedów, ~2 h CPU):
python experiments/exp_cascade_stealth.py

# wszystkie eksperymenty po fixach (4 równoległe ścieżki):
bash experiments/run_reruns.sh lane1   # exp_cascade + sweep + arch
bash experiments/run_reruns.sh lane2   # stealth + baselines + labelnoise
bash experiments/run_reruns.sh lane3   # OSINT-shuffle + Enron + fpr
bash experiments/run_reruns.sh lane4   # kontrasty statystyczne

# figury z nowych CSV:
python analysis/plot_cascade.py
```

Odtwarzalność: eksperymenty są deterministyczne (nasılıwańe `seed`), z dwoma zastrzeżeniami:
liczba wątków torch może dawać różnice rzędu ±0.01 AUC między maszynami, a LightGBM
zaleca się `n_jobs=1` (przy wielu wątkach na Windows trening bywa o rzędy wielkości wolniejszy).

## Kompilacja artykułów

Kompilacja przez tectonic w Docker (bez instalowania LaTeX-a):

```bash
docker run --rm -v "$PWD/A-Journal://work" -w //work fabianhauser/tectonic main.tex
docker run --rm -v "$PWD/A-Journal://work" -w //work fabianhauser/tectonic supplementary.tex
docker run --rm -v "$PWD/B-Conference://work" -w //work fabianhauser/tectonic main.tex
```

## Naprawione wycieki danych (leak-aware)

Wcześniejsza wersja pakietu miała trzy usterki, usunięte przed przeliczeniem WSZYSTKICH wyników:

1. **momentum** (baseline path-based) liczył wysyłki z przyszłych kubełków `[b, b+W]` → okno przeszłe `[b−W, b−1]`
2. **liczniki volume/COMPA** liczone na całym strumieniu (w tym zbiór testowy) → tylko kubełki `< b`
3. **cechy węzłów GNN** ze stopni liczonych na całym strumieniu → tylko ze splitu treningowego

Zasada po fixie: każda cecha strumieniowa liczona wyłącznie z przeszłych kubełków (≤ b−1),
zgodnie z pamięcią modelu temporalnego. Szczegóły: komentarze `% ZMIANA` w plikach `.tex`.

Korpus Enron jest publiczny (Klimt & Yang, 2004); nie jest dołączony do repozytorium
ze względu na rozmiar — wystarczy wypakować oryginalny maildir i wskazać ścieżkę.
