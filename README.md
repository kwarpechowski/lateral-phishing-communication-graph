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
  operacyjną (0.120 przy FPR 1%). Tasowanie kolejności zdarzeń zabiera mu przewagę (→ 0.72), więc
  sygnał jest dynamiczny, nie topologiczny.
- **Atak rozłożony w czasie (stealth)**: detektor wolumetryczny nie ma sygnału (0.50 przy K=1, czujność
  0.00) — liczniki COMPA są przyczynowo-zdarzeniowe (widzą wczesniejsze zdarzenia bieżącego kubełka), więc
  widzą burst w blast (0.63 na grafie case study, 0.83 na SNAP), ale jedno wysłanie na kubełek zostawia
  każde okno przyczynowe puste. Temporal trzyma się powyżej (0.57 przy K=1). Kluczowa statystyka:
  temporal vs volume **+0.104 AUC, p<0.001, δ=0.96**; temporal vs kontekst ręczny — **brak istotnej różnicy
  (−0.008, p=0.39, w obu protokołach benign)**: sygnał odbiorczy czytają równie dobrze model i cechy ręczne.
- **Sam bit rytmu nie wnosi nic** w tym harnessie (rhythm-only 0.51; poczta poza rytmem: ataki 78% ≈
  legalna 79%) — sygnałem jest fala po stronie *odbiorców*, nie rytm nadawcy.
- **Granica modelu zagrożeń**: sygnał wymaga fan-outu ≥ 2 — atak chirurgiczny na jednego odbiorcę jest
  niewidoczny dla wszystkich detektorów (~0.50). Propagacja (kaskada) nie jest konieczna — wystarcza
  synchroniczna wysyłka do kilku odbiorców (volume 0.70–0.83, temporal 0.71–0.93 przy p_inf=0). Ablacja hoppera bez cechy precursor
  (hopper-bez-precursor) pokazuje, że to precursor (≈ freshness kontekstu ręcznego) nosi jego sygnał
  stealth: 0.72→0.61 przy K=1. Trudne negatywy (łagodne fale legalne, 30% liczby ataków, w rytmie)
  prawie nic nie kosztują: ≤0.013 AUC każdego detektora.
- **Walidacja na prawdziwych topologiach**: Enron ~1970 kont w OBU reżimach — blast: temporal 0.82
  (tasowanie → 0.73, hierarchia jak na grafie case study); stealth: kontekst ręczny 0.88 (czujność 0.28!),
  temporal 0.86, 1-hop 0.60. Na SNAP reżim stealth daje przewagę sygnału odbiorczego nad volume
  (+0.15/+0.23 AUC: ctx 0.86/0.83, temporal 0.78/0.83 vs volume 0.59–0.64), a blast wygrywa volume (0.83)
  — jego przyczynowo-zdarzeniowe liczniki widzą burst, którego oknom strictly-past odbiorczych detektorów
  brakuje na rzadkiej topologii.
- **Granice operacyjne**: przy realnej częstości ataku (1:10⁴) na każdy wykryty atak przypada ~830 fałszywych
  alarmów → narzędzie triage dla analityków, nie autonomiczna obrona. Szum etykiet najszybciej degraduje
  model nauczony (temporal 0.72→0.60 przy 20% szumu). W reżimie skrytym czujność to r≈0.03, czyli
  ~3300 fałszywych alarmów na wykryty atak przy 1:10⁴.

### Artykuł B (IEEE) — multiplex grafu wiedzy i rhythm-leak pitfall

- **Multiplex + rytm**: AUC 0.70→0.87 i likwidacja 100% fałszywych alarmów na legalnych nowych nadawcach.
  Uczciwa kontrola (tasowanie **tylko** warstw OSINT) pokazuje jednak, że zysk niosą kontakty+rytm, a OSINT
  dokłada tylko ~0.03 AUC — wartość metody leży po stronie fałszywych alarmów, nie wykrywania podszywek
  (zimna impersonacja: czujność 0.04 przy FPR 1%). Nowy tc-split lokalizuje ten margines: na plasterku
  "znany nadawca" multipleks nic nie zmienia (0.79=0.79), na plasterku "w rytmie" (atak mimicry)
  0.58→0.64, na "nowym nadawcy" 0.81→0.90 — tasowanie OSINT cofa oba plasterki, więc to przyczynowy
  wkład OSINT. Przy 100% fabrykacji OSINT multipleks spada do poziomu shuffle (0.86 vs 0.855) i fałszywe
  alarmy na nowych nadawcach wracają (0.10) — pełny fałszywy slad jest nierozróżnialny od prawdziwego.
- **Odkrycie metodyczne — rhythm-leak pitfall**: rytm nadawcy liczony na pełnej historii zawyża czujność
  detektora rytmu do 0.73 przy FPR 1%; uczciwy podział czasowy (rytm z wczesnych 60%) daje **0.00 nawet przy
  FPR 5%**, bo ponad 5% legalnej poczty wypada poza uczciwie policzonym rytmem. Pełny multiplex trzyma
  czujność **0.14 (AUC 0.82)** — warstwy strukturalne liczone z wczesnych 60%, jak rytm. Ostrzeżenie dla
  całej klasy detektorów behawioralnych.
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

### A3 (recenzja wewnetrzna r2): dwa harnessy i dlaczego liczby „volume" roznia sie miedzy tabelami

W paczce sa DWA harnessy o roznych protokolach benign (oba z jedna regula przyczynowosci
zdarzeniowej po A1 — liczniki i cechy widza wylacznie zdarzenia o scisle wczesniejszym
znaczniku czasu t(e) = kubelek + pozycja na liscie):

1. **harness domyslnej kaskady** (`exp_cascade._simulate`): atak = naiwna kaskada
   (P_INFECT=0.6, H=4), benign = punktowy ruch kontaktow W RYTMIE (60/kubełek).
   Uzywaja go: Tab. S2 (sweep p_inf), Fig. S5 (krzywa FPR), Tab. 1 (hierarchia).
   Tu COMPA osiaga AUC 0.83-0.88 i czulosc@1% ~0.26 — jej liczniki widza rozwijajacy
   sie burst, a benign w rytmie nie zaslania cechy nowosci czasu.
2. **harness stealth** (`exp_cascade_stealth._simulate_stealth` oraz
   `cascadebench.build_scenario`): atak skryty/blast z matched benign 3:1, 15% ruchu
   POZA rytmem, 30% mimicry rytmu przez atakujacego. Uzywaja go: Tab. 2 (stealth sweep),
   Tab. 3 (baselines), kontrasty Wilcoxona (headline-CI). Tu COMPA pod blastem spada do
   ~0.63-0.70 AUC i czulosci 0.03-0.06: benign off-rhythm niszczy ceche nowosci czasu,
   a mimicry odbiera bit rytmu. To NIE jest sprzecznosc danych — to dwa rozne pytania:
   „czy volume widzi kaskade wzgledem latwego tla" (tak, 0.83) i „czy widzi ja wzgledem
   tla dopasowanego do ataku" (ledwo, 0.63).

Tozsamosc modeli (A3): **„Temporal GNN" w Tab. 1 to klasa `TemporalGNN` z
`exp_cascade.py`** (TGN-lite z GRU-pamiecia, 120 epok, 5 ziaren) → 0.871/0.099 po A1.
Wiersz „Library TGN" to oficjalny TGN z PyG (`exp_cascade_tgn.py`), a „TGN-lite /
TGAT-lite" z Tab. S1 to ta sama architektura trenowana w `exp_cascade_arch.py`
(100 epok) — roznice rzędu 0.01 AUC to epoch-count + niedeterminizm watkow, nie rozne modele.

### Komendy

```bash
cd code

# pojedynczy eksperyment (np. stealth, 20 seedów, ~2 h CPU):
python experiments/exp_cascade_stealth.py

# wszystkie eksperymenty po fixach (5 równoległych ścieżek, OMP=4):
bash experiments/run_reruns2.sh lane1  # panel cb (COMPA/hopper/hopper-no-precursor/reczny/TGN)
bash experiments/run_reruns2.sh lane2  # exp_cascade + org/scale + sweep + arch + tgn
bash experiments/run_reruns2.sh lane3  # Enron (split + kaskady blast/stealthy) + stealth sweep
bash experiments/run_reruns2.sh lane4  # headline-CI + mechanism + fpr + realgraphs + compromised
bash experiments/run_reruns2.sh lane5  # NOWE: legal_cascade + tc_split + osint_shuffle + multiplex

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

### Artykuł C (IEEE, roboczy) — graf wiedzy jako narzędzie i cel ataku

Trzecia publikacja kompletu: knowledge graphs jako narzędzie operacyjne (CTI KG,
note-grafy, GraphRAG) i jako powierzchnia ataku (fabrykacja OSINT, trucie korpusu,
fabrykacja trójek). Dwa tory ewaluacji: kontrolowany (multipleks 150 person z flagami
fabrykacji) i publiczny (FB15k-237). Wyniki:

- **Detekcja sfabrykowanych krawędzi (RQ1) — negatywny**: przy fabrykacji i.i.d. z
  rozkładu łącznego cechy istnienia są bez mocy (AUC ~0.50), kontrola permutacji 0.50;
  pozorny sygnał spójności (0.56–0.65) to artefakt definicji genuineness (ujawniony).
- **Trucie GraphRAG-lite (RQ2)**: atak zoptymalizowany pod zapytanie (recepta PoisonedRAG)
  korumpuje 100% top-5 ofiar; bramka spójności (wsparcie wewn. LUB ≥2 niezależne ślady
  OSINT) zeruje go przy k=0 — atakujący musi posadzić ≥2 niezależne sfałszowane publiczne
  ślady na ofiarę. Cena na czystym korpusie: 92.5% ofiar traci z top-5 legalny
  międzyorganizacyjny jedno-faktowy ślad (lekcja FP z artykułu B).
- **FB15k-237 (RQ3)**: spójność relacyjna (wspólni sąsiedzi) ranguje sfabrykowane trójki
  AUC 0.71–0.72 przy cechach istnienia 0.52 (kontrast Wilcoxon p=0.005, δ=1.0, 10 ziaren);
  ale sens@FPR1% ≈ 0 (podłoga izolowanych trójek prawdziwych) — narzędzie triażu, nie filtr
  progowy.
- **Wyścig zbrojeń feature-by-feature**: fabrykator path-preserving (tail z 2-hop sąsiedztwa
  h — przeciwnik ZNA cechę wspólnych sąsiadów) neutralizuje ją: cn AUC 0.71→0.56, combined
  0.72→0.59, a najlepszą cechą staje się suma stopni (0.71) — czyli dokładnie ta, którą
  fabrykatory marginalne uczyniły bezużyteczną. Sygnał nie znika, tylko się przemieszcza.
- **Bramka w dwóch trybach**: hard (skip) i soft (demote ×0.25) — oba zerują atak
  zoptymalizowany przy k=0; tryb demote to ranking-prior dla analityka (nic nie jest ukrywane).

| Tabela w C | Skrypt (`code/experiments/`) | CSV (`code/results/`) |
|---|---|---|
| C: fabrykacja OSINT (Tab. I) + kontrasty | `exp_fab_detection.py` | `exp_fab_edge_detection.csv`, `exp_fab_contrasts.csv` |
| C: trucie GraphRAG-lite (Tab. II) | `exp_rag_poison.py` | `exp_rag_poison.csv` |
| C: FB15k-237 (Tab. III) + kontrasty | `exp_kg_poison_fb15k.py` | `exp_kg_poisoning.csv`, `exp_kg_contrasts.csv` |

FB15k-237 pobierany do `code/data/fb15k237/` (publiczny, ConvE repo). Trzy skrypty są
czystym stdlib (bez sklearn) i deterministyczne (C1/C3: 10 ziaren; C2: 5 ziaren).
