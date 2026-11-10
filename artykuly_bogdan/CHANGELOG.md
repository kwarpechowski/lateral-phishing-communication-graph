# Changelog poprawek: artykuły A (JNCA) i B (konferencja)

**Data:** 2026-11-08
**Podstawa:** [review_A-Journal.md](review_A-Journal.md), [review_B-Conference.md](review_B-Conference.md). Numery „pkt” odnoszą się do punktów w tych recenzjach.
**Oryginały:** nietknięte w `A-Journal.zip` i `B-Conference.zip`.

> **Uwaga (runda 1):** w rundzie 1 PDF-y nie były przekompilowane. **Od rundy 2 PDF-y są aktualne**, zob. sekcję „Runda 2” na końcu.

Poprawki dzielą się na dwa rodzaje:
- **[ZROBIONE]**: zmiana tekstu, którą dało się wykonać bez dostępu do kodu i danych.
- **[TODO w .tex]**: wymaga sprawdzenia kodu albo ponownego uruchomienia eksperymentów. W pliku źródłowym jest komentarz `% TODO (author): ...` w odpowiednim miejscu. Pełną listę daje `grep -rn "TODO (author)" A-Journal B-Conference`.

---

## Artykuł A: `A-Journal/`

### main.tex
- **[ZROBIONE] Abstrakt** (pkt 6, 12): „temporal graph network … holds (AUC 0.60)” → „degrades less and stays above chance (AUC 0.60)”. Usunięto „with large effect sizes” dla grafów rzeczywistych i zastąpiono „the regime-level contrast replicates in direction”.
- **[ZROBIONE] Highlights** (pkt 6, 10, 12). Wszystkie mają ≤ 85 znaków:
  - „survives stealthy attacks where volume detection collapses” → „degrades less than volume detection under stealthy attacks”;
  - „causally temporal” → „temporal”;
  - „replicates … with large effect sizes” → „The regime-level contrast replicates on Enron and two SNAP graphs”.
- **[POZOSTAWIONE]** TODO o DOI Zenodo i finansowaniu NASK (pkt 22). To decyzja autorów.

### highlights.txt
- **[ZROBIONE]** (pkt 21): plik zawierał highlights artykułu B (cross-layer, susceptibility). Zastąpiłem go highlights z `main.tex`.

### en/introduction.tex
- **[ZROBIONE]** (pkt 12): wkład 3 „with paired tests and effect sizes” → „with the regime-level contrast reproduced in direction on each graph”.

### en/related_work.tex
- **[ZROBIONE]** (pkt 25): usunięto cytowanie `mcauley2015image` (rekomendacje obrazów, niezwiązane z tematem). Wpis w `.bib` zostaje, ale BibTeX go nie wypisze.

### en/methodology.tex
- **[ZROBIONE]** (pkt 10): akapit „The shuffle control is exact … formally justifying” zastąpiono opisem kontroli jako *ablacji czasowej*. Dodano, że tasowanie niszczy też współwystąpienia w kubełku, więc spadek jest górnym ograniczeniem wkładu samego porządku (co tłumaczy, czemu shuffled 0,72 < static 0,77). Dodano zastrzeżenie, że „causal” jest w tekście używane w sensie ablacji. Pozostałe wystąpienia „causally temporal” zostawiłem, bo to zastrzeżenie je obejmuje.
- **[ZROBIONE]** (pkt 17): FPR podany jawnie w tych samych jednostkach (10⁻² vs 4·10⁻⁶). Zmieniono „liberal lower bound” na „liberal operating point”.

### en/results.tex
- **[ZROBIONE]** (pkt 13): „every headline contrast … ≥20 seeds” → stealth, baselines i label noise mają 20 seedów, hierarchia detektorów 5. Poprawione w dwóch miejscach (setup i „Variance and significance”).
- **[ZROBIONE]** (pkt 5): usunięto zdanie z AUC 0,49 dla „static detector”, sprzeczne z Static GNN = 0,77 w Tab. 1. Zastąpiono odwołaniem do Static GNN (0,77 / 0,063). **[TODO]** Jeśli 0,49 to inny detektor, trzeba go przywrócić z opisem.
- **[TODO]** (pkt 19): dodać liczby dla biblioteki TGN do Tab. 1.
- **[ZROBIONE]** (pkt 6): sekcja stealthy: „holds” → „degrades less and stays above chance”, plus zdanie, że przy K=1 sygnał jest słaby w wartościach bezwzględnych.
- **[TODO]** (pkt 7): wyjaśnić AUC volume < 0,5 i opisać, na jakim reżimie trenowano detektor.
- **[ZROBIONE]** (pkt 2): w tekście i podpisie Tab. 3 poprawiono błędne stwierdzenie, że przy K=1 wygrywa path-based. Wygrywa kontekst ręczny (0,76 > path 0,68 > temp 0,65). Dodano zdanie, że Tab. 3 pochodzi z drugiego harnessu (`cascadebench/`), stąd inne wartości przy K=1 niż w Tab. 2. **[TODO]** Opisać w Methodology, czym różnią się oba harnessy.
- **[ZROBIONE]** (pkt 11): walidacja indukcyjna: tekst mówi teraz wprost, że temporal GNN *nie* prowadzi w AUC (kontekst ręczny 0,86–0,88 vs 0,80–0,84), a przewaga dotyczy tylko czułości przy FPR 1%. Dodano brakujący punkt 1,6× dla 30 firm.
- **[TODO]** (pkt 9): wyjaśnić, czemu temporal AUC jest wyższe przy p_inf=0,05 (0,96) niż przy 0,6 (0,90).
- **[ZROBIONE]** (pkt 4): usunięto „Wilcoxon temporal vs random p<0.001” na grafach rzeczywistych (przy 3 seedach minimalne p = 0,25). Replikacje opisano jako potwierdzenia kierunkowe. **[TODO]** Wyjaśnić, na czym liczono pierwotny test, i najlepiej testować temporal vs volumetric na większej liczbie seedów. **[TODO]** (pkt 8): wyjaśnić SNAP blast (0,70) < stealthy (0,88).
- **[TODO]** (pkt 3): wskazać dokładny „stealthy point” dla Wilcoxona i Cliff's δ. Wartości +0,081 i +0,064 nie zgadzają się z Tab. 2 ani 3(a). **Liczb nie zmieniałem**, bo nie mam danych.
- **[ZROBIONE]** (pkt 17): porównanie z Ho et al. podane jako FPR (~2500×) zamiast zestawiania „FP na wykryty atak” z „FP na milion”. Ta sama poprawka w podpisie Tab. baserate.

### en/discussion.tex
- **[ZROBIONE]** (pkt 18): usunięto niezmierzone „sub-millisecond scoring” i zastąpiono „header-only scoring”.

### en/conclusion.tex
- **[ZROBIONE]** (pkt 6): „holds at 0.60–0.85” → „degrades to 0.60–0.78 but stays above chance”. 0,85 to wartość dla blast, a nie stealthy. Dodano „in direction” dla replikacji.

### supplementary.tex
- **[TODO]** (pkt 1, **krytyczne**): cecha *momentum* w baselinie path-based używa okna [b, b+W], czyli przyszłych kubełków. Trzeba sprawdzić kod. Jeśli to prawdziwy look-ahead, należy zmienić okno na [b−W, b] i powtórzyć eksperymenty. Jeśli to literówka, wystarczy poprawić wzór.

### figures/
- **[ZROBIONE]** (pkt 26): 6 nieużywanych figur (z artykułu B) przeniesiono do `figures/_unused/`. Niczego nie usunąłem.

---

## Artykuł B: `B-Conference/main.tex`

- **[ZROBIONE] Interpretacja Tab. susc** (pkt 1, **krytyczne**): executive ma Susc 0,924, czyli jest **najbardziej** narażony, a mid/junior (0,838–0,840) najmniej. Tekst twierdził odwrotnie. Akapit przepisany. Dodano możliwe wyjaśnienie (bogate osadzenie poszerza przestrzeń, w której atak wygląda spójnie). Wniosek zmieniono z „what protects is embedding in the graph rather than status” na „seniority does not buy protection”. W Limitations usunięto „exposure tracking graph embedding”.
- **[ZROBIONE]** (pkt 9): po Def. 2 dodano zdanie, że Susc_D to rezydualna ekspozycja zostawiana przez detektor, a nie ludzka podatność, i że wyższa wartość oznacza większą ekspozycję. W abstrakcie i konkluzji „organizational susceptibility metric” → „detector-relative susceptibility metric”. **Tytułu nie zmieniałem**; zmiana „Organizational Susceptibility” to decyzja autorów.
- **[ZROBIONE]** (pkt 2): abstrakt i konkluzja mówią teraz wprost, że zysk jest po stronie fałszywych alarmów, a cold impersonation pozostaje prawie niewykrywana (czułość 0,04). **Tytułu nie zmieniałem** (sugestia z recenzji).
- **[ZROBIONE]** (pkt 3): stwierdzenie „advantage stems from actual cross-layer structure” złagodzone. Dodano, że kontrola tasuje wszystkie cechy, więc sama nie izoluje wkładu *spójności* między warstwami. „keeps every feature” → „keeps the feature marginals”. **[TODO]** Dodać kontrolę tasującą tylko warstwy OSINT.
- **[TODO]** (pkt 4): zdefiniować tc dla nadawcy zewnętrznego bez historii (τ(s) niezdefiniowane).
- **[ZROBIONE]** (pkt 5): fusion w reżimie mimicry (0,848) jest niższe niż sam multiplex (0,862). Usunięto „best of both” w tekście i abstrakcie, a w Tab. content przeniesiono pogrubienie na rzeczywiste maksima.
- **[ZROBIONE]** (pkt 6, częściowo): „~0.49 rather than 0.87” → „~0.5 rather than 0.88”, zgodnie z Tab. enronsplit. **[TODO]** Contacts-only ma 0,57 / 0,70 / 0,98 w trzech miejscach; trzeba opisać konfiguracje w podpisach albo je ujednolicić.
- **[TODO]** (pkt 7–8): opisać wstrzykiwanie ataków na Enronie i to, czy kontakty, co-recipients i domena też są estymowane tylko na wczesnych 60%.
- **[ZROBIONE]** (pkt 17): sekcja „Data and Code Availability” jest teraz objęta `\ifanon`. W wersji anonimowej brzmi „will be released upon acceptance”, bez Zenodo.
- **[ZROBIONE]** (pkt 16): 3 nieużywane figury przeniesiono do `figures/_unused/`.

---

## Niezrobione (wymaga decyzji autorów lub większego nakładu pracy)

| Co | Artykuł | Dlaczego nie zrobione |
|---|---|---|
| Ponowne uruchomienie eksperymentów (look-ahead, shuffle tylko OSINT, więcej seedów na SNAP, adaptacyjny atakujący, alternatywny model propagacji, rzadszy graf) | A, B | Brak kodu i danych w paczce |
| Odchylenia standardowe / CI w tabelach A (pkt 14) | A | Wymaga per-seed CSV z pakietu reprodukcyjnego |
| Static GNN na Enronie (pkt 20) | A | Wymaga eksperymentu |
| Przeformułowanie wspólnych akapitów A i B (wstęp, related work), ujawnienie powiązania redakcji | A, B | Decyzja redakcyjna autorów; wymaga przepisania dużych fragmentów |
| Zmiana tytułu B | B | Decyzja autorów |
| Numeracja sekcji suplementu (pkt 24), overfull hbox (pkt 23) | A | Kosmetyka; overfull trzeba sprawdzić po kompilacji |
| Mocniejszy baseline treściowy niż TF-IDF, opis generatora treści (pkt 12) | B | Wymaga eksperymentu i wiedzy o generatorze |
| Skrócenie wątku o atakach i obronach GNN w related work B (pkt 15) | B | Decyzja autorów |
| DOI Zenodo, potwierdzenie finansowania | A, B | Dane spoza paczki |


---
---

# Runda 2 (2026-11-08)

**Podstawa:** [review2_A-Journal.md](review2_A-Journal.md), [review2_B-Conference.md](review2_B-Conference.md). Numery „pkt” odnoszą się do tych recenzji.

## Kompilacja
- Zainstalowano **tectonic** (silnik LaTeX, `brew install tectonic`) i **poppler** (`pdftotext`, `pdfinfo`).
- Do `A-Journal/` dodano katalog `thumbnails/` z ikonami szablonu Elsevier CAS (z oficjalnego pakietu `els-cas-templates` na CTAN). Bez niego `main.tex` się nie kompilował.
- Zbudowano: `A-Journal/main.pdf` (19 stron), `A-Journal/supplementary.pdf` (6 stron), `B-Conference/main.pdf` (7 stron). Stare PDF-y są nadpisane, ale oryginały nadal leżą w zipach.
- Polecenie: `cd A-Journal && tectonic -X compile main.tex` (analogicznie `supplementary.tex` i `B-Conference/main.tex`).
- Pozostaje ostrzeżenie `Overfull \hbox 117pt` przy `\maketitle` w A. Pochodzi z szablonu CAS i na stronie tytułowej nie widać przepełnienia.
- **Kod eksperymentów nie został znaleziony na tym komputerze** (szukano w katalogu domowym, na Pulpicie, w iCloud, w Spotlight i na `/Volumes/backup2`). Jest na innym komputerze. Wszystkie punkty wymagające kodu zostały oznaczone `% TODO (author, review 2 ...)`.

## Artykuł A
- **[ZROBIONE] (pkt 1, krytyczne)** Naprawiono 3 zdania wycięte przez komentarze TODO z rundy 1 (`results.tex`: monotoniczność czułości, akapit o ataku chirurgicznym, zdanie z Wilcoxonem). Sprawdzono w PDF przez `pdftotext`.
- **[ZROBIONE] (pkt 2)** Teza „sygnał wymaga propagacji” zastąpiona przez „sygnał wymaga fan-out ≥ 2” w T2 (wstęp), RQ2 (konkluzja), tabeli roadmap i Discussion (interpretacja i Limitations). W `results.tex` dodano akapit: przy $p_\mathrm{inf}=0$ i fan-out ≥ 2 atak jest wykrywany przez volume i temporal GNN, bo synchroniczna wysyłka sama tworzy falę; kaskada jest ważna, bo pozwala dotrzeć do wielu ofiar *bez* takiego burstu. **[TODO]** Wyjaśnić, czemu kontekst ręczny ma 0,45–0,50 przy fan-out 3–5.
- **[TODO] (pkt 3)** Rola bitu rytmu (oczekiwane AUC ≈ 0,80 vs raportowane 0,70 / 0,60 / 0,46). Komentarz w `results.tex` przy Tab. stealth.
- **[TODO] (pkt 4)** Look-ahead w cechach węzłów $X$ GNN-ów, w licznikach volume i *burst* z bieżącym kubełkiem. Komentarz w `methodology.tex`. Zdanie „so it can only use the past” zawężono do „its memory can only use the past”.
- **[ZROBIONE] (pkt 5)** Mocne sformułowania ujednolicone z abstraktem: „holds / causally temporal / does not break down / robust to stealth / survives / maintains its advantage / exactly” zastąpiono przez „degrades less / stays above chance / depends on temporal order / in direction”. Zmiany objęły wstęp, Model, Results (tekst, podpisy Tab. detectors, stealth, realgraphs, Fig. stealth, roadmap), Discussion (z tytułem podsekcji), konkluzję, abstrakt („survives” → „degrades far less than volume”) i podpisy w suplemencie. Słowo „causal” zostało tam, gdzie obejmuje je zastrzeżenie z Methodology („ablation sense”).
- **[ZROBIONE] (pkt 6)** Odwrócona logika kontroli shuffle w `methodology.tex` poprawiona.
- **[ZROBIONE] (pkt 7)** Predictivity probe: „remains significant under a paired test” → „keeps its direction… a directional check, not a significance test”.
- **[TODO] (pkt 8)** Tab. S4 (label noise): wskazać K i harness. Komentarz w suplemencie.
- **[ZROBIONE] (pkt 9)** Opis COMPA (konta w sieciach społecznościowych, profil behawioralny) i Hoppera (lateral movement z logów logowania, nie e-mail) poprawiony w related work, Methodology i S6. Baseliny opisano jako „inspired by” z przeniesieniem idei na e-mail.
- **[ZROBIONE, częściowo] (pkt 10)** Usunięto argument „N=150 = core users Enronu, więc ta sama skala”. Teraz tekst mówi, że graf Enronu ma 1990 kont, więc replikacja testuje też większą skalę. **[TODO]** Opisać konstrukcję grafu i różnicę z 818 nadawcami w B.
- **[ZROBIONE] (pkt 12)** „Every result… with the time-shuffle control” → opis faktycznego zakresu metryk i kontroli. „Dropping headline AUC” → „rather than relying on headline AUC alone”.
- **[ZROBIONE] (pkt 13)** Wstęp: usunięto utożsamienie „2% błędu benchmarku = spear phishing i BEC”.
- **[ZROBIONE] (pkt 14)** Tab. baserate: 0,22 → 0,21 (spójne z r = 0,14). W podpisie dodano, że „1:n” to stosunek ataków do legalnych zdarzeń, π = 1/(n+1).
- **[ZROBIONE] (pkt 15)** Liczby seedów rozpisane dokładnie (5 / 8 / 5 lub 3 / 10 / 8 Enron / 3 SNAP). Usunięto uzasadnienie „lower variance on a fixed structure” dla tabel syntetycznych. „Real-graph validations use 8/5/3” → „8 (Enron) and 3 (SNAP)”.
- **[ZROBIONE] (pkt 16)** S6: „200 trees” → „200–250 trees as in Tab. S1”.
- **[ZROBIONE] (pkt 17)** „a requirement we are beginning to meet” → „we report that gap as the open requirement”.
- **[BEZ ZMIAN]** Pkt 11 (otwarte TODO z rundy 1), 18 (DOI Zenodo), 19 (eksperymenty z rundy 1): wymagają kodu lub decyzji autorów.

## Artykuł B
- **[ZROBIONE] (pkt 1, krytyczne)** Naprawiono 2 zdania wycięte przez komentarze TODO. Sprawdzono w PDF.
- **[ZROBIONE] (pkt 2)** Abstrakt: „fusion keeps the best of the two” → „stays close to the better of the two in each regime”. Ta poprawka była zapowiedziana w rundzie 1, ale nie została wykonana.
- **[ZROBIONE, częściowo] (pkt 3)** Do sekcji Enron dodano wyjaśnienie, że czułość binarnej cechy rytmu przy stałym FPR jest funkcją schodkową. Spadek do 0,00 przy AUC 0,75 jest częściowo artefaktem progu, a sam przeciek jest realny (AUC 0,86 → 0,75). W Discussion „almost entirely leakage” → „inflated by leakage (and… by the step at the FPR threshold)”. Usunięto pogrubienie 0,00 w Tab. enronsplit. **[TODO]** Czułość przy FPR 2% i 5%; czy ataki są wyłączone z estymacji rytmu na pełnej historii.
- **[ZROBIONE, częściowo] (pkt 4)** Sprzeczny ranking „contacts > static multiplex” zastąpiono twierdzeniem „Multiplex+rhythm leads across the whole noise range”. **[TODO]** Sprawdzić w wynikach sweepu szumu i ewentualnie przywrócić pełny ranking.
- **[BEZ ZMIAN] (pkt 5)** $tc$ dla nadawcy zewnętrznego, shuffle tylko OSINT, wstrzykiwanie ataków na Enronie: TODO z rundy 1 zostają.
- **[ZROBIONE] (pkt 6)** Def. 2 przeformułowana na $\mathrm{Susc}_{D,c}$ per kanał ataku $c$ (impersonacja zewnętrzna lub przejęty kontakt), zgodnie z tym, co liczy Tab. susc. Estymator odwołuje się teraz do „each channel c”.
- **[ZROBIONE] (pkt 7)** „Organizational susceptibility” → „detector-relative susceptibility” w słowach kluczowych, wkładzie 4, tytule sekcji i podpisie Tab. susc. Related work rozdziela teraz podatność ludzką od strukturalnej ekspozycji detektora. **Tytułu nadal nie zmieniałem**, to decyzja autorów (tytuł mówi „Organizational Susceptibility”, a reszta tekstu „detector-relative”).
- **[ZROBIONE] (pkt 8)** „Causal” usunięte lub złagodzone: Model („testable by ablation… bounds rather than isolates”), Method („shuffle control”), pytanie badawcze we wstępie, Discussion („a real signal… as the ablations show”). Akapit o cyrkularności mówi teraz, że shuffle (0,51 < contacts-only) świadczy o łącznej strukturze cech, a niekoniecznie o samej spójności między warstwami.
- **[ZROBIONE] (pkt 9)** Dodano zdanie, że duża część wzrostu AUC pochodzi z kanału przejętych kont (rytm), a dla impersonacji zysk leży w redukcji FP.
- **[TODO] (pkt 10–12)** Nieużywane $C(s,v)$ i $w_L$, 538 vs 518 krawędzi domeny, fabrykacja a teza o „multilayer consistency”. Komentarze przy odpowiednich miejscach w `main.tex`.
- **[ZROBIONE, do weryfikacji] (pkt 14)** Zdanie o miarach jawnych vs konkatenacji przepisane jednoznacznie: linear 0,860 vs 0,840, trees 0,858 vs 0,860. **Przypisanie liczb to moja interpretacja** dawnego zdania; komentarz TODO prosi o weryfikację.
- **[ZROBIONE] (pkt 16)** Wstęp: to samo złagodzenie argumentu „98%” co w A.
- **[BEZ ZMIAN]** Pkt 13 (Enron 818 vs 1990), 17 (DOI), 18 (tekst wspólny z A), 19 (limit stron, obecnie 7), 20 (pozostałe z rundy 1).

## Lista otwartych TODO
`grep -rn "TODO (author" A-Journal B-Conference --include=*.tex`
