# Recenzja (runda 2): Artykuł A (JNCA)

**Tytuł:** *Lateral Phishing as a Violation of the Organizational Communication Graph: Cascade Dynamics and Leak-Aware Evaluation*
**Stan:** źródła `.tex` po poprawkach z [CHANGELOG.md](CHANGELOG.md), 2026-11-08. Na maszynie nie ma LaTeX-a, więc `main.pdf` i `supplementary.pdf` to **stare** wersje. Recenzja dotyczy źródeł.

**Ocena ogólna:** *major revision.* Poprawki z rundy 1 poszły w dobrą stronę: abstrakt, highlights, Wilcoxon na grafach rzeczywistych, walidacja indukcyjna, base rate. Wprowadziły jednak **trzy uszkodzone zdania** (komentarze TODO wklejone w środek akapitu). Zostawiły też kilka miejsc, w których stare, mocniejsze sformułowania przeczą nowym, ostrożniejszym. Do tego dochodzą nowe problemy merytoryczne. Najważniejsze z nich: wynik dla ataku chirurgicznego przeczy tezie, że „sygnał wymaga propagacji”, a sam bit rytmu powinien dawać wyższe AUC, niż raportują detektory, które go zawierają.

---

## Krytyczne

### 1. Trzy zdania wycięte przez komentarze TODO (błąd wprowadzony w rundzie 1)
W LaTeX-u `%` wycina wszystko do końca linii. W trzech miejscach tekst artykułu trafił do tej samej linii co komentarz i **zniknie z PDF-a**:

| Miejsce | Co wypada | Jak będzie wyglądał PDF |
|---|---|---|
| [results.tex:138](A-Journal/en/results.tex#L138) | „The temporal sensitivity column is not monotone in $K$” | „…that it does not collapse. (a dip at K=3: …) while the corresponding AUC rises…” |
| [results.tex:184](A-Journal/en/results.tex#L184) | „In the surgical regime (block b, no propagation at all, p_inf=0) a single-recipient” | „…leading from K=2 on. attack is undetectable by every detector…” |
| [results.tex:308](A-Journal/en/results.tex#L308) | „the receiving-side signal” | „…Wilcoxon test and Cliff's δ, beats the volume detector significantly…” |

**Poprawka:** przenieść tekst po komentarzu do nowej linii. Najlepiej przed wysyłką usunąć z akapitów wszystkie komentarze `% TODO (author)`.

### 2. Atak chirurgiczny przeczy tezie „sygnał wymaga propagacji”
[Tab. baselines](A-Journal/en/results.tex#L188-L215), blok (b), ma $p_\mathrm{inf}=0$, czyli **zero propagacji**. Mimo to przy fan-out 5 temporal GNN osiąga **0,96**, a volume 0,98. Wykrywalność zależy więc od **fan-outu**, a nie od propagacji. Wysyłka do kilku osób z jednej firmy (w klice) tworzy „falę po stronie odbiorców” bez żadnej kaskady. Tymczasem:
- T2 ([introduction.tex:42-43](A-Journal/en/introduction.tex#L42-L43)) i RQ2 w konkluzji mówią „the signal needs propagation”;
- [discussion.tex:16-18](A-Journal/en/discussion.tex#L16-L18): „the sender cannot hide without giving up propagation altogether — and giving up propagation is exactly the surgical regime”. To nieprawda: atakujący może zrezygnować z propagacji i nadal zostać wykryty przy fan-out ≥ 2;
- tytuł i rama artykułu („cascade dynamics”) sugerują, że to kaskada niesie sygnał.

**Poprawka:** przeformułować T2 na „the signal needs fan-out ≥ 2”. Trzeba też uczciwie przyznać, że przy $p_\mathrm{inf}=0$ i fan-out ≥ 2 sygnał to synchroniczna wysyłka. Detektor wolumenowy ją widzi, a temporal GNN wcale nie wygrywa. Osobno do wyjaśnienia: dlaczego kontekst ręczny (cecha *burst* liczy właśnie trafionych sąsiadów ofiary) ma 0,45–0,50 przy fan-out 5, skoro w klice powinien to widzieć.

### 3. Bit rytmu sam powinien dać AUC ≈ 0,80, a detektory, które go mają, raportują mniej
Z [results.tex:31](A-Journal/en/results.tex#L31) wynika, że 15% legalnej poczty jest off-rhythm, a 25% ataków naśladuje rytm. Sam bit $tc$ daje więc AUC = ½·(0,75 + 0,85) = **0,80**. Ten bit jest w cechach 1-hop (S6), więc trafia też do $x_{sv}$ obu GNN-ów i do kontekstu ręcznego. Tymczasem:
- 1-hop: **0,70** (Tab. detectors), 0,65 (Enron);
- temporal GNN przy K=1: **0,60** (Tab. stealth), a z $x_{sv}$ powinien mieć ≥ 0,80;
- wszystkie detektory przy fan-out 1: **0,45–0,48**.

Możliwe wyjaśnienia: w reżimach stealthy i surgical atakujący naśladuje rytm, szum etykiet obniża AUC albo rytm liczy się inaczej. Żadne z nich nie jest opisane. Jeśli to rytm, a nie fala, daje kontekstowi ręcznemu 0,76 przy K=1, osłabia to główną tezę. **Poprawka:** dla każdego reżimu podać odsetek ataków i legalnych zdarzeń z $tc=0$ oraz dodać do Tab. stealth wiersz „rhythm only” lub „1-hop”.

### 4. Look-ahead nadal niewyjaśniony; doszły dwa nowe źródła
- *momentum* w path-based używa $[b, b+W]$ ([supplementary.tex:178](A-Journal/supplementary.tex#L178)). To TODO z rundy 1 i nadal jest otwarte.
- **Nowe:** cechy węzłów $X$ obu GNN-ów to „in-degree, out-degree **from the event stream**” ([supplementary.tex:182-187](A-Journal/supplementary.tex#L182-L187)). Jeśli liczy się je na całym strumieniu, obejmują przyszłe zdarzenia, także zdarzenia ataku. Tekst twierdzi, że temporal GNN „can only use the past” ([methodology.tex:59](A-Journal/en/methodology.tex#L59)), a to jest prawdą tylko dla pamięci, nie dla $X$.
- **Nowe:** cechy volume („computed on the observed stream”, np. łączna liczba zdarzeń nadawcy) mają ten sam problem. *Burst* w kontekście ręcznym obejmuje bieżący kubełek $b$, a TGN czyta stan z $b-1$. To asymetria na korzyść kontekstu ręcznego, która może tłumaczyć jego wygraną przy K=1.

W pracy o *leak-aware evaluation* każdy z tych punktów trzeba jednoznacznie zamknąć.

---

## Poważne

### 5. Ostrożniejsze sformułowania nie zostały przeniesione do całego tekstu
Abstrakt i highlights mówią teraz „degrades less”, a Methodology zastrzega, że „causal” znaczy tylko „ablacja”. Te same twierdzenia w innych miejscach nadal brzmią mocniej:

| Miejsce | Brzmienie | Dlaczego przeczy reszcie tekstu |
|---|---|---|
| [introduction.tex:23](A-Journal/en/introduction.tex#L23) | „the one thing a stealthy attacker cannot remove” | przy K=1 AUC 0,60, czułość 0,039 |
| [introduction.tex:39-40](A-Journal/en/introduction.tex#L39-L40), [:48](A-Journal/en/introduction.tex#L48), [:61-62](A-Journal/en/introduction.tex#L61-L62) | „does not break down”, „robust to stealth”, „signal holds” | jw. |
| [results.tex:78](A-Journal/en/results.tex#L78) (Tab. roadmap) | „holds; causally temporal” | jw. |
| [results.tex:158](A-Journal/en/results.tex#L158), [:231-232](A-Journal/en/results.tex#L231-L232), [:253](A-Journal/en/results.tex#L253), [:256](A-Journal/en/results.tex#L256), [:291](A-Journal/en/results.tex#L291) | „holds” | jw. |
| [conclusion.tex:15-16](A-Journal/en/conclusion.tex#L15-L16) | „maintains its advantage” | jw. |
| [model.tex:114-116](A-Journal/en/model.tex#L114-L116) | „shuffling … destroys **exactly** the information it uses” | [methodology.tex:82-86](A-Journal/en/methodology.tex#L82-L86): „the ablation is not perfectly selective” |
| [discussion.tex:21-22](A-Journal/en/discussion.tex#L21-L22) | „makes the claim causal rather than correlational: it removes exactly the temporal order” | jw. oraz „not in the sense of causal identification” |

Recenzent zauważy, że abstrakt i wstęp mówią co innego.

### 6. Odwrócona logika kontroli tasującej
[methodology.tex:28-29](A-Journal/en/methodology.tex#L28-L29): „shuffles the temporal order of events, which zeroes the signal **if it is an artifact of topology** or identity”. Jest odwrotnie: jeśli sygnał wynika z topologii, tasowanie czasu go **nie** zmienia, a zeruje go wtedy, gdy sygnał jest dynamiczny. W [methodology.tex:154-155](A-Journal/en/methodology.tex#L154-L155) jest poprawnie.

### 7. Methodology obiecuje test istotności, którego Results już nie ma
[methodology.tex:36-37](A-Journal/en/methodology.tex#L36-L37): predictivity probe sprawdza, czy „the headline contrast … remains **significant** there under a paired test”. Po poprawkach [results.tex:256-259](A-Journal/en/results.tex#L256-L259) mówi wprost, że na grafach rzeczywistych testu nie ma (n = 3–8). Trzeba ujednolicić, najlepiej zmieniając definicję probe na „reproduced in direction”.

### 8. Tab. S4 (label noise) opisuje „stealthy attack”, ale liczby nie pasują do żadnego reżimu stealthy
Przy 0% szumu: vol **0,79**, path **0,72**, manual 0,66, temp 0,82. W Tab. stealth i Tab. baselines(a) volume pod atakiem stealthy ma 0,44–0,78, a path nigdy nie przekracza 0,68. Volume 0,79 to raczej poziom blast. Trzeba podać, jakie K i jaki harness, bo inaczej teza „learned GNN degrades fastest” nie odnosi się do tytułowego scenariusza.

### 9. Hopper to detekcja *lateral movement* (logowania), nie e-maili
[supplementary.tex:180-181](A-Journal/supplementary.tex#L180-L181): „The full Hopper system additionally uses the lure content and the URL of the message”. To błąd faktyczny. Hopper (Ho et al., USENIX Sec 2021, wpis `ho2021hopper`: „Modeling and Detecting Lateral Movement”) działa na grafie logowań między maszynami i nie przetwarza treści ani URL-i wiadomości. Podobnie COMPA (Egele et al., NDSS 2013) dotyczy kont w **sieciach społecznościowych** i profilu behawioralnego (czas, język, linki), a nie „sudden spike in volume”. Opis baselinów trzeba poprawić („inspired by the path-based idea of Hopper, transferred to email”). Inaczej recenzent znający te prace uzna porównanie za nierzetelne.

### 10. Enron: 150 czy 1990 kont? A w artykule B 818
[results.tex:26-27](A-Journal/en/results.tex#L26-L27) uzasadnia $N=150$ tym, że to „liczba core users Enronu, więc grafy porównujemy w tej samej skali”. W Tab. realgraphs graf Enronu ma jednak **1990** kont, a artykuł B używa Enronu z **818** nadawcami. Uzasadnienie nie trzyma się własnych danych. Trzeba opisać konstrukcję grafu (filtr, okres) i usunąć argument o tej samej skali.

### 11. Otwarte TODO z rundy 1, które recenzent wychwyci od razu
- Volume AUC 0,44 < 0,5 i brak protokołu treningu w eksperymencie stealth (pkt 7 rundy 1).
- Dwa harnessy: Tab. stealth (K=1: 0,44/0,60) i Tab. baselines(a) (K=1: 0,48/0,65, manual 0,76). Suplement twierdzi, że oba „implement identical feature definitions”, a liczby się różnią. Trzeba powiedzieć, czym się różnią (generator, splity, proporcje klas).
- Sekcja Rigor: +0,081 i +0,064 nie pasują do żadnej tabeli. Do tego „learned GNN beats hand-crafted (+0,064, p<0,001)” jest sprzeczne z Tab. baselines(a) przy K=1 (0,65 < 0,76).
- Na SNAP blast (0,70) < stealthy (0,88); sweep $p_\mathrm{inf}$: 0,96 przy 0,05 vs 0,90 przy 0,6.
- Static GNN 0,49 vs 0,77 (zdanie usunięte, TODO zostało).

---

## Drobne

12. **„Every result … with the time-shuffle control”** ([results.tex:66-68](A-Journal/en/results.tex#L66-L68)): Tab. stealth, baselines i realgraphs(b) nie mają kontroli shuffle, a Tab. baselines, S2 i S4 nie mają czułości@FPR. Trzeba złagodzić. Podobnie „dropping headline AUC” ([methodology.tex:33](A-Journal/en/methodology.tex#L33)) kłóci się z tym, że narracja w większości opiera się na AUC.
13. **„98% → pozostałe 2% to spear phishing i BEC”** ([introduction.tex:10-12](A-Journal/en/introduction.tex#L10-L12)): 2% błędu na benchmarku to nie to samo co udział spear phishingu w ruchu. To skrót myślowy, który łatwo zaatakować. To samo jest w B.
14. **Base rate:** 1:3 oznacza tu iloraz szans (π = 1/4). Wtedy przy r = 0,14 wychodzi 0,21 FP/atak, a nie 0,22 (0,22 wychodzi przy r = 0,139). Warto dopisać „odds 1:3” i ujednolicić r.
15. **Uzasadnienie liczby seedów** „because variance is lower on a fixed graph structure” ([results.tex:302-305](A-Journal/en/results.tex#L302-L305)) obejmuje też Tab. S2–S3, które są syntetyczne. Krzywa FPR ma 10 seedów, a nigdzie tego nie wymieniono. Nie wiadomo też, co oznacza „5” w „8/5/3”.
16. **Hiperparametry:** S1 podaje „n_estimators = 200–250”, a S6 „200 trees”.
17. **„a requirement we are beginning to meet”** ([discussion.tex:54](A-Journal/en/discussion.tex#L54)) przy odległości ~2500× od progu brzmi zbyt optymistycznie.
18. **Data availability** mówi „is openly archived on Zenodo”, a DOI jeszcze nie ma. Przed wysyłką trzeba zdeponować pakiet lub zmienić czas na przyszły.
19. **Pozostałe z rundy 1:** brak SD/CI w tabelach, brak wiersza z library TGN, brak static GNN na Enronie, alternatywny model propagacji i adaptacyjny atakujący, numeracja suplementu, overfull hbox (do sprawdzenia po kompilacji).

---

## Co się poprawiło od rundy 1
- Abstrakt i highlights są uczciwe („degrades less”, „replicates in direction”). Wszystkie highlights mają ≤ 85 znaków (najdłuższy dokładnie 85).
- Wilcoxon p<0,001 na 3 seedach usunięty; replikacje opisane jako kierunkowe.
- Walidacja indukcyjna mówi wprost, że temporal GNN nie wygrywa w AUC.
- Kontrola shuffle opisana jako ablacja z ograniczeniem selektywności (w Methodology).
- Porównanie z Ho et al. w jednostkach FPR (~2500×) jest poprawne.
- `highlights.txt` pasuje do artykułu A, a niepotrzebne figury są przeniesione do `_unused/`.

## Priorytety
1. Naprawić 3 uszkodzone zdania (pkt 1), skompilować i przeczytać PDF.
2. Przeformułować T2 i Discussion: fan-out, nie propagacja (pkt 2).
3. Wyjaśnić rolę bitu rytmu w każdym reżimie (pkt 3) i zamknąć wszystkie look-ahead (pkt 4).
4. Przeszukać tekst pod kątem „holds / causal / exactly / robust” i ujednolicić z abstraktem (pkt 5–7).
5. Poprawić opis Hopper/COMPA (pkt 9) i grafu Enronu (pkt 10).
