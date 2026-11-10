# Recenzja — Artykuł A (journal, JNCA)

**Tytuł:** *Lateral Phishing as a Violation of the Organizational Communication Graph: Cascade Dynamics and Leak-Aware Evaluation*
**Autorzy:** K. Warpechowski, B. Księżopolski
**Docelowe czasopismo (wg komentarza w `main.tex`):** Journal of Network and Computer Applications (Elsevier)
**Objętość:** 19 stron + 6 stron suplementu

**Ocena ogólna:** *major revision.* Pomysł jest dobry, a artykuł wyjątkowo uczciwy wobec własnych ograniczeń. Jest jednak kilka niespójności liczbowych między tabelami a tekstem, jedno potencjalne źródło wycieku danych w baselinie (co jest szczególnie niewygodne w pracy o *leak-aware evaluation*) oraz kilka sformułowań mocniejszych niż dane, na których się opierają. Recenzent, który przeliczy tabele, znajdzie to wszystko.

---

## Mocne strony

1. **Jasna, falsyfikowalna teza.** Atak lateral phishing jest tu opisany jako naruszenie grafu komunikacji w czasie, a sygnałem ma być fala po stronie *odbiorców*, a nie nadawcy. Teza jest przełożona na dwa pytania badawcze (RQ1/RQ2) i podtezy T1/T2, a tabela [Tab. roadmap](A-Journal/en/results.tex#L70-L88) wprost łączy je z dowodami. Czytelnik wie, czego szukać.
2. **Dobrze uchwycona różnica między stroną wysyłającą a odbierającą.** Pokazanie, że atakujący rozkładający wysyłki w czasie ("stealthy", parametr K) psuje detekcję wolumenową, to konkretny i praktycznie istotny wynik. Uzasadnienie, dlaczego tak się dzieje, jest przekonujące ([discussion.tex:11-24](A-Journal/en/discussion.tex#L11-L24)).
3. **Uczciwie wskazane ślepe pole.** Atak "chirurgiczny" na jednego odbiorcę (fan-out 1) jest zmierzony i nazwany wprost jako niewidoczny dla wszystkich detektorów ([results.tex:186-191](A-Journal/en/results.tex#L186-L191)). Wyznaczenie dolnej granicy modelu zagrożeń to rzadko spotykana uczciwość.
4. **Arytmetyka base-rate.** [Tab. baserate](A-Journal/en/results.tex#L314-L332) przelicza się poprawnie: przy r=0,14 i α=1% wychodzi ~714 FP na wykryty atak przy częstości 1:10⁴. Wniosek "narzędzie do triage, nie autonomiczna obrona" to dojrzałe stanowisko, które recenzenci cenią.
5. **Kontrola przez tasowanie czasu.** Tasowanie kolejności zdarzeń jako kontrola mechanizmu jest prostym i dobrym pomysłem. Wynik 0,90 → 0,72 jest czytelny.
6. **Replikacja na trzech rzeczywistych grafach** (Enron, email-Eu-core, CollegeMsg) zamiast tylko na danych syntetycznych. Sformułowanie twierdzenia o przenośności na poziomie reżimów, a nie pełnego rankingu ([methodology.tex:483-488](A-Journal/en/methodology.tex#L483-L488)), jest rozsądne.
7. **Pełna specyfikacja cech** wszystkich detektorów w suplemencie (S6), stałe hiperparametry bez strojenia, deklarowany pakiet reprodukcyjny.
8. **Porządny warsztat.** Spójna notacja ([Tab. notation](A-Journal/en/model.tex#L357-L381)), dobre figury TikZ, kompletne deklaracje Elsevier (CRediT, AI, competing interests).

---

## Słabe strony

### Krytyczne (do poprawienia przed wysłaniem)

1. **Możliwy wyciek przyszłości w baselinie path-based.** W [supplementary.tex](A-Journal/supplementary.tex) (S6) cecha *momentum* jest zdefiniowana jako "number of distinct targets of s in buckets **[b, b+W]**", czyli korzysta z przyszłych kubełków. Tymczasem TGN jest starannie przyczynowy (pamięć z b−1). To albo literówka, albo realna asymetria. Jeśli realna, baseline ma przewagę informacyjną. Podobnie *burst* w kontekście ręcznym obejmuje bieżący kubełek b. W artykule o metodologii odpornej na wycieki trzeba to wyjaśnić lub poprawić.

2. **Tabele 2 i 3(a) podają różne liczby dla tego samego scenariusza.** Dla K=1 przy p_inf=0,6:
   - Tab. 2 (stealth): vol 0,44, temp 0,60
   - Tab. 3(a) (baselines): vol 0,48, **manual 0,76**, temp 0,65

   W Tab. 3(a) przy K=1 **wygrywa kontekst ręczny (0,76), a nie path-based (0,68)**, a mimo to tekst mówi, że "the path-based detector wins only under extreme stealth" ([results.tex:169-171](A-Journal/en/results.tex#L169-L171) oraz podpis tabeli). Temporal GNN przegrywa tu z kontekstem ręcznym, co osłabia narrację "temporal GNN is the only detector with no breakdown regime". Trzeba poprawić tekst i wyjaśnić, czym różnią się ustawienia obu tabel (w suplemencie wspomniane są dwa harnessy: `experiments/` i `cascadebench/`).

3. **Sekcja Rigor jest sprzeczna z tabelami.** [results.tex:282-287](A-Journal/en/results.tex#L282-L287) podaje: "temporal GNN +0,081 AUC nad volume" (Tab. 2 daje +0,16, Tab. 3a +0,17) oraz "learned GNN beats hand-crafted (+0,064)" (Tab. 3a przy K=1 pokazuje odwrotnie: 0,65 vs 0,76). Nie wiadomo, który "stealthy point" jest testowany. Należy go jednoznacznie wskazać i uzgodnić liczby.

4. **Statystycznie niemożliwe p<0,001.** Na grafach SNAP są 3 seedy, na Enronie 8. Minimalne dwustronne p testu Wilcoxona to 0,25 dla n=3 i ~0,008 dla n=8. Zdanie "across the three real graphs the Wilcoxon test of temporal vs random gives p<0.001" ([results.tex:238-239](A-Journal/en/results.tex#L238-L239)) jest więc możliwe tylko przy testowaniu po zdarzeniach (pseudoreplikacja) albo po połączeniu seedów. Do tego porównanie z losowym klasyfikatorem jest trywialne; istotny jest kontrast **temporal vs volume**. Trzeba opisać jednostkę testu i testować właściwy kontrast.

5. **Static GNN raz ma 0,49, a raz 0,77.** [results.tex:93-94](A-Journal/en/results.tex#L93-L94) mówi "a static detector... is at chance (AUC 0,49)", a Tab. 1 dwie linijki niżej podaje Static GNN = 0,77. Bez wyjaśnienia, czym są te dwa różne detektory statyczne, wygląda to na błąd.

### Poważne (merytoryczne)

6. **"Holds" przy AUC 0,60 to nadinterpretacja.** W nagłówkowym wyniku (abstrakt, highlights, wnioski) temporal GNN przy K=1 ma AUC 0,60 i czułość 0,039 przy FPR 1%. To niewiele powyżej losowości. Uczciwiej byłoby napisać "degrades less" niż "holds" lub "survives".

7. **Wynik volume poniżej losowości (0,44) wymaga wyjaśnienia, nie tylko komentarza.** Klasyfikator trenowany i testowany na tym samym reżimie nie powinien systematycznie spadać poniżej 0,5; odwrócenie znaku dałoby 0,56. Wynik < 0,5 sugeruje przesunięcie rozkładu między train a test, np. trening na mieszance reżimów i test na K=1. Trzeba opisać protokół (na czym trenowany jest detektor w eksperymencie stealth). Akapit "not a tuning artifact" ([results.tex:290-294](A-Journal/en/results.tex#L290-L294)) nie odpowiada na to pytanie.

8. **SNAP: temporal GNN słabszy na blast (0,70) niż na stealthy (0,88).** To kontrintuicyjne: łatwiejszy atak jest wykrywany gorzej. Brak komentarza. Recenzent zapyta, czy oba reżimy mają te same proporcje klas i ten sam protokół.

9. **Sweep p_inf jest kontrintuicyjny i niewyjaśniony.** W Tab. S2 temporal GNN ma 0,96 przy p_inf=0,05, a 0,90 przy domyślnym 0,6. Mniej propagacji daje lepszą detekcję sygnału, który rzekomo *wymaga* propagacji. Potrzebne wyjaśnienie (np. zmiana proporcji klas albo inny skład zdarzeń ataku).

10. **"Causal" jest nadużywane.** Tasowanie czasu to ablacja, a nie identyfikacja przyczynowa. Zdanie "the shuffle control is exact... formally justifying" ([methodology.tex:529-532](A-Journal/en/methodology.tex#L529-L532)) jest za mocne. Tasowanie niszczy też korelacje współwystąpień, a nie tylko porządek. Wymowne jest to, że shuffled (0,72) wypada *gorzej* niż static GNN (0,77), czyli kontrola usuwa więcej niż samą informację o kolejności. Lepiej pisać "temporal ablation control".

11. **Walidacja indukcyjna nie pokazuje przewagi w AUC.** W Tab. S3 temporal ma niższe AUC niż kontekst ręczny przy wszystkich skalach (0,83/0,80/0,84 vs 0,86/0,86/0,88). Przewaga dotyczy wyłącznie czułości przy FPR 1%, a stosunek nie jest monotoniczny (4,4× → 1,6× → 1,4×). Tekst główny ([results.tex:208-211](A-Journal/en/results.tex#L208-L211)) powinien to powiedzieć wprost.

12. **"Large effect sizes" w abstrakcie nie ma pokrycia dla grafów rzeczywistych.** Cliff's δ jest podany tylko dla punktu syntetycznego. Na grafach rzeczywistych (3–8 seedów) nie ma rozmiarów efektu ani przedziałów ufności.

13. **Niespójna liczba seedów.** Tekst mówi "every headline contrast is averaged over ≥20 seeds" ([results.tex:47](A-Journal/en/results.tex#L47)), a Tab. 1, czyli główna hierarchia detektorów, ma 5 seedów.

14. **Brak odchyleń standardowych lub CI w tabelach głównych.** Przy różnicach rzędu 0,02–0,05 w czułości (np. dip przy K=3 tłumaczony "seed-level variability") bez rozrzutu nie da się ocenić, co jest istotne.

15. **Atak jest wstrzykiwany zgodnie z własnym modelem autorów.** To samo założenie (kaskada z p_inf i głębokością H) definiuje zarówno atak, jak i detektor. Autorzy to przyznają, ale warto dodać choćby jeden alternatywny model propagacji (np. atakujący wybiera cele według pozycji w grafie, a nie losowo), żeby pokazać, że sygnał nie jest artefaktem generatora.

16. **Zbyt słaby model przeciwnika.** Wyłączono adaptacyjnego atakującego, który zna graf. A to właśnie on jest naturalnym następnym krokiem po "stealthy" (np. wysyłka tylko do osób, których sąsiedzi nie są zainfekowani). Choćby prosty eksperyment z takim atakującym znacznie wzmocniłby pracę.

### Drobniejsze

17. **Porównanie z progiem Ho et al.** ([results.tex:305-306](A-Journal/en/results.tex#L305-L306)) zestawia "FP na wykryty atak" (~700) z "FP na milion maili" (<4). To różne jednostki. Lepiej porównywać FPR (1e-2 vs 4e-6, czyli ~2500×).
18. **"Sub-millisecond scoring"** ([discussion.tex:39](A-Journal/en/discussion.tex#L39)) nigdzie nie jest zmierzone. Trzeba dodać pomiar albo usunąć.
19. **Library TGN "matches them on the operational metric"** ([results.tex:102-103](A-Journal/en/results.tex#L102-L103)) bez liczb. Należy dodać wiersz do tabeli.
20. **Na Enronie brak static GNN** w Tab. 4(a), więc nie da się powtórzyć porównania static vs temporal na danych rzeczywistych.
21. **Plik [highlights.txt](A-Journal/highlights.txt) jest nieaktualny.** Mówi o cross-layer, predictivity probe i susceptibility, czyli treściach artykułu B. Highlights w `main.tex` są poprawne. Do Elsevier wysyła się osobny plik highlights, więc trzeba go podmienić.
22. **TODO przed wysłaniem:** DOI Zenodo i potwierdzenie finansowania (NASK) ([main.tex:121-128](A-Journal/main.tex#L121-L128)).
23. **Overfull hbox 117pt** w linii 99 (`\maketitle`), prawdopodobnie highlights lub tytuł. Trzeba sprawdzić PDF.
24. **Numeracja suplementu:** sekcja S2 zawiera Tab. S2, ale sekcja S5 zawiera Tab. S4. Lepiej numerować sekcje inaczej niż tabele (np. A–F).
25. **Related work zawiera zbędne cytowania**, np. `mcauley2015image` (rekomendacje obrazów w e-commerce) i kilka prac o rolach użytkowników. Recenzent odbierze to jako "citation padding". Warto przyciąć.
26. **Figury w paczce, których artykuł nie używa:** fig_ablation, fig_enron_audit, fig_layer_by_threat, fig_layer_selection, fig_multiplex_3d, fig_sensitivity (należą do artykułu B). Należy je usunąć z paczki submission.
27. **Gęstość grafu syntetycznego.** Klika w każdej firmie i tylko 7% krawędzi między firmami to bardzo wygodne środowisko dla kaskady. Jest to przyznane, ale warto dodać choćby jeden wariant mniej gęsty (np. z knobem dekorelacji ρ z artykułu B).

---

## Relacja z artykułem B

Oba artykuły dzielą studium przypadku (19 firm, 150 person, 556 krawędzi), wstęp, statystyki BEC i dużą część related work. Lateral/compromised account pojawia się też w B jako drugi kanał ataku. Przy wysyłce warto:
- zacytować artykuł B (choćby "under review") i ujawnić ten związek redakcji;
- przeformułować powtórzone akapity (narzędzia antyplagiatowe wydawców wychwycą wspólny tekst).

## Priorytety poprawek (skrót)

1. Wyjaśnić lub usunąć look-ahead w cechach path-based i burst (pkt 1).
2. Uzgodnić Tab. 2, Tab. 3(a) i sekcję Rigor; poprawić zdanie o tym, kto wygrywa przy K=1 (pkt 2–3).
3. Poprawić statystykę na grafach rzeczywistych (pkt 4, 12).
4. Złagodzić "holds", "causal" i "large effect sizes" (pkt 6, 10, 12).
5. Wyjaśnić anomalie: 0,44 < losowy, SNAP blast < stealthy, sweep p_inf (pkt 7–9).
