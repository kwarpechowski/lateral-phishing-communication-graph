# Recenzja (runda 2): Artykuł B (konferencja, IEEEtran)

**Tytuł:** *Spear Phishing as a Violation of the Domain Knowledge Graph: Cross-Layer Inconsistency, Organizational Susceptibility, and the Rhythm-Leak Pitfall*
**Stan:** źródło `main.tex` po poprawkach z [CHANGELOG.md](CHANGELOG.md), 2026-11-08. `main.pdf` jest **starą** wersją, bo LaTeX nie jest zainstalowany.

**Ocena ogólna:** *weak accept / major revision.* Najpoważniejszy błąd rundy 1, czyli odwrócona interpretacja Tab. susc, jest naprawiony i nowy akapit jest dobry. Poprawki zostawiły jednak **dwa uszkodzone zdania**, a jedna poprawka zapowiedziana w changelogu („best of both” w abstrakcie) **nie została wykonana**. Najmocniejszym wkładem pozostaje rhythm-leak pitfall, ale ma nową słabość: spadek 0,73 → 0,00 to w dużej mierze artefakt binarnej cechy przy progu FPR 1%.

---

## Krytyczne

### 1. Dwa zdania wycięte przez komentarze TODO (błąd wprowadzony w rundzie 1)

| Miejsce | Co wypada | Jak będzie wyglądał PDF |
|---|---|---|
| [main.tex:313](B-Conference/main.tex#L313) | „The most striking difference is in the false alarms: the contacts” | „…bounded by the ablation below. detector flags 100% of legitimate new senders…” |
| [main.tex:349](B-Conference/main.tex#L349) | „whereas the temporal” | „…(contacts 0.57 → static 0.65), consistency feature dominates…” |

**Poprawka:** przenieść tekst do nowej linii, a przed wysyłką usunąć z akapitów komentarze TODO.

### 2. „Fusion keeps the best of the two” nadal jest w abstrakcie
[main.tex:61-62](B-Conference/main.tex#L61-L62). Changelog podaje, że to usunięto, ale poprawiono tylko sekcję wyników. Tab. content pokazuje fusion **0,848 < 0,862** w reżimie mimicry. Proponowane brzmienie: „and fusing them stays close to the better modality in each regime”.

### 3. Rhythm-leak: spadek 0,73 → 0,00 to w dużej mierze artefakt binarnej cechy
Detektor „temporal layer” korzysta z jednego bitu $tc$. Przy rytmie z pełnej historii legalne zdarzenia z okresu ewaluacji są on-rhythm **z konstrukcji**, więc próg $tc=0$ mieści się w FPR ≤ 1% i czułość równa się odsetkowi ataków off-rhythm (0,73). Przy rytmie z wczesnego okna ponad 1% legalnych zdarzeń wypada off-rhythm. Wtedy **żaden próg** binarnego wyniku nie spełnia FPR ≤ 1%, więc czułość wynosi dokładnie 0, choć AUC to wciąż **0,75** ([Tab. enronsplit](B-Conference/main.tex#L412-L428)).

Wniosek („rhythm-only number is almost entirely leakage”) jest więc mocniejszy, niż pozwalają dane. Przeciek istnieje (AUC 0,86 → 0,75), ale skok czułości do zera wynika głównie z nieciągłości przy α = 1%. **Poprawka:**
- podać czułość także przy FPR 2% i 5% albo partial AUC;
- napisać wprost, że przy binarnym wyniku czułość@FPR jest funkcją schodkową;
- wyjaśnić, czy zdarzenia ataku są wyłączane z estymacji rytmu na pełnej historii. Jeśli tak, to jest to wyciek etykiety, a nie tylko wyciek czasowy, i trzeba to nazwać.

Bez tego recenzent znający metryki progowe podważy główny wynik metodologiczny.

### 4. Sprzeczny ranking detektorów
[main.tex:369](B-Conference/main.tex#L369): „multiplex+rhythm > **contacts > static multiplex** holds across the whole noise range”. Według ablacji ([main.tex:347](B-Conference/main.tex#L347), Fig. ablation) jest **contacts 0,57 → static 0,65**, czyli static multiplex > contacts. Jedna z tych wersji jest błędna. To zależy też od nierozwiązanego TODO o trzech wartościach contacts-only (0,57 / 0,70 / 0,98).

---

## Poważne

### 5. Nadal otwarte TODO z rundy 1, czyli rdzeń zarzutów recenzenta
- **$tc$ dla nadawcy zewnętrznego** ([main.tex:180-181](B-Conference/main.tex#L180-L181)). Bez tej definicji nie wiadomo, co odróżnia cold BEC od cold legit sender. Obaj są „outside every layer”, więc jedyną różnicą jest $tc$. Wynik 0,038 sugeruje, że ta różnica jest pomijalna, ale trzeba to napisać.
- **Shuffle tylko warstw OSINT.** To jedyny eksperyment, który testuje tezę z tytułu. Obecna kontrola daje 0,51, czyli **mniej** niż contacts-only (0,70), co pokazuje, że niszczy wszystko, a nie samą spójność. Dlatego argument w [main.tex:321-322](B-Conference/main.tex#L321-L322) („shuffle … keeps the feature marginals and still falls to chance, so the gain depends on the actual layer structure”) jako dowód „nietautologiczny” jest słaby.
- **Wstrzykiwanie ataków na Enronie** i to, czy kontakty, co-recipients i domena też są estymowane tylko na wczesnych 60% ([main.tex:398-400](B-Conference/main.tex#L398-L400)).

### 6. Def. 2 nie pasuje do estymatora
Def. 2 definiuje Susc dla „credible (domain-knowledge-respecting) attack” z $\mathcal{A}(v)$ = „plausible **impersonating** senders”. Tab. susc liczy go jednak dla (a) cold BEC „outside every layer”, który z definicji nie respektuje wiedzy domenowej, i (b) przejętego konta, które niczego nie podszywa. Albo trzeba rozszerzyć definicję („any attack channel $c$, $\mathrm{Susc}_{D,c}$”), albo nie nazywać tych wierszy realizacją Def. 2.

### 7. Terminologia „organizational susceptibility” jest niespójna
Abstrakt i Def. 2 mówią teraz „detector-relative”, ale tytuł, słowa kluczowe, wkład 4 ([main.tex:107](B-Conference/main.tex#L107)), nagłówek sekcji i podpis Tab. susc nadal mówią „organizational susceptibility”. Do tego [main.tex:139-142](B-Conference/main.tex#L139-L142) wywodzi z prac o *ludzkiej* podatności, że „it can be measured structurally, which we do”. To stawia znak równości, który Def. 2 sama potem odrzuca.

### 8. „Causal” nadal bez zastrzeżeń
[main.tex:194-195](B-Conference/main.tex#L194-L195): „shuffling … destroys **exactly** the information it uses”, [main.tex:216-220](B-Conference/main.tex#L216-L220): „the causal control”, [main.tex:509](B-Conference/main.tex#L509): „a real, **causal** signal”. Po przyznaniu w [main.tex:309-311](B-Conference/main.tex#L309-L311), że kontrola nie izoluje spójności, trzeba to złagodzić („ablation control”). W artykule A zrobiono to w Methodology.

### 9. Headline AUC nie mierzy tytułowego ataku
Ablacja: samo $tc$ daje 0,826 z 0,862. Fig. bythreat: dla nowych nadawców czas jest bezużyteczny (0,50), a dla przejętych kont OSINT jest bezużyteczny (0,50). Zatem 0,70 → 0,87 to w dużej części wykrywanie **przejętych kont** rytmem (temat artykułu A), a nie podszywki zewnętrznej. Abstrakt już mówi „the temporal-consistency feature carries most of the signal”. Warto jednak dodać jedno zdanie, że wzrost AUC pochodzi z kanału lateral, a zysk dla impersonacji leży w redukcji FP. Rozważyć też podanie AUC osobno dla każdego kanału ataku w Tab. inv1.

### 10. Wagi $w_L$ i pokrycie $C(s,v)$ nie są nigdzie używane
Wkład 1 to „per-pair coverage $C(s,v)$”, a Tab. layers podaje ręczne wagi. Wyniki opierają się jednak na LightGBM na wektorze przynależności. $C(s,v)$ nie pojawia się w żadnym eksperymencie (to pkt 18 rundy 1, nadal otwarty). Albo trzeba pokazać detektor oparty na $C$, albo usunąć wagi i zdegradować $C$ do roli intuicji.

### 11. Domena: „one domain per company”, a krawędzie się nie zgadzają
[main.tex:363](B-Conference/main.tex#L363) mówi, że generator „assigns one domain per company”. Wtedy warstwa *Domain* powinna mieć tyle samo krawędzi co *Hierarchy* (518), a Tab. layers podaje **538**. Trzeba wyjaśnić, skąd 20 dodatkowych krawędzi (np. wspólne domeny publiczne?).

### 12. Fabrykacja: teza a wynik
„Fabricating a footprint is cheap but does not create multilayer consistency” ([main.tex:186](B-Conference/main.tex#L186)), a 50% fabrykacji daje 0,87 → 0,74. Brakuje dyskusji, ile sfałszowanych krawędzi potrzeba na jedną ofiarę (pkt 11 rundy 1, nadal otwarty).

---

## Drobne

13. **Enron 818 nadawców (B) vs 1990 kont (A)** przy tym samym korpusie. Jeśli oba artykuły trafią do recenzji równolegle, trzeba to wyjaśnić w obu (różne filtry? okresy?).
14. **„improving AUC under a linear model (0,840 → 0,860) and tying under trees”** ([main.tex:342-344](B-Conference/main.tex#L342-L344)): nie wiadomo, który zestaw daje którą liczbę, a „almost match” kłóci się z „improving”. Proponuję tabelę 2×2 albo jednoznaczne zdanie.
15. **Pogrubione $\mathbf{0.00}$** w Tab. enronsplit: pogrubienie zwykle oznacza najlepszy wynik. Lepiej wyróżnić inaczej albo zostawić bez wyróżnienia.
16. **Wstęp: 98% / „the residual”**: ten sam skrót myślowy co w A (błąd benchmarku ≠ udział spear phishingu).
17. **Data availability:** „is archived on Zenodo” w wersji nieanonimowej, a DOI jeszcze nie ma.
18. **Tekst wspólny z A:** opis organizacji ([main.tex:256-262](B-Conference/main.tex#L256-L262)) i related work prawie słowo w słowo. Ryzyko text recycling i salami slicing jest bez zmian i nadal trzeba je ujawnić redakcji.
19. **Limit stron:** 7 stron według starego PDF-u. Po kompilacji sprawdzić limit docelowej konferencji.
20. **Pozostałe z rundy 1:** baseline treściowy TF-IDF i opis generatora treści, wielkość i balans zbioru syntetycznego, wątek ataków na GNN w related work bez eksperymentu, „co-activity” vs $tc$.

---

## Co się poprawiło od rundy 1
- **Tab. susc jest zinterpretowana poprawnie.** Executive jest najbardziej narażony, dodano rozsądną hipotezę i zastrzeżenie o sprzężeniu w generatorze, a wniosek „seniority does not buy protection” jest trafny.
- Abstrakt mówi wprost, że zysk jest po stronie FP, a cold impersonation ma czułość 0,04.
- Def. 2 ma objaśnienie (rezydualna ekspozycja detektora, wyższa = gorzej).
- W wynikach dodano zastrzeżenie, że kontrola shuffle nie izoluje spójności.
- Fusion w tekście wyników jest opisany uczciwie, a pogrubienia w Tab. content są poprawne.
- Data availability jest objęte `\ifanon`.

## Priorytety
1. Naprawić 2 uszkodzone zdania i „best of the two” w abstrakcie (pkt 1–2). To kilka minut pracy.
2. Uzupełnić rhythm-leak o metrykę niezależną od nieciągłości przy 1% i wyjaśnić, czy ataki są wyłączone z estymacji rytmu (pkt 3).
3. Uzgodnić ranking contacts vs static (pkt 4) i trzy wartości contacts-only.
4. Zdefiniować $tc$ dla zewnętrznego nadawcy i dodać shuffle tylko OSINT (pkt 5).
5. Ujednolicić „susceptibility” i „causal” w całym tekście (pkt 6–8).
