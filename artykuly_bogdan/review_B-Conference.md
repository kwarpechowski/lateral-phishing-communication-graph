# Recenzja — Artykuł B (konferencja, IEEEtran)

**Tytuł:** *Spear Phishing as a Violation of the Domain Knowledge Graph: Cross-Layer Inconsistency, Organizational Susceptibility, and the Rhythm-Leak Pitfall*
**Autorzy:** K. Warpechowski, B. Księżopolski
**Format:** IEEE conference, 7 stron

**Ocena ogólna:** *weak accept / major revision, zależnie od konferencji.* Najmocniejszym i najbardziej przenośnym wkładem jest **rhythm-leak pitfall** na Enronie. Mechanizm "cross-layer inconsistency" jest za to słabszy, niż sugerują tytuł i abstrakt: sygnał niesie głównie rytm, a nie warstwy OSINT, a sam atak tytułowy (zewnętrzna podszywka) jest praktycznie niewykrywany (czułość 0,038). Do tego interpretacja tabeli podatności jest **odwrócona** względem liczb.

---

## Mocne strony

1. **Rhythm-leak pitfall to konkretny, wartościowy wynik metodologiczny.** Estymacja rytmu na pełnej historii zawyża czułość detektora rytmu z 0,00 do 0,73 ([Tab. enronsplit](B-Conference/main.tex#L396-L412)). Wynik jest na danych rzeczywistych, łatwy do powtórzenia i przydatny dla całej klasy detektorów behawioralnych. To instancja ostrzeżeń Arp et al. i TESSERACT pokazana konkretnie dla e-maila.
2. **Wyjątkowa uczciwość.** Autorzy sami mówią, że rytm niesie większość sygnału ([main.tex:335-339](B-Conference/main.tex#L335-L339)), nazywają cyrkularność definicji nowego nadawcy ([main.tex:307-313](B-Conference/main.tex#L307-L313)) i nie twierdzą, że uporządkowanie według seniority to odkrycie. Recenzenci to docenią.
3. **Dobra kontrola artefaktu generatora.** Knob dekorelacji ρ pokazuje, że prawie identyczne warstwy strukturalne to cecha populacji, a nie metody ([Tab. topo](B-Conference/main.tex#L360-L374)).
4. **Analiza per zagrożenie** ([Fig. bythreat](B-Conference/main.tex#L376-L381)): OSINT jest kluczowy dla nowych nadawców, czas dla przejętych kont. To czysty i intuicyjny rozdział ról.
5. **Fusion graf + treść.** Pokazanie, że treść i graf zawodzą na różnych atakach (tell-tale vs mimicry), to dobra motywacja do systemu hybrydowego.
6. **Dobrze zaprojektowana populacja "graph-first"** z jawnym schematem 8 kategorii i wagami warstw ([Tab. layers](B-Conference/main.tex#L273-L295)).
7. **Bootstrap CI** w tabeli podatności, czego brakuje w wielu pracach konferencyjnych.
8. **Interpretowalność.** Cztery jawne miary spójności prawie dorównują pełnej konkatenacji warstw (0,858 vs 0,860), co jest dobrym argumentem praktycznym.

---

## Słabe strony

### Krytyczne

1. **Interpretacja Tab. susc jest odwrócona.** Według tabeli executive ma **najniższą** czułość (0,076), czyli **najwyższe** Susc (0,924), więc to executive jest **najbardziej** narażony. Tekst mówi odwrotnie ([main.tex:428-431](B-Conference/main.tex#L428-L431)): "executives... are **best protected** at 0,924, while... junior roles are **most exposed** at 0,838–0,840". Wniosek "what protects is embedding in the graph rather than status" ([main.tex:435](B-Conference/main.tex#L435)) wymaga więc ponownego przemyślenia: jeśli executive ma najwięcej warstw i *mimo to* jest najbardziej narażony, to bogatsze osadzenie w grafie raczej **szkodzi**, np. dlatego, że atakujący łatwiej udaje spójność. Trzeba poprawić tekst lub tabelę. Recenzent wychwyci to od razu.

2. **Atak tytułowy jest prawie niewykrywalny.** Tytuł i abstrakt dotyczą zewnętrznej podszywki (external impersonation), a dla cold BEC czułość przy FPR 1% wynosi **0,038 [0,00; 0,115]** (Susc 0,96). Większość "sukcesu" mechanizmu dotyczy zatem albo (a) **nie flagowania** prawdziwych nowych nadawców (redukcja FP, częściowo z konstrukcji), albo (b) **przejętych kont** via rytm, czyli tematu artykułu A. Abstrakt powinien jasno mówić, że wartość metody leży po stronie redukcji fałszywych alarmów, a nie wykrywania podszywek. Można też rozważyć zmianę tytułu.

3. **Kontrola shuffle nie testuje tezy o *cross-layer*.** Tasowanie "layer membership and rhythm" niszczy *wszystkie* cechy, łącznie z kontaktami i rytmem, więc spadek do 0,51 jest oczywisty (nawet contacts-only daje 0,70). Żeby pokazać, że liczy się *spójność między warstwami*, trzeba tasować **tylko warstwy OSINT** (zostawiając kontakty i rytm) albo tasować przynależność *niezależnie* w każdej warstwie, z zachowaniem marginesów. Dopiero wtedy spadek z 0,87 do ~0,8x świadczyłby o wkładzie spójności. W obecnej formie zdanie "the advantage stems from actual cross-layer structure, not from feature capacity" ([main.tex:301-302](B-Conference/main.tex#L301-L302)) nie jest uzasadnione.

4. **Rytm dla nadawcy zewnętrznego jest niezdefiniowany.** tc(s,v,t)=1[t∈τ(s)] wymaga znanego rytmu nadawcy. Dla zewnętrznego podszywacza (cold BEC, "outside all layers") τ(s) nie istnieje. Trzeba opisać, jak tc jest liczone dla takich zdarzeń. Jeśli z definicji wynosi 0, to rytm staje się po prostu wskaźnikiem "nieznany nadawca" i to on, a nie warstwy, odpowiada za sygnał.

### Poważne

5. **Fusion nie daje "the best of both".** W reżimie mimicry fusion = **0,848 < 0,862** (sam multiplex), a mimo to wartość fusion jest pogrubiona i opisana jako "fusion yields the best of both" ([main.tex:469-470](B-Conference/main.tex#L469-L470), [Tab. content](B-Conference/main.tex#L474-L485)). Należy napisać "nearly matches the better of the two" i usunąć pogrubienie.

6. **Niespójne liczby między tabelą, tekstem a figurą:**
   - contacts-only: 0,70 (Tab. inv1) vs **0,57** (ablacja, Fig. ablation) vs **0,98** "contact AUC" przy ρ=0 (Tab. topo);
   - pełny model: 0,87 (Tab. inv1) vs 0,862 (ablacja);
   - Enron, realistyczna czułość: "~0,49 rather than 0,87" w tekście ([main.tex:387](B-Conference/main.tex#L387)) vs 0,51 / 0,88 w Tab. enronsplit.

   Jeśli to różne konfiguracje (inne proporcje zdarzeń, inny klasyfikator), każda tabela musi to jasno mówić.

7. **Na Enronie nie wiadomo, jak powstają ataki.** Sekcja Method opisuje budowę multipleksu (818 nadawców, ~170k zdarzeń), ale nie opisuje, *jak wstrzykiwane są zdarzenia ataku* (kto jest atakującym, jaki ma rytm, ile ich jest). Bez tego wynik AUC 0,94 przy "legitimate traffic on-rhythm" wygląda trywialnie: jeśli legit z definicji jest on-rhythm, a atak off-rhythm, to rytm separuje klasy idealnie.

8. **Czy pozostałe warstwy na Enronie też nie przeciekają?** Rytm jest liczony na wczesnych 60%. A kontakty, co-recipients i domena? Jeśli są liczone na pełnej historii, to "full multiplex keeps 0,51" może zawierać ten sam typ wycieku, który artykuł demaskuje. Trzeba to jawnie zapisać.

9. **"Susceptibility" to myląca nazwa.** W related work podatność to cecha człowieka (osobowość, presja czasu). Tu Susc = 1 − czułość *detektora*, czyli miara luki detektora, a nie podatności organizacji. Wniosek "channel dominates rank" wynika wprost z tego, że detektor ma inne cechy dla każdego kanału, a nie z właściwości organizacji. Proponuję "detector-relative exposure" lub "residual exposure", co jest już zgodne z Def. 2.

10. **Cyrkularność głębsza niż przyznana.** Nie tylko nowy nadawca z definicji leży na warstwie OSINT. Także atak z definicji jest "outside all layers, off-rhythm" ([main.tex:225-226](B-Conference/main.tex#L225-L226)). Obie klasy są więc konstruowane z tych samych cech, których używa detektor. Kontrola fabrykacji (do 50% fałszywych krawędzi) częściowo to łagodzi, ale potrzebny jest scenariusz, w którym atakujący *celowo* fabrykuje krawędzie OSINT do konkretnej ofiary (realistyczne: wspólna konferencja jest tania do zasymulowania).

11. **Teza "fabrication is cheap but does not create multilayer consistency" jest w napięciu z wynikiem**, że 50% fabrykacji obniża AUC 0,87 → 0,74. Trzeba to przedyskutować: ile fałszywych krawędzi realnie potrzebuje atakujący na pojedynczą ofiarę?

12. **TF-IDF jako baseline treściowy jest bardzo słaby,** a nie wiadomo, jak generowana była treść maili (LLM? szablony?). Wynik "content blind w mimicry 0,502" jest wtedy zdeterminowany przez generator. Trzeba opisać źródło treści i dodać choćby jeden model transformerowy.

13. **Brak opisu wielkości i balansu zbioru** syntetycznego: liczba zdarzeń, proporcja klas, liczba ataków na seed, sposób wyboru ofiar.

### Drobniejsze

14. **Za wąska nowość metodologiczna rhythm-leak:** to w gruncie rzeczy klasyczny temporal leakage (TESSERACT). Warto podkreślić *skalę* (0,73 → 0,00) i *łatwość popełnienia błędu* w praktyce, zamiast przedstawiać to jako nowe zjawisko.
15. **Related work cytuje ataki i obrony na GNN** (Zügner, GNNGuard itd.), ale w artykule nie ma eksperymentu z adwersarzem grafowym. Trzeba albo dodać eksperyment, albo skrócić wątek.
16. **Nieużywane figury w paczce:** fig_layer_selection, fig_multiplex_3d, fig_sensitivity. Należy je usunąć lub wykorzystać (np. 3D multipleks ładnie zilustrowałby Def. 1 zamiast samej organizacji).
17. **Anonimizacja:** jeśli konferencja jest double-blind, poza `\anontrue` trzeba też ukryć odniesienia do "separate study" (artykuł A), DOI Zenodo i paragraf Acknowledgment (zrobione przez `\ifanon`, ale "Data and Code Availability" nie jest objęte).
18. **Wagi w_L** ([Tab. layers](B-Conference/main.tex#L273-L295)) są ustawione ręcznie bez uzasadnienia, a klasyfikator (LightGBM) i tak uczy się własnych wag. Trzeba wyjaśnić, gdzie C(s,v) z wagami jest faktycznie używane.
19. **Warstwa "Temporal: co-activity" (2528 krawędzi) i cecha tc to dwie różne rzeczy** o podobnej nazwie. Lepiej je rozróżnić w tekście.
20. **Limit stron:** 7 stron. Sprawdzić limit docelowej konferencji (często 6 + referencje).

---

## Relacja z artykułem A

- Wspólne studium przypadku, wstęp (statystyki BEC, "98%") i related work prawie słowo w słowo. Trzeba przeformułować i ujawnić powiązanie (ryzyko zarzutu *salami slicing* lub *text recycling*).
- Kanał "compromised account / lateral" w B jest wykrywany rytmem, czyli tym samym sygnałem off-rhythm co w A. Warto jasno zaznaczyć, że B nie twierdzi niczego nowego o tym kanale.

## Priorytety poprawek (skrót)

1. **Odwrócona interpretacja Tab. susc** (pkt 1): to błąd faktyczny.
2. Uczciwie przeformułować abstrakt i tytuł wokół redukcji FP oraz rhythm-leak, a nie wykrywania podszywek (pkt 2).
3. Zastąpić shuffle kontrolą tasującą tylko OSINT (pkt 3) i zdefiniować tc dla nadawcy zewnętrznego (pkt 4).
4. Poprawić "best of both" (pkt 5) i uzgodnić liczby (pkt 6).
5. Opisać wstrzykiwanie ataków i ewentualny wyciek pozostałych warstw na Enronie (pkt 7–8).
