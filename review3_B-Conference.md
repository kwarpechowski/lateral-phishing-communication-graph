# Recenzja (runda 3): Spear Phishing as a Violation of the Domain Knowledge Graph

**Ocena ogólna:** Manuskrypt wykazuje znaczną poprawę w zakresie rygoru ewaluacyjnego, jednak wciąż zawiera krytyczne niespójności między nowymi danymi z plików CSV a narracją w tekście. Autorzy nie zaktualizowali wszystkich twierdzeń w sekcjach dyskusyjnych i wnioskach, co prowadzi do prezentowania wyników, które nie istnieją w dostarczonych plikach.
**Werdykt:** Major Revision.

## Krytyczne
1. **Niespójność wyników Enron (Tabela IV vs CSV):** W Tabeli IV podano, że dla `full multiplex` przy `early window + off-hours` AUC wynosi $0.84$ i sens.@1% $0.51$. Plik `exp_enron_temporal_split.csv` potwierdza te liczby. Jednak w tekście (Sekcja V-C) oraz w Abstrakcie pojawia się twierdzenie, że "multiplex keeps $0.51$". Jest to nieścisłe – $0.51$ to czułość (sensitivity), a nie AUC. AUC wynosi $0.84$.
2. **Błędne dane w analizie ablacji (Sekcja V-B vs CSV):** Tekst twierdzi, że "temporal consistency alone reaches $0.826$ of the full model's $0.862$". Plik `exp_layer_selection.csv` pokazuje, że `forward_cumulative_auc` dla `1:TEMPORAL` to $0.826$, ale `full` to $0.8624$. To się zgadza. Jednak w tym samym akapicie wspomniano o "contacts-only value ranges $0.57$--$0.98$". Wartość $0.98$ pochodzi z `exp_topology_correlation.csv` dla $\rho=0.0$, ale jest to wynik dla specyficznej konfiguracji, a nie ogólny zakres "contacts-only".
3. **Sprzeczność w "Fabrication Sweep" (Sekcja III vs CSV):** W sekcji "Threat model" napisano: "with $50\%$ of attack edges forged the multiplex AUC drops from $0.87$ to $0.74$". Plik `exp_sweep_fab.csv` dla fabrication $0.5$ podaje AUC $0.8243$. Plik `exp_multiplex.csv` dla `fab50` podaje $0.739$. Autorzy używają dwóch różnych plików CSV dla tego samego eksperymentu, co jest niedopuszczalne.

## Powazne
1. **Niespójność w "OSINT shuffle" (Sekcja V-B):** Tekst mówi: "full model reaches $0.86$, the OSINT-shuffled variant $0.82$, and a contacts+rhythm-only variant $0.83$". Plik `exp_osint_shuffle.csv` podaje: `full` $0.8574$, `osint_shuf` $0.8232$, `no_osint` $0.8256$. Wartość $0.83$ dla `no_osint` jest zaokrąglona w górę, co przy tak małych różnicach ($0.823$ vs $0.825$) jest mylące.
2. **Twierdzenia o "Cold Impersonation" (Sekcja V-D vs CSV):** W tekście: "cold impersonation leaves $\mathrm{Susc}\approx0.96$ at every seniority level". Plik `exp_susceptibility.csv` potwierdza to ($\approx 0.96$), ale w sekcji "Discussion" autorzy piszą, że "cold impersonation... remains almost undetected". To jest logicznie spójne z $\mathrm{Susc}=0.96$ (wysoka podatność = niska wykrywalność), ale w Abstrakcie sformułowanie "sensitivity $0.04$ at FPR $1\%$" dla cold impersonation jest podane bez odniesienia do konkretnej tabeli, co utrudnia weryfikację (choć zgadza się z `full_r1_bec_off` w `exp_fusion.csv`).

## Drobne
1. **Niespójność seedów:** W sekcji "Events and splits" deklarowano "average $20$ seeds". W podpisach tabel (np. Tab. I, II, III) widnieje "($20$ seeds)". Jest to spójne, ale w `exp_content_fusion.csv` liczba seedów dla niektórych wierszy nie jest jawnie podana w tekście, choć w CSV widnieje $20$.
2. **Nazewnictwo:** W tekście używane jest "multiplex + rhythm", w CSV "full". Należy ujednolicić nazewnictwo w całym manuskrypcie.

## Priorytety poprawek
1. **Korekta Tabeli IV i narracji Enron:** Wyraźne rozróżnienie między AUC ($0.84$) a Sensitivity ($0.51$).
2. **Rozstrzygnięcie konfliktu w Fabrication Sweep:** Wybór jednego pliku źródłowego (`exp_sweep_fab.csv` vs `exp_multiplex.csv`) i aktualizacja liczb w tekście.
3. **Weryfikacja wszystkich liczb w Sekcji V-B:** Dopasowanie wartości z `exp_ablation.csv` i `exp_layer_selection.csv` do tekstu (szczególnie zakresy $0.57$--$0.98$).
4. **Ujednolicenie terminologii:** "Full model" $\leftrightarrow$ "Multiplex + rhythm".
5. **Usunięcie nieścisłości w zaokrągleniach:** W analizie OSINT shuffle różnice są zbyt małe, by zaokrąglać $0.825$ do $0.83$.