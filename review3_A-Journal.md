# Recenzja (runda 3): Lateral Phishing as a Violation of the Organizational Communication Graph

**Ocena ogólna:** Manuskrypt wykazuje znaczną poprawę w zakresie rygoru metodologicznego i spójności danych po naprawie wycieków. Jednakże, w tekście wciąż występują pozostałości starej narracji (twierdzenia o wyższości modelu temporalnego nad ręcznie tworzonym kontekstem), które są bezpośrednio sprzeczne z załączonymi wynikami statystycznymi (p=0.62). 
**Werdykt:** Minor Revision (wymagana korekta narracji w celu dostosowania jej do dowodów statystycznych).

## Krytyczne
1. **Sprzeczność z dowodami statystycznymi (Overclaiming):** 
   - **Sekcja `en/results.tex` (Sec. 6.4):** Autor twierdzi, że "the temporal GNN... leads from $K=3$ on". 
   - **Sekcja `en/results.tex` (Sec. 6.7):** Autor pisze, że "the receiving-side signal beats the volume detector significantly... the temporal GNN and the hand-crafted context are statistically indistinguishable at this point ($p=0.62$)".
   - **Problem:** W wielu miejscach (np. Sec. 6.3, 6.4) narracja sugeruje, że Temporal GNN jest "liderem" lub "najlepszym" detektorem. Tymczasem plik `cb_headline_contrasts.csv` oraz analiza w Sec. 6.7 wprost mówią, że różnica między `temporalny-GNN` a `reczny-kontekst` jest "znikoma" (dAUC=0.01, p=0.62). Autor nie może twierdzić o wyższości modelu GNN nad kontekstem ręcznym, jeśli test Wilcoxona tego nie potwierdza.
2. **Niespójność liczb w tabelach vs CSV:**
   - **Tabela 2 (Tab. 2):** Wartość sensitivity@FPR1% dla Temporal GNN podana jako $0.121$. W `exp_cascade.csv` wartość to $0.1207$ (poprawnie zaokrąglone). Jednak dla "Temporal (time shuffled)" podano $0.026$, podczas gdy w CSV jest $0.0262$.
   - **Tabela 3 (Tab. 3):** Dla $K=1$ AUC dla `temp.` podano $0.59$. W `exp_cascade_stealth.csv` jest $0.5918$. Dla $K=5$ AUC `temp.` podano $0.76$, w CSV jest $0.7577$.
   - **Tabela 4 (Tab. 4):** W bloku (a) dla $K=1$ AUC `temp.` podano $0.67$. W `cb_hopper_stealth.csv` jest $0.6748$. Dla $K=3$ AUC `temp.` podano $0.72$, w CSV jest $0.7182$.
   - **Wniosek:** Liczby są w przybliżeniu zgodne, ale autor stosuje niespójne zaokrąglenia (raz w górę, raz w dół), co w rygorystycznym czasopiśmie jak JNCA powinno być ujednolicone do 3 miejsc po przecinku.

## Powazne
1. **Pozostałości starej narracji (Stale twierdzenia):**
   - **Sekcja `en/results.tex` (Sec. 6.3):** Fraza "leaving the temporal GNN in the lead" jest zbyt silna w obliczu wyników z Tab. 4, gdzie przy $K=1$ i $K=2$ detektor `path` i `manual` (hand-crafted) mają wyższe AUC ($0.76$ i $0.75$ vs $0.67$).
   - **Sekcja `en/results.tex` (Sec. 6.6):** Twierdzenie, że "the temporal GNN... beating the hand-crafted context on average across seeds" na Enron. W `exp_cascade_enron.csv` Temporal GNN ma $0.8308$, a Tab_ctx $0.7714$. Tutaj różnica istnieje, ale w kontekście ogólnym (synthetic) jest ona nieistotna statystycznie.
2. **Logika "Blind Spot":**
   - Autor słusznie identyfikuje fan-out=1 jako blind spot (AUC $\sim 0.50$). Jednak w Sec. 6.5 twierdzi, że "the signal needs fan-out, not a cascade". Jest to logicznie spójne z wynikami w `cb_surgical_fanout.csv`, ale wymaga wyraźniejszego zaznaczenia w podsumowaniu, aby nie mylić "sygnału kaskady" z "sygnałem synchronizacji odbiorców".

## Drobne
1. **Niespójność seedów:** 
   - W `en/results.tex` (Sec. 6.1) mowa o 20 seedach dla stealthy attacker, ale w Tab. 2 (Cascade detectors) jest 5 seedów. W Tab. 4 jest 20. W Tab. 5 (Real graphs) jest 8 i 3. Choć jest to opisane w podpisach, w tekście głównym (Sec. 6.7) brakuje jasnej tabeli zbiorczej z liczbą seedów dla każdego eksperymentu.
2. **Referencje:** Brak wykrytych `??`, ale należy sprawdzić, czy `\citep{birthriya2025detection}` i `\citep{sturman2025security}` są aktualne (daty 2025 sugerują preprinty/forthcoming).

## Priorytety poprawek (max 5)
1. **Korekta narracji:** Usunięcie sformułowań "in the lead" lub "beats" w odniesieniu do Temporal GNN vs Hand-crafted context w reżimach, gdzie $p=0.62$. Zastąpienie ich stwierdzeniem o "porównywalnej skuteczności".
2. **Ujednolicenie zaokrągleń:** Wszystkie liczby w tabelach muszą być spójne z CSV (zalecane 3 miejsca po przecinku).
3. **Doprecyzowanie Tab. 4:** Wyraźniej zaznaczyć w tekście, że przy ekstremalnym stealth ($K=1$) detektory oparte na ścieżkach/kontekście są skuteczniejsze niż GNN, co osłabia tezę o bezwzględnej wyższości modelu temporalnego.
4. **Weryfikacja spójności:** Sprawdzenie, czy w całej sekcji Discussion nie pozostały frazy sugerujące, że GNN "rozwiązuje" problem lepiej niż metody heurystyczne.
5. **Tabela seedów:** Dodanie w Supplementary lub w tekście jednej, przejrzystej tabeli: `Eksperyment` $\to$ `Liczba seedów`.