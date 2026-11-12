# Co naprawdę liczymy w tej tabelce — przewodnik popularnonaukowy

> Dokument towarzyszy tabelom wyników artykułu A (Tab. 3–6) i B (Tab. II–VI).
> Cel: wyjaśnić osobie nietechnicznej (lub recenzentowi zmęczonemu liczbami),
*co się dzieje w symulacji*, zanim pojawi się AUC.

---

## 1. Jedno zdanie o czym jest cała praca

Wyobraź sobie, że ktoś **zhackował konto służbowe** w firmie
(`jan.kowalski@firma.pl`) i w jego imieniu wysyła maile do kolegów z pracy.
Taki atak nazywa się *lateral phishing* — „phishing boczny", bo atak
przychodzi **od znajomego adresu wewnątrz organizacji**, nie od fałszywego
księcia z Nigerii. Pytanie badawcze brzmi:

> **Czy sam graf „kto z kim i kiedy pisze" zdradza przejęcie konta —
> nawet gdy treść maila jest idealnie normalna, godziny wysyłki
> wyglądają na zwyczajne, a nadawca nie wysyła niczego „na masę"?**

Żeby na to odpowiedzieć, symulujemy takie ataki na prawdziwych
i sztucznych grafach firmowych i sprawdzamy, które detektory je łapią.

---

## 2. Z czego zbudowany jest eksperyment

### Dane: graf komunikacji = lista zdarzeń

Cały „świat" symulacji to lista prostych zdarzeń:

```
(nadawca, odbiorca, czas, etykieta)
```

np. `(Jan → Anna, wtorek godz. 10:00, legalne)`.
Czas upływa w „kubełkach" (np. godzina) — jeden kubełek to jedna godzina.
Etykieta mówi, czy zdarzenie jest legalne (0), czy to atak (1).
**Uwaga: treść maili nie istnieje w tym świecie.** Widzimy tylko
*kto, do kogo i kiedy* — tak jak serwer pocztowy, który loguje nagłówki.

Trzy źródła danych:

| Graf | Co to jest | Skąd |
|---|---|---|
| **case study** | Sztuczna firma: ~150 osób w 2 firmach, kanały komunikacji, rytmy dnia | generator |
| **Enron** | Realny korpus 500 tys. maili z upadłej firmy Enron (lata 2000–2002) | publiczny zbiór badawczy |
| **SNAP** (email-Eu-core, CollegeMsg) | Realne sieci mailowe uczelni | zbiór Stanford SNAP |

### Atak: zaraźliwa kaskada

Przejęte konto nie pisze do wszystkich naraz (to by było łatwe do
wykrycia — „nagle 50 maili"). Model ataku to **kaskada**:

1. Hacker przejmuje konto Jana („patient zero").
2. Jan pisze do kilku kolegów. Część z nich **zaraża się**
   (z prawdopodobieństwem ~60%) — hacker przejmuje też ich konta.
3. Zarażeni piszą do swoich znajomych... i tak dalej, jak fala.

Ważne: **atak zawsze idzie po istniejących kontaktach** — hacker pisze
do ludzi, z którymi dana osoba naprawdę zwykle rozmawia. Dlatego
pojedynczy mail wygląda zupełnie normalnie.

### Dwa reżimy ataku

- **Blast (naiwny):** zarażone konta wysyłają serię maili w jednej
  godzinie. Łatwy do zauważenia, bo wygląda jak wybuch aktywności.
- **Stealthy (skryty):** każde zarażone konto wysyła **po jednym
  mailu na godzinę** (albo rzadziej), rozłożone w czasie. Bez żadnego
  wybuchu. To jest reżim, który nas naprawdę interesuje — taki atak
  przeprowadziłoby kompetentne państwo cyberprzestępcze.

---

## 3. Detektory — konkurenci w tabeli

Tabelka porównuje metody wykrywania. Każda czyta ten sam strumień
zdarzeń, ale zadaje inne pytanie:

| Detektor | Pytanie, które zadaje | Analogia z życia |
|---|---|---|
| **1-hop** | „Czy Jan zna Annę?" | Ochroniarz sprawdza listę gości |
| **Volumetric (COMPA)** | „Czy Jan dzisiaj wysłał podejrzanie dużo i o nietypowych porach?" | System alarmowy o nietypowej aktywności konta — **to jest klasa metod faktycznie wdrażanych w praktyce** |
| **Path-based (hopper)** | „Czy ta wiadomość ma wyjaśnienie w postaci wcześniejszego zdarzenia?" | Detektyw szukający motywu |
| **Hand-crafted context** | „Czy *sąsiedzi* Anny dostali ostatnio maile? Czy Jan sam coś dostał?" | Analityk, który patrzy na otoczenie odbiorcy |
| **Static GNN** | Sieć neuronowa czytająca strukturę grafu (kto zna kogo) | Fotograf firmy — widzi układ, nie zna czasu |
| **Temporal GNN** | Sieć neuronowa z **pamięcią**: czyta zdarzenia po kolei i pamięta historię każdego węzła | Weteran-firmowiec, który pamięta, jak każdy zwykle pracuje |

Kluczowa idea całej pracy: pierwsze cztery pytania dotyczą **nadawcy**
(Jana). Nasz główny detektor patrzy na **odbiorcę** (Annę): nie pyta
„czy Jan zachowuje się dziwnie?", tylko „czy w otoczeniu Anny
zbudowała się fala?". To nazywamy **sygnałem odbiorczym**
(*receiving-side signal*). Atak może idealnie naśladować normalne
zachowanie nadawcy — ale żeby się rozprzestrzeniać, musi
**dotknąć wielu sąsiadów tej samej osoby w krótkim czasie**,
i to zostawia ślad po stronie odbiorcy.

**Kontrola shuffle:** żeby udowodnić, że sieciowa pamięć naprawdę
czyta *kolejność* zdarzeń, tasujemy czasy wszystkich zdarzeń
i trenujemy jeszcze raz. Jeśli wynik spada do poziomu modelu bez
czasu — sygnał naprawdę był temporalny.

---

## 4. Metryki — czyli co znaczy „0.76"

### AUC: jakość rankingu

Wyobraźmy sobie, że detektor układa **wszystkie zdarzenia w kolejce
od najbardziej podejrzanych do najmniej podejrzanych**. AUC odpowiada
na pytanie: „jeśli wybiorę losowo jeden atak i jeden legalny mail —
jak często atak będzie wyżej w kolejce?"

- AUC = 0.50 → rzut monetą (detektor niczego nie widzi)
- AUC = 0.70 → słabo, ale widzi
- AUC = 0.85 → dobrze
- AUC = 0.95 → bardzo dobrze

**AUC nie mówi nic o progu alarmu** — to czysta jakość sortowania.

### sensitivity@FPR 1%: realny punkt pracy

Tu staje się poważnie. W prawdziwym SOC (zespół bezpieczeństwa)
ustawiamy próg tak, żeby **na 100 legalnych maili fałszywie
zaalarmować maksymalnie 1** (FPR = 1%). I dopiero teraz pytamy:
**jaki ułamek prawdziwych ataków złapiemy przy takim progu?**

- 0.10 = łapiemy 1 na 10 ataków
- 0.20 = 1 na 5

To metryka brutalna — mało detektorów wygląda dobrze przy
FPR 1%. Dlatego właśnie ją lubimy.

### Tabelka „bazowa" (base rate)

Nawet przy sensownej czułości: w prawdziwej firmie ataków jest
ekstremalnie mało (1 na 10 000 maili). Z tabelki Bayesa wynika, że
przy czułości 0.10 i FPR 1% **na każde prawdziwe wykrycie przypadają
~1000 fałszywych alarmów**. Wniosek uczciwie zapisany w pracy:
te detektory to **narzędzia triażu i rankingu** (pomagają ustawić
priorytety analitykom), nie autonomiczne systemy blokujące.

---

## 5. Jak czytać główną tabelę (przykład: Enron, reżim skryty)

```
                       AUC    sens@1%
Volumetric (COMPA)     0.69    0.00
Hand-crafted context   0.87    0.21   (wariant det — patrz pkt 6)
Temporal GNN           0.85    0.22
```

Co to mówi po ludzku:

1. **Volumetric ma 0.00 czułości.** Atak skryty wysyła po jednym
   mailu na godzinę — dla liczników wolumenu *nic się nie dzieje*.
   Metody wdrażane w praktyce są na taki atak **ślepe**.
2. **Sygnał odbiorczy działa** (0.84–0.87 AUC). Fala dotykająca
   sąsiadów odbiorcy zostawia ślad, którego atak nie może
   uniknąć bez rezygnacji z rozprzestrzeniania się.
3. Ale uwaga — to wersja z deterministycznymi odstępami między
   mailami ataku. Po kontroli „emp" (odstępy losowane z realnego
   rozkładu tła) kontekst ręczny spada do ~0.71, a **nauczony
   model temporalny trzyma 0.77** — patrz pkt 6.

### Co oznaczają kolumny det vs emp

Recenzja wykryła, że nasz generator ataków robił coś naiwnego:
wstrzykiwał maile ataku w **idealnie równych odstępach** (co godzinę
dokładnie). Detektory mogły czytać ten „metronom" zamiast fali
odbiorczej. Dlatego każdy wynik skrytego ataku jest teraz liczony
dwa razy:

- **det** — odstępy deterministyczne (jak dotychczas),
- **emp** — odstępy ataku **losowane z rozkładu odstępów prawdziwego
  ruchu w tej samej firmie** (atak pulsuje dokładnie jak tło).

Dodatkowo mierzymy **KS** (test statystyczny: czy czasy ataku w ogóle
odróżniają się od tła — KS ~0.75 = „wyraźnie", ~0.05 = „nie do
odróżnienia") i **sondę czasową** (uproszczony detektor, który widzi
*tylko* czasy, bez grafu — jeśli on osiąga AUC 0.75, to znaczy, że
duża część pozornej skuteczności to artefakt iniekcji, nie sygnał).

---

## 6. Wielki morał całej tabelki

1. **Wolumen pada pod atakiem skrytym** — dokładnie w reżimie,
   który jest realnym zagrożeniem. To jest główna teza (T1).
2. **Sygnał odbiorczy przeżywa** kontrolę iniekcji — ale w wersji
   „emp" na realnych grafach **tylko model nauczony (temporal GNN)
   trzyma marżę nad wolumenem**; cechy ręczne wracają do jego poziomu.
   Teza jest więc mocna, ale węższa niż na pierwszy rzut oka.
3. **Uczciwe granice metody są wpisane w tabelę:** atak
   jedno-odbiorcowy (bez fali) jest niewidoczny dla wszystkich;
   przy transferze na firmę niewidzianą w treningu wygrywa
   wolumen; pod realistycznym szumem etykiet nauczony model
   traci przewagę najszybciej.
4. **W blast wygrywa wolumen** — na realnych grafach (Enron, SNAP)
   wybuch aktywności jest dokładnie tym, co liczniki wolumenu
   lubią. Atak skryty istnieje po to, żeby tego uniknąć.

Zapis na końcówkę: tabela nie mówi „mamy gotowy produkt". Mówi:
**„sygnał istnieje, jest mierzalny, ma znane granice — a metody
stosowane dziś w praktyce nie widzą najgroźniejszej wersji ataku."**

---

## 7. Szybki słowniczek

| Termin | Po ludzku |
|---|---|
| **kubełek (bucket)** | jednostka czasu (godzina) w symulacji |
| **kaskada** | reakcja łańcuchowa przejęć kont |
| **patient zero** | pierwsze przejęte konto |
| **victim split** | podział danych: detektor testowany na ofiarach, których nie widział w treningu (żeby nie był „uczniem, który widział odpowiedzi") |
| **look-ahead leak** | błąd metodologiczny: cecha zdarzenia liczona z użyciem informacji z przyszłości (np. „ile mailów wyśle Jan w ciągu najbliższej godziny") — my go usuwamy, bo w prawdziwym systemie przyszłości nie ma |
| **KS** | test: czy dwa rozkłady czasów (atak vs tło) są statystycznie różne |
| **bootstrap CI** | przedział niepewności: „gdybyśmy powtórzyli eksperyment z nowym ziarnem losowości, wynik w 95% wypadłby w tym zakresie" |
| **shuffle control** | kontrola: tasujemy czasy i sprawdzamy, czy sygnał znika — jeśli tak, był naprawdę czasowy |
| **Cliff's δ** | miara wielkości efektu: jak bardzo jeden detektor jest lepszy od drugiego (duży = przytłaczająco, mały = w granicach szumu) |

---

*Dokument roboczy (nie część submisji). Liczby przykładowe odpowiadają
wynikom fali R4 z 2026-11-12; ostateczne wartości w tabelach artykułu
generuje `analysis/gen_tables.py` bezpośrednio z plików CSV.*
