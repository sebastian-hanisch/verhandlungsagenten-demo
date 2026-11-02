# Verhandlungsagenten: was trägt gegen unbekannte Gegner? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-verhandlungsagenten-demo.streamlit.app/)** -->

Stück der Linie **regeln-verhandeln** (Arbeitsname; Name, Bereich und Themenfeld werden erst bei der Einbindung in die Website festgelegt) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Bereich Kooperation und Wettbewerb. Es ist das Gegenstück zu [`verhandlung-demo`](https://github.com/sebastian-hanisch/verhandlung-demo), das das Gleichgewicht eines Ablaufs mit **einem** Anteil und bekannten Nutzen rechnet: hier **spielen Programme** gegeneinander, die den Nutzen des Gegners nicht kennen. Und es folgt auf [`kuchen-vergroessern-demo`](https://github.com/sebastian-hanisch/kuchen-vergroessern-demo), das zeigt, dass es bei mehreren Punkten Gewinn gibt; hier wird gemessen, welche Strategie ihn gegen unbekannte Gegner hebt. Vehikel: ein **Verlader** und ein **Spediteur** verhandeln 3 bis 5 Punkte (Preis, Lieferfenster, Zahlungsziel, Haftung, Rückladung) mit je 3 bis 5 Werten, Nutzen auf 0 bis 1 normiert. **Strategien, Nutzen und Fristen sind Modellannahmen, keine Aussagen über echte Verhandler oder Sprachmodelle.**

Abgrenzung: `axelrod-turnier-demo` lässt 16 Strategien im wiederholten Gefangenendilemma gegeneinander antreten (ein Spiel, kein Verhandlungsablauf mit Frist und mehreren Punkten). `tit-for-tat-demo`, `rauschen-kooperation-demo` und `zero-determinant-demo` zeigen Tit-for-Tat als Automaten im Dilemma (Diskontfaktor, Vergeltung); **gleicher Name, anderer Mechanismus:** hier ist „Tit-for-Tat (relativ)“ ein Nachgeben im Verlauf (ich gebe so viel nach, wie der andere mir entgegenkommt). `noregret-demo` lernt Strategien in einem anderen Spiel, `bandit-demo` und `q-learning-demo` lernen im Bandit bzw. im Raster ohne Verhandlungsbezug (der Q-Lerner hier ist neu und klein geschrieben, keine Kopie). `gewichte-erheben-demo` fragt die Gewichte eines Entscheiders ab; hier schätzt ein Programm die Gewichte des Gegners aus dessen Angeboten. `pareto-dominanz-demo` kennt Dominanz, aber keine zwei Seiten.

## Kernfrage

*Zwei Programme handeln mehrere Punkte einer Tour aus und kennen die Nutzen des anderen nicht: welche Strategie trägt, was lernt ein Gegnermodell, und wann scheitern beide?* Die Antwort hat sechs Teile (alle Zahlen: vier Zufalls-Seeds, Pool aus sechs Strategien, 4 Punkte × 4 Werte, Reservationsnutzen 0,2, Regel „letzte Runde“ an, wenn nicht anders genannt). (1) **Conceder, der früh nachgibt, verliert:** er liegt in 9 von 9 Zellen (Frist × Gewichtsunterschied) auf dem letzten Rang, auch mit den Exponenten 2 und 10 (Robustheit). (2) **Die Spitze ist nur bei längerer Frist eine Gruppe:** bei T = 20 und s = 0,5 führen Boulware + Modell (0,574), Boulware (0,569) und Tit-for-Tat (0,558) innerhalb von 0,016, ähnlich eng bei T = 20 mit s = 1 und bei T = 60; bei T = 6 führt Tit-for-Tat in allen drei Gewichtsunterschieden, bei T = 20 und s = 0 ebenfalls, dort als Einzelsieger; Boulware + Modell liegt nur in 4 von 9 Zellen vor Boulware. (3) **Mit der Regel „letzte Runde“ zahlt sich Hartnäckigkeit gegen Boulware, Linear und Conceder nur für den nicht zuletzt Bietenden aus:** 0,716 als erste Position gegen 0,248 als zweite bei T = 20, umgekehrt bei T = 21 (0,268 gegen 0,720). (4) **Das Gegnermodell verbessert die Gewichtsschätzung nur wenig und wird gegen einen Conceder getäuscht:** nach 20 Angeboten eines linearen Gegners ist der Gewichtsabstand 0,172 (Raten gleicher Gewichte: 0,220), gegen einen Conceder wird die Rangkorrelation −0,398 (gegen Tit-for-Tat schwach positiv, gegen Hartnäckig, Boulware und Linear deutlich positiv); es schließt trotzdem 40,3 % der Summenlücke zum Orakel. (5) **Die Zerlegung in Gebot, Annahme und Gegnermodell hat keine feste Rangfolge:** beim eigenen Nutzen trägt die Annahme 0,461 der Varianz, die Kurve 0,402, das Modell 0,008; bei der Summe die Kurve 0,719 und das Modell 0,186; ohne eine Stufe steigt der Anteil der Kurve am eigenen Nutzen auf 0,910. (6) **Ein kleiner Q-Lerner liegt im Mittel vorn (0,657 gegen 0,579 für Boulware + Modell nach 20.000 Episoden), gegen Linear und Conceder aber dahinter; gemessen nur gegen diesen Pool und diesen Generator** und nicht mit mehr Training besser (50.000 Episoden: 0,640).

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (Vormessung vom 2026-11-01) wurde nicht blind übernommen: jede Zahl ist im Bau neu gemessen, mit eigenem SplitMix64-Strom statt `numpy.random` und mit Standardabweichungen über vier Zufalls-Seeds (n − 1; der Plan rechnete mit n). Abweichungen und neue Funde:

- **Boulware + Modell liegt nicht „in den Zellen T ≥ 20, s ≥ 0,5 vor Boulware um 0,002 bis 0,011“** (Plan): in 4 von 9 Zellen vor Boulware, der Unterschied reicht von −0,004 bis +0,009 und ist in 7 von 9 Zellen kleiner als die größere der beiden Standardabweichungen über die Seeds (höchstens 0,007); bei T = 60 und s = 0,5 ist es ein Gleichstand (−0,0001), der Sieger dieser Zelle ist Boulware (2 von 4 Seeds). Die Aussage „Spitzengruppe“ trägt, die Aussage „Boulware + Modell vorn“ nicht.
- **Lernender:** Training mit 50.000 Episoden (jeweils eigener Lauf mit voller ε-Abnahme) ist nicht besser als mit 20.000 (0,640 gegen 0,657, Std 0,015 gegen 0,005); der Plan hatte Prüfpunkte eines einzigen langen Laufs (0,618 / 0,648 / 0,658, monoton). Bei der Bedingung s = 1 liegen 3 von 4 Trainings-Seeds vor beiden festen Strategien (Plan: 2 von 4).
- **Endspiel-Ausnutzer** (offener Punkt des Plans) ist als siebte Strategie im Auswahlfeld und im Live-Turnier, **nicht** im Pool der Raster-Zellen; gemessen wird er in drei Pools (6, 7, 9 Strategien, E3). Ohne Gegnermodell hat das Endspiel keine Wirkung (toter Zweig, im Test dokumentiert), darum gibt es die Strategie nur mit Häufigkeitsmodell.
- **Regel „letzte Runde“** (offener Punkt): Voreinstellung an, wählbar, Wirkungstabelle in E3.
- **Domäne mit Stück G:** keine gemeinsame Domäne und keine Kopie von G; der Spezialfall m = 2, g = 1, r = 0 wird im Orakel-Test gegen die Reihenfolge-Regel von G geprüft (hier unabhängig neu geschrieben: alle 360 Ecken liegen auf der Aufzählungs-Grenze).
- **Zerlegung** mit Stufenwahl statt festem Rangsatz (offener Punkt): vier Stufenmengen in der Tabelle.
- **Domänenstatistik** (Nash-Summe nach Gewichtsunterschied) ist live (400 Domänen je Zeile), nicht vorgerechnet.
- **Kein Mutationstest** (DEMO-PLAYBOOK §12 kennt ihn nicht mehr); stattdessen Orakel-, Eigenschafts-, Zweig- und Randfalltests.

## Modell

Eine Domäne hat *n* Punkte mit je *m* Werten (Bündel = ein Wert je Punkt, *mⁿ* Bündel). Jede Seite *p* hat Gewichte *w_p* (Summe 1) und je Punkt eine Wertliste auf [0, 1] mit Minimum 0 und Maximum 1; der Nutzen eines Bündels ist die gewichtete Summe, der größte Nutzen einer Seite ist 1. Der **Gewichtsunterschied** *s* mischt die Gewichte (0: gleich, 1: unabhängig, Dirichlet(1)), der **Gegensatz** *g* die Wertlisten (0: unabhängig, 1: genau gegenläufig). Ohne Einigung bekommt jede Seite den **Reservationsnutzen** *r*. Der Ablauf: abwechselnde Angebote, der Antwortende nimmt an oder lehnt ab, Frist *T* Runden, kein Diskont; die erste Position bietet in den ungeraden Runden (bei geradem *T* bietet die zweite zuletzt). **Regel „letzte Runde“** (wählbar): in Runde *T* nimmt der Antwortende jedes Angebot ab seinem Reservationsnutzen an.

**Gebot:** Zielnutzen *U*(*t*) = 1 − (1 − *r*) τ^(1/β), τ = (*t* − 1)/(*T* − 1) (Boulware β = 0,2, Linear β = 1, Conceder β = 5; Formelfamilie laut Sekundärliteratur, Faratin u. a. 1998 nicht im Volltext gelesen), Hartnäckig (Ziel 1), Tit-for-Tat (relativ), und die Wahl **auf der Isolinie** (Fenster 0,05 über dem Ziel): unter den Bündeln im Fenster das mit dem größten geschätzten Gegnernutzen (Häufigkeitsmodell oder Orakel), ohne Modell ein fester Zufallsvektor je Domäne. **Annahme:** ab Reservation, Schwelle, linearer Zeitplan oder „nächstes eigenes Gebot“ (die Strategien des Pools nutzen das letzte). **Lerner:** Q-Tabelle mit 90 Zuständen und 9 Zielstufen. Formeln, Lesehilfe und Symboltabelle stehen im 📐-Expander der App.

## Methodik

- **Orakel** (`tests/test_oracle_verhandlungsagenten.py`, anderer Rechenweg): Pareto-Grenze und Nash-Lösung durch den Sweep im Kern gegen die O(N²)-Schleife in reinem Python (in 100 Zufallsdomänen) und gegen Maximierer gewichteter Summen (300 Fälle); der Spezialfall m = 2, g = 1 gegen die Reihenfolge-Regel von G; an der Mini-Domäne (`tests/conftest.py`) alles von Hand: neun Nutzenpaare, Grenze {0, 1, 2, 5, 8}, rationale Grenze {1, 2, 5}, Nash-Bündel 2 mit Produkt 0,20, Zielkurven 0,975 / 0,6 / 0,30356 bei τ = ½, drei Handläufe mit *T* = 3 und das Häufigkeitsmodell (Gewichte 0,4 / 0,6 nach den Angeboten 2, 2, 5).
- **Eigenschaftstests** (`tests/test_eigenschaften_verhandlungsagenten.py`): Umnummerieren der Punkte und Werte, Determinismus, gleichgerichtete Domäne, Rollentausch im Selbstspiel, Kennzahlen konsistent, Strategiewerte unabhängig von der Reihenfolge im Pool. Eine affine Umrechnung der Nutzen ist keine Eigenschaft (die Zielkurven setzen den größten Nutzen 1 voraus).
- **Zweig-Tests** (`tests/test_branches.py`): jede abschaltbare Einheit (Annahmeregel, Gegnermodell, Endspiel, Fenster, Regel „letzte Runde“, Tit-for-Tat-Anteil) weicht auf mindestens einer Instanz ab; dokumentierte Gleichheit: Tit-for-Tat gegen Tit-for-Tat oder Hartnäckig ist zahlengleich Hartnäckig gegen Hartnäckig (40 von 40 Domänen).
- **Referenzsoftware und Literaturwerte:** keine (GENIUS und NegMAS liegen nicht im Test-venv, ein Literaturwert für diese Strategien liegt nicht vor): **offen**, kein Test aus der Erinnerung.
- **Messreihen** (`tools/make_grid.py`, Datei `data/messreihen.json` mit Hash der Quellen): 480 Aufträge, jeder mit festen Seeds aus seiner Bezeichnung (FNV-1a, SplitMix64); die Datei wird in `tests/test_measurements.py` stichprobenweise neu gerechnet (zehn Aufträge).
- **Zufall:** `vha_rng.py` ist eine wörtliche Kopie von `ab-test-demo/abt_rng.py` (SplitMix64) mit Kopietest.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (die Tabellen werden mit denselben Zeilenbauern wie in der App erzeugt und dort als Text in der README gesucht). „Std“ ist die Standardabweichung (n − 1) über vier unabhängige Zufalls-Seeds; eine Zelle des Rasters besteht aus 4 Zufalls-Seeds zu je 100 Domänen (Endspiel, Zerlegung und Robustheit: 50), jede Strategie spielt gegen jede andere in beiden Positionen, **der Wert einer Strategie ist das Mittel über die fünf anderen**. Die Höhe der Nutzen hängt am Erzeuger der Domänen, nicht an der Wirklichkeit.

| Nr. | Hypothese | Stand | Befund |
|---|---|---|---|
| H1 | Conceder ist im Pool die schwächste Strategie | **Ja** | Letzter in 9 von 9 Zellen (T ∈ {6, 20, 60} × s ∈ {0; 0,5; 1}) und in 36 von 36 Zelle-Seed-Paaren; in 17 von 18 Reglerwerten der Robustheit (Ausnahme Reservationsnutzen 0,5: dort ist Hartnäckig Letzter) |
| H2 | Bei längerer Frist ist die Spitze eine Gruppe (Boulware, Boulware + Modell, Tit-for-Tat); ihre Reihenfolge hängt an Frist, Gewichtsunterschied und Pool | **Ja**, die Reihenfolge nicht | (T = 20, s = 0,5): Boulware + Modell 0,574, Boulware 0,569, Tit-for-Tat 0,558; Boulware + Modell liegt in 4 von 9 Zellen vor Boulware (−0,004 bis +0,009), bei T = 60, s = 0,5 gleichauf; im Pool mit zwei Orakel-Agenten liegt Boulware (0,596) vor Boulware + Modell (0,591), in den 50 Domänen je Seed der sechs Standardstrategien umgekehrt (0,563 gegen 0,566) |
| H3 | Tit-for-Tat gewinnt bei kurzer Frist | **Ja** | T = 6: Rang 1 in 3 von 3 Gewichtsunterschieden (je 4 von 4 Seeds); auch bei T = 7 und beim Boulware-Exponenten 0,1; bei T = 20 und s = 0 ebenfalls (4 von 4); ab T = 20 mit s ≥ 0,5 nicht mehr Erster |
| H4 | Hartnäckigkeit nutzt Zugeständnisse aus | **Nur für den nicht zuletzt Bietenden** (mit der Regel „letzte Runde“) | gegen Boulware, Linear und Conceder (dieselben Nutzen), Standarddomäne, T = 20, gegen Boulware: 0,716 (Std 0,019) als erste Position, 0,248 (Std 0,007) als zweite, die zuletzt bietet; T = 21 umgekehrt: 0,268 gegen 0,720. Ohne die Regel „letzte Runde“: 0,258 und 0,238. Hartnäckig gewinnt außerdem bei Reservationsnutzen 0 und bei Gegensatz g = 0 (je 0,900, 4 von 4 Seeds) |
| H5 | Zwei Hartnäckige scheitern | **Ja** | Einigungsrate 0,090 bei T = 6, 20 und 60 (s = 0,5; 0,06 bis 0,10 je nach s und Domänenmenge); Tit-for-Tat gegen Tit-for-Tat ist zahlengleich (gibt nur nach, wenn der andere nachgibt) |
| H6 | Das Häufigkeitsmodell lernt Gewichte schnell und gut | **Nein** | Gewichtsabstand nach 20 Angeboten 0,172 (Linear), 0,197 (Boulware), 0,205 (Conceder), 0,210 (Tit-for-Tat), 0,220 (Hartnäckig) gegen 0,220 beim Raten gleicher Gewichte; gegen Conceder Rangkorrelation −0,398 nach 20 Angeboten (+0,070 nach 8): das Modell wird gegen Conceder getäuscht |
| H7 | Das Modell schließt einen Teil der Lücke zum Orakel (Summe, Abstand), nützt dem eigenen Nutzen wenig | **Ja** | Boulware: 40,3 % der Summenlücke (1,034 → 1,085 → 1,161), 41,9 % der Abstandslücke; eigener Nutzen 0,551 → 0,573 → 0,608; Linear 44,2 %, Conceder 47,3 % der Summenlücke; bei Conceder **sinkt** der eigene Nutzen mit Modell (0,395 → 0,389 → 0,384), die Summe steigt |
| H8 | Rangfolge der Bausteine Gebot > Annahme > Gegnermodell (laut Abstract nach Auftrag, nicht geprüft) | **Nein** | hängt an Kennzahl und Stufen: eigener Nutzen Akzeptanz 0,461 > Kurve 0,402 > Modell 0,008; Summe Kurve 0,719 > Modell 0,186 > Akzeptanz 0,032; ohne die Stufe „ab Reservation“ trägt die Kurve 0,910 des eigenen Nutzens, ohne diese Stufe und ohne Tit-for-Tat das Modell 0,752 der Summe |
| H9 | Die Rangfolge hängt an Frist und Gewichtsunterschied | **Ja** | Rangkorrelation der Zellen mit (T = 20, s = 0,5): T = 6 0,71 / 0,71 / 0,83, T = 20 0,71 / 1,00 / 1,00, T = 60 0,94 / 0,89 / 0,94 (s = 0 / 0,5 / 1); der Sieger wechselt bei T = 20 zwischen s = 0 (Tit-for-Tat) und s = 0,5 (Boulware + Modell) |
| H10 | Wer zuletzt bietet, gibt nach; die Regel „letzte Runde“ prägt Einigung und Rang | **Ja** | ohne die Regel fällt Hartnäckig von 0,457 auf 0,255 (Einigungsrate 0,454 auf 0,069, Rang 5 auf 6), der eigene Nutzen der fünf übrigen ändert sich um höchstens 0,010 und ihre Rangfolge bleibt; Erstbieter-Vorteil im Selbstspiel von Boulware bei T = 6 +0,408 (Boulware + Modell +0,361), bei T = 20 −0,058 (+0,003), bei T = 60 −0,019 (−0,010) |
| H11 | Ein Q-Lerner schlägt die festen Strategien stabil | **Ja** im Mittel über den Pool, nicht gegen alle Gegner (gelb) | 0,615 / 0,657 / 0,640 nach 5.000 / 20.000 / 50.000 Episoden (Std 0,010 / 0,005 / 0,015) gegen Boulware + Modell 0,579; Übertragung auf sechs Bedingungen: 4 von 4 Trainings-Seeds vorn, bei s = 1 nur 3 von 4; stark gegen Hartnäckig (0,441 gegen 0,219), hinter Boulware + Modell gegen 2 von 6 Gegnern (Linear, Conceder); an Pool und Generator gebunden |
| H12 | Umnummerieren, Rollentausch, Determinismus | **Ja** | 288 Aushandlungen mit umnummerierten Punkten und Werten ohne Abweichung; gleichgerichtete Domäne: 216 von 216 Paaren einigen sich in Runde 1 auf (1; 1); Rollentausch im Selbstspiel höchstens 3,5 Standardfehler in 6 von 6 Strategien |
| H13 | Der Endspiel-Ausnutzer gewinnt, wo die Regel „letzte Runde“ gilt | **Ja**, zu einem Preis | Pool aus 7 Strategien (T = 20): 0,602 (Std 0,012) gegen 0,573 für Boulware + Modell, Rang 1 in 4 von 4 Seeds; die Einigungsrate fällt von 0,914 auf 0,850; Referenzwert mit bekanntem Gegnernutzen (Orakel + Endspiel) 0,657 im Pool aus 9 Strategien; mit der Regel „letzte Runde“ aus nicht gemessen |

**Domäne und Orakel** (E1; 400 Zufallsdomänen je Zeile, 4 × 4, g = 0,5, r = 0,2):

| Gewichtsunterschied s | Gegensatz g | Gewichtsabstand | Bündel auf der Grenze | Summe im Nash-Bündel | größte Summe |
|---|---|---|---|---|---|
| 0,00 | 0,50 | 0,000 | 27,0 | 1,213 | 1,228 |
| 0,25 | 0,50 | 0,109 | 21,1 | 1,262 | 1,276 |
| 0,50 | 0,50 | 0,218 | 17,6 | 1,329 | 1,339 |
| 0,75 | 0,50 | 0,322 | 16,3 | 1,417 | 1,423 |
| 1,00 | 0,50 | 0,427 | 16,0 | 1,512 | 1,519 |

**Sieger je Zelle** (E2; eigener Nutzen; in Klammern: in wie vielen der vier Seeds auch einzeln Rang 1):

| Frist T | s = 0,0 | s = 0,5 | s = 1,0 |
|---|---|---|---|
| 6 | Tit-for-Tat (4 von 4) | Tit-for-Tat (4 von 4) | Tit-for-Tat (4 von 4) |
| 20 | Tit-for-Tat (4 von 4) | Boulware + Modell (4 von 4) | Boulware + Modell (4 von 4) |
| 60 | Boulware (4 von 4) | Boulware (2 von 4) | Boulware + Modell (4 von 4) |

**Zelle T = 20, s = 0,5** (Hauptzelle):

| Strategie | Eigener Nutzen (Std) | Summe | Abstand zur Grenze | Einigungsrate | Nash-Produkt rel. | Rang je Seed |
|---|---|---|---|---|---|---|
| Boulware + Modell | 0,574 (0,007) | 1,087 | 0,119 | 0,908 | 0,477 | 1, 1, 1, 1 |
| Boulware | 0,569 (0,004) | 1,055 | 0,138 | 0,906 | 0,437 | 2, 2, 2, 2 |
| Tit-for-Tat | 0,558 (0,005) | 1,022 | 0,170 | 0,814 | 0,501 | 3, 3, 3, 3 |
| Linear | 0,484 (0,005) | 1,078 | 0,126 | 0,909 | 0,493 | 4, 4, 4, 4 |
| Hartnäckig | 0,457 (0,017) | 0,681 | 0,375 | 0,454 | 0,049 | 5, 5, 5, 5 |
| Conceder | 0,346 (0,001) | 1,055 | 0,130 | 0,909 | 0,370 | 6, 6, 6, 6 |

**Zelle T = 6, s = 0,5** (kurze Frist):

| Strategie | Eigener Nutzen (Std) | Summe | Abstand zur Grenze | Einigungsrate | Nash-Produkt rel. | Rang je Seed |
|---|---|---|---|---|---|---|
| Tit-for-Tat | 0,594 (0,005) | 0,909 | 0,227 | 0,747 | 0,244 | 1, 1, 1, 1 |
| Boulware | 0,503 (0,006) | 0,972 | 0,182 | 0,873 | 0,259 | 3, 2, 2, 2 |
| Boulware + Modell | 0,501 (0,003) | 0,999 | 0,164 | 0,883 | 0,274 | 2, 3, 3, 3 |
| Hartnäckig | 0,457 (0,017) | 0,681 | 0,375 | 0,454 | 0,049 | 4, 4, 4, 4 |
| Linear | 0,409 (0,004) | 0,995 | 0,174 | 0,902 | 0,336 | 5, 5, 5, 5 |
| Conceder | 0,306 (0,003) | 0,984 | 0,172 | 0,907 | 0,257 | 6, 6, 6, 6 |

**Zelle T = 60, s = 0,5** (lange Frist; Gleichstand an der Spitze):

| Strategie | Eigener Nutzen (Std) | Summe | Abstand zur Grenze | Einigungsrate | Nash-Produkt rel. | Rang je Seed |
|---|---|---|---|---|---|---|
| Boulware | 0,626 (0,005) | 1,109 | 0,105 | 0,909 | 0,522 | 2, 1, 1, 2 |
| Boulware + Modell | 0,626 (0,006) | 1,133 | 0,090 | 0,909 | 0,551 | 1, 2, 2, 1 |
| Linear | 0,518 (0,004) | 1,117 | 0,102 | 0,909 | 0,547 | 3, 3, 3, 3 |
| Tit-for-Tat | 0,499 (0,004) | 1,059 | 0,149 | 0,818 | 0,574 | 4, 4, 4, 4 |
| Hartnäckig | 0,457 (0,017) | 0,681 | 0,375 | 0,454 | 0,049 | 5, 5, 5, 5 |
| Conceder | 0,367 (0,002) | 1,087 | 0,111 | 0,909 | 0,424 | 6, 6, 6, 6 |

**Wer zuletzt bietet** (E3; Hartnäckig gegen Boulware, Tit-for-Tat und Hartnäckig; gegen Linear und Conceder ergeben sich in allen Zellen dieselben Nutzen und Einigungsraten wie gegen Boulware, nur die mittlere Einigungsrunde unterscheidet sich):

| Frist T | Hartnäckig bietet | Gegner | Nutzen Hartnäckig (Std) | Nutzen Gegner | Einigungsrate | Einigungsrunde |
|---|---|---|---|---|---|---|
| 20 | zuerst, nicht zuletzt | Boulware | 0,716 (0,019) | 0,231 | 1,000 | 19,9 |
| 20 | zuerst, nicht zuletzt | Tit-for-Tat | 0,205 (0,001) | 0,248 | 0,060 | 20,0 |
| 20 | zuerst, nicht zuletzt | Hartnäckig | 0,205 (0,001) | 0,248 | 0,060 | 20,0 |
| 20 | als Zweite, zuletzt | Boulware | 0,248 (0,007) | 0,205 | 0,060 | 19,9 |
| 20 | als Zweite, zuletzt | Tit-for-Tat | 0,248 (0,007) | 0,205 | 0,060 | 20,0 |
| 20 | als Zweite, zuletzt | Hartnäckig | 0,248 (0,007) | 0,205 | 0,060 | 20,0 |
| 21 | zuerst, zuletzt | Boulware | 0,268 (0,010) | 0,206 | 0,085 | 21,0 |
| 21 | zuerst, zuletzt | Tit-for-Tat | 0,268 (0,010) | 0,206 | 0,085 | 21,0 |
| 21 | zuerst, zuletzt | Hartnäckig | 0,268 (0,010) | 0,206 | 0,085 | 21,0 |
| 21 | als Zweite, nicht zuletzt | Boulware | 0,720 (0,010) | 0,229 | 0,998 | 21,0 |
| 21 | als Zweite, nicht zuletzt | Tit-for-Tat | 0,206 (0,002) | 0,268 | 0,085 | 21,0 |
| 21 | als Zweite, nicht zuletzt | Hartnäckig | 0,206 (0,002) | 0,268 | 0,085 | 21,0 |

Dasselbe ohne die Regel „letzte Runde“:

| Frist T | Hartnäckig bietet | Gegner | Nutzen Hartnäckig (Std) | Nutzen Gegner | Einigungsrate | Einigungsrunde |
|---|---|---|---|---|---|---|
| 20 | zuerst, nicht zuletzt | Boulware | 0,258 (0,010) | 0,206 | 0,073 | 19,0 |
| 20 | zuerst, nicht zuletzt | Tit-for-Tat | 0,200 (0,000) | 0,200 | 0,000 | – |
| 20 | zuerst, nicht zuletzt | Hartnäckig | 0,200 (0,000) | 0,200 | 0,000 | – |
| 20 | als Zweite, zuletzt | Boulware | 0,238 (0,008) | 0,205 | 0,048 | 19,9 |
| 20 | als Zweite, zuletzt | Tit-for-Tat | 0,200 (0,000) | 0,200 | 0,000 | – |
| 20 | als Zweite, zuletzt | Hartnäckig | 0,200 (0,000) | 0,200 | 0,000 | – |
| 21 | zuerst, zuletzt | Boulware | 0,258 (0,010) | 0,206 | 0,073 | 21,0 |
| 21 | zuerst, zuletzt | Tit-for-Tat | 0,200 (0,000) | 0,200 | 0,000 | – |
| 21 | zuerst, zuletzt | Hartnäckig | 0,200 (0,000) | 0,200 | 0,000 | – |
| 21 | als Zweite, nicht zuletzt | Boulware | 0,238 (0,008) | 0,205 | 0,048 | 20,0 |
| 21 | als Zweite, nicht zuletzt | Tit-for-Tat | 0,200 (0,000) | 0,200 | 0,000 | – |
| 21 | als Zweite, nicht zuletzt | Hartnäckig | 0,200 (0,000) | 0,200 | 0,000 | – |

**Die Regel „letzte Runde“ als Auswahl** (A: an, B: aus; Pool-Turnier T = 20, s = 0,5):

| Strategie | A: Regel an, eigener Nutzen | B: Regel aus, eigener Nutzen | Rang A / B | Einigungsrate A / B |
|---|---|---|---|---|
| Boulware + Modell | 0,574 | 0,572 | 1 / 1 | 0,908 / 0,818 |
| Boulware | 0,569 | 0,567 | 2 / 2 | 0,906 / 0,811 |
| Tit-for-Tat | 0,558 | 0,549 | 3 / 3 | 0,814 / 0,790 |
| Linear | 0,484 | 0,481 | 4 / 4 | 0,909 / 0,816 |
| Hartnäckig | 0,457 | 0,255 | 5 / 6 | 0,454 / 0,069 |
| Conceder | 0,346 | 0,344 | 6 / 5 | 0,909 / 0,814 |

**Endspiel-Ausnutzer** (E3; T = 20, s = 0,5, 4 × 50 Domänen), Pool aus 7 Strategien und Pool aus 9 (dazu zwei Orakel-Agenten, die den wahren Nutzen des Gegners kennen):

| Strategie | Eigener Nutzen (Std) | Summe | Einigungsrate | Rang je Seed |
|---|---|---|---|---|
| Endspiel-Ausnutzer | 0,602 (0,012) | 1,068 | 0,850 | 1, 1, 1, 1 |
| Boulware + Modell | 0,573 (0,006) | 1,098 | 0,914 | 2, 3, 2, 3 |
| Boulware | 0,571 (0,004) | 1,063 | 0,910 | 3, 2, 3, 2 |
| Tit-for-Tat | 0,554 (0,006) | 1,024 | 0,818 | 4, 4, 4, 4 |
| Linear | 0,477 (0,007) | 1,086 | 0,920 | 5, 5, 5, 5 |
| Hartnäckig | 0,409 (0,010) | 0,657 | 0,410 | 6, 6, 6, 6 |
| Conceder | 0,335 (0,006) | 1,044 | 0,919 | 7, 7, 7, 7 |

| Strategie | Eigener Nutzen (Std) | Summe | Einigungsrate | Rang je Seed |
|---|---|---|---|---|
| Boulware + Orakel + Endspiel | 0,657 (0,004) | 1,200 | 0,937 | 1, 1, 1, 1 |
| Boulware + Orakel | 0,610 (0,007) | 1,194 | 0,937 | 3, 2, 3, 2 |
| Endspiel-Ausnutzer | 0,609 (0,012) | 1,112 | 0,881 | 2, 3, 2, 3 |
| Boulware | 0,596 (0,005) | 1,110 | 0,933 | 4, 4, 5, 4 |
| Boulware + Modell | 0,591 (0,006) | 1,139 | 0,936 | 5, 5, 4, 5 |
| Tit-for-Tat | 0,575 (0,007) | 1,084 | 0,862 | 6, 6, 6, 6 |
| Linear | 0,481 (0,009) | 1,128 | 0,940 | 7, 7, 7, 7 |
| Hartnäckig | 0,410 (0,009) | 0,692 | 0,439 | 8, 8, 8, 8 |
| Conceder | 0,346 (0,007) | 1,090 | 0,939 | 9, 9, 9, 9 |

**Gegnermodell** (E4; s = 0,5, Beobachter spielt Linear und nimmt nie an, T = 40; nach k Angeboten des Gegners):

| Gegner | Gewichtsabstand k = 1 | Gewichtsabstand k = 8 | Gewichtsabstand k = 20 | Rangkorrelation k = 1 | Rangkorrelation k = 8 | Rangkorrelation k = 20 |
|---|---|---|---|---|---|---|
| Hartnäckig | 0,220 | 0,220 | 0,220 | +0,643 | +0,629 | +0,626 |
| Boulware | 0,220 | 0,219 | 0,197 | +0,643 | +0,635 | +0,741 |
| Linear | 0,220 | 0,172 | 0,172 | +0,643 | +0,778 | +0,589 |
| Conceder | 0,220 | 0,220 | 0,205 | +0,643 | +0,070 | −0,398 |
| Tit-for-Tat | 0,220 | 0,201 | 0,210 | +0,643 | +0,637 | +0,063 |

**Wirkung des Modells** (Programm gegen die fünf Basisgegner, T = 20, s = 0,5):

| Programm | Eigener Nutzen | Summe | Abstand zur Grenze | Einigungsrate |
|---|---|---|---|---|
| Boulware + kein Modell | 0,551 (0,005) | 1,034 (0,004) | 0,153 (0,003) | 0,905 |
| Boulware + Häufigkeitsmodell | 0,573 (0,006) | 1,085 (0,004) | 0,120 (0,002) | 0,909 |
| Boulware + Orakel | 0,608 (0,004) | 1,161 (0,007) | 0,075 (0,002) | 0,909 |
| Linear + kein Modell | 0,508 (0,003) | 1,067 (0,005) | 0,136 (0,006) | 0,909 |
| Linear + Häufigkeitsmodell | 0,516 (0,002) | 1,115 (0,007) | 0,105 (0,003) | 0,909 |
| Linear + Orakel | 0,539 (0,004) | 1,175 (0,009) | 0,068 (0,001) | 0,909 |
| Conceder + kein Modell | 0,395 (0,001) | 1,054 (0,005) | 0,139 (0,002) | 0,909 |
| Conceder + Häufigkeitsmodell | 0,389 (0,003) | 1,093 (0,008) | 0,109 (0,002) | 0,909 |
| Conceder + Orakel | 0,384 (0,002) | 1,137 (0,006) | 0,078 (0,001) | 0,909 |

**Zerlegung** (E5; 48 Agenten, Varianzanteil η² der Haupteffekte, Std über vier Seeds). Alle 48 Agenten:

| Kennzahl | Kurve | Akzeptanz | Modell | Rest |
|---|---|---|---|---|
| eigener Nutzen | 0,402 (0,018) | 0,461 (0,019) | 0,008 (0,002) | 0,128 (0,006) |
| Summe der Nutzen | 0,719 (0,025) | 0,032 (0,010) | 0,186 (0,021) | 0,064 (0,005) |
| Abstand zur Pareto-Grenze | 0,781 (0,024) | 0,005 (0,001) | 0,162 (0,017) | 0,052 (0,006) |
| Einigungsrate | 0,999 (0,002) | 0,000 (0,000) | 0,000 (0,000) | 0,001 (0,001) |

Ohne die Stufe „ab Reservation“:

| Kennzahl | Kurve | Akzeptanz | Modell | Rest |
|---|---|---|---|---|
| eigener Nutzen | 0,910 (0,008) | 0,037 (0,004) | 0,019 (0,005) | 0,034 (0,003) |
| Summe der Nutzen | 0,702 (0,027) | 0,009 (0,001) | 0,259 (0,025) | 0,031 (0,002) |
| Abstand zur Pareto-Grenze | 0,746 (0,023) | 0,006 (0,001) | 0,219 (0,020) | 0,029 (0,003) |
| Einigungsrate | 0,999 (0,001) | 0,000 (0,000) | 0,000 (0,000) | 0,000 (0,001) |

Ohne Tit-for-Tat:

| Kennzahl | Kurve | Akzeptanz | Modell | Rest |
|---|---|---|---|---|
| eigener Nutzen | 0,472 (0,015) | 0,373 (0,017) | 0,010 (0,003) | 0,145 (0,007) |
| Summe der Nutzen | 0,096 (0,032) | 0,089 (0,025) | 0,632 (0,058) | 0,182 (0,007) |
| Abstand zur Pareto-Grenze | 0,038 (0,014) | 0,023 (0,003) | 0,757 (0,021) | 0,182 (0,010) |
| Einigungsrate | 0,230 (0,066) | 0,035 (0,027) | 0,172 (0,060) | 0,563 (0,049) |

Ohne beides (27 Agenten):

| Kennzahl | Kurve | Akzeptanz | Modell | Rest |
|---|---|---|---|---|
| eigener Nutzen | 0,930 (0,010) | 0,022 (0,003) | 0,019 (0,005) | 0,030 (0,003) |
| Summe der Nutzen | 0,152 (0,051) | 0,023 (0,003) | 0,752 (0,047) | 0,073 (0,009) |
| Abstand zur Pareto-Grenze | 0,056 (0,022) | 0,022 (0,003) | 0,841 (0,023) | 0,081 (0,007) |
| Einigungsrate | 0,301 (0,102) | 0,005 (0,009) | 0,219 (0,061) | 0,475 (0,050) |

Annahmeregeln im Einzelnen:

| Annahmeregel (Kurve Linear, kein Modell) | Eigener Nutzen | Summe | Einigungsrate |
|---|---|---|---|
| ab Reservation | 0,348 (0,005) | 1,067 (0,008) | 0,924 |
| Schwelle 0,5 | 0,476 (0,008) | 1,076 (0,017) | 0,924 |
| Schwelle 0,7 | 0,465 (0,009) | 1,049 (0,019) | 0,924 |
| Schwelle 0,9 | 0,444 (0,004) | 1,033 (0,013) | 0,924 |
| Zeitplan linear | 0,502 (0,009) | 1,091 (0,019) | 0,924 |
| nächstes Gebot | 0,502 (0,008) | 1,091 (0,019) | 0,924 |
| Schwelle 0,9, ohne Regel „letzte Runde“ | 0,441 (0,004) | 0,983 (0,008) | 0,834 |
| nächstes Gebot, ohne Regel „letzte Runde“ | 0,500 (0,008) | 1,049 (0,015) | 0,843 |
| ab Reservation, ohne Regel „letzte Runde“ | 0,346 (0,005) | 1,026 (0,007) | 0,847 |

**Robustheit** (E6; Basis T = 20, s = 0,5, 4 Punkte × 4 Werte, r = 0,2, g = 0,5, Fenster 0,05; jede Zeile ändert einen Regler):

| Änderung (alles andere wie in der Basis) | Sieger | Seeds mit diesem Sieger | Eigener Nutzen des Siegers | Einigung des Siegers | Letzter |
|---|---|---|---|---|---|
| Basis | Boulware + Modell | 4 von 4 | 0,575 | 0,906 | Conceder |
| Boulware-Exponent 0,1 | Tit-for-Tat | 4 von 4 | 0,570 | 0,798 | Conceder |
| Boulware-Exponent 0,5 | Boulware + Modell | 2 von 4 | 0,581 | 0,907 | Conceder |
| Conceder-Exponent 2 | Boulware + Modell | 4 von 4 | 0,569 | 0,906 | Conceder |
| Conceder-Exponent 10 | Boulware + Modell | 2 von 4 | 0,575 | 0,906 | Conceder |
| Frist T = 21 | Boulware + Modell | 4 von 4 | 0,580 | 0,910 | Conceder |
| Frist T = 7 | Tit-for-Tat | 4 von 4 | 0,579 | 0,728 | Conceder |
| Frist T = 6 | Tit-for-Tat | 4 von 4 | 0,593 | 0,745 | Conceder |
| Fenster 0,02 | Boulware | 3 von 4 | 0,577 | 0,906 | Conceder |
| Fenster 0,10 | Boulware + Modell | 4 von 4 | 0,569 | 0,907 | Conceder |
| 3 Punkte × 4 Werte | Boulware + Modell | 4 von 4 | 0,572 | 0,904 | Conceder |
| 5 Punkte × 3 Werte | Boulware + Modell | 3 von 4 | 0,561 | 0,902 | Conceder |
| 4 Punkte × 3 Werte | Boulware + Modell | 3 von 4 | 0,561 | 0,903 | Conceder |
| 2 Punkte × 5 Werte | Tit-for-Tat | 2 von 4 | 0,569 | 0,797 | Conceder |
| Reservationsnutzen 0 | Hartnäckig | 4 von 4 | 0,900 | 1,000 | Conceder |
| Reservationsnutzen 0,5 | Boulware + Modell | 2 von 4 | 0,584 | 0,701 | Hartnäckig |
| Gegensatz g = 0 | Hartnäckig | 4 von 4 | 0,900 | 0,945 | Conceder |
| Gegensatz g = 1 | Boulware + Modell | 4 von 4 | 0,540 | 0,899 | Conceder |

Bei 18 Reglerwerten gewinnt Boulware + Modell in 11, Tit-for-Tat in 4, Hartnäckig in 2 und Boulware in 1.

**Lernender** (E7; Training gegen die fünf Basisgegner bei T = 20, s = 0,5, Test an neuen Domänen gegen alle sechs Pool-Strategien):

| Training | Mittel über 6 Gegner (Std) | Spanne der vier Trainings-Seeds | Einigungsrate |
|---|---|---|---|
| 5.000 Episoden | 0,615 (0,010) | 0,600 bis 0,623 | 0,911 |
| 20.000 Episoden | 0,657 (0,005) | 0,650 bis 0,661 | 0,895 |
| 50.000 Episoden | 0,640 (0,015) | 0,626 bis 0,662 | 0,897 |
| Boulware + Modell (fest) | 0,579 (0,002) | – | – |
| Boulware (fest) | 0,559 (0,004) | – | – |
| Linear (fest) | 0,494 (0,005) | – | – |
| Conceder (fest) | 0,379 (0,002) | – | – |

Je Gegner (nach 20.000 Episoden, Std über die Trainings-Seeds; feste Strategien auf denselben Test-Domänen):

| Gegner | Lernender (Std) | Boulware | Boulware + Modell | Linear | Conceder |
|---|---|---|---|---|---|
| Hartnäckig | 0,441 (0,008) | 0,219 | 0,219 | 0,219 | 0,219 |
| Boulware | 0,719 (0,013) | 0,535 | 0,559 | 0,407 | 0,330 |
| Linear | 0,700 (0,010) | 0,705 | 0,731 | 0,577 | 0,432 |
| Conceder | 0,720 (0,012) | 0,795 | 0,785 | 0,728 | 0,555 |
| Tit-for-Tat | 0,645 (0,019) | 0,474 | 0,567 | 0,583 | 0,432 |
| Boulware + Modell | 0,719 (0,006) | 0,624 | 0,616 | 0,452 | 0,305 |

Übertragung auf andere Bedingungen:

| Bedingung (Training bei T = 20, s = 0,5, Regel an) | Lernender (Std) | Boulware + Modell | Boulware | Linear | Trainings-Seeds vor beiden |
|---|---|---|---|---|---|
| T = 10 | 0,644 (0,014) | 0,537 | 0,522 | 0,460 | 4 von 4 |
| T = 21 | 0,652 (0,009) | 0,579 | 0,561 | 0,498 | 4 von 4 |
| T = 40 | 0,652 (0,008) | 0,609 | 0,595 | 0,520 | 4 von 4 |
| Regel „letzte Runde“ aus | 0,598 (0,008) | 0,578 | 0,557 | 0,492 | 4 von 4 |
| s = 0 | 0,629 (0,007) | 0,534 | 0,530 | 0,461 | 4 von 4 |
| s = 1 | 0,674 (0,017) | 0,653 | 0,627 | 0,562 | 3 von 4 |

Im Training ausgelassene Gegner (20.000 Episoden):

| Im Training ausgelassen | Lernender gegen ihn (Std) | Lernender gegen die vier übrigen | Boulware / Linear / Conceder gegen ihn |
|---|---|---|---|
| Conceder | 0,667 (0,020) | 0,607 | 0,795 / 0,728 / 0,555 |
| Boulware | 0,646 (0,025) | 0,617 | 0,535 / 0,407 / 0,330 |

**Gegenprobe mit anderen Zufalls-Seeds:** die Hauptzelle (T = 20, s = 0,5) mit den Seeds 11 bis 14 statt 1 bis 4 (4 × 100 Domänen, eigene Domänen aus anderer Bezeichnung) ergibt dieselbe Rangfolge der Spitze: Boulware + Modell vor Boulware vor Tit-for-Tat auf Rang 1, 2 und 3 in allen vier Seeds, Conceder Letzter (`test_claims.py`).

## Ehrliche Grenzen

Die Tabelle steht wortgleich in der App (Abschnitt „Wo die Annahmen enden“).

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Strategien sind Eigenbauten** (Formelfamilie der Zeitkurven nach Sekundärliteratur, Tit-for-Tat relativ, Häufigkeitsmodell, Endspiel, Lerner) | Es gibt keine Referenzimplementierung (GENIUS, NegMAS) und keinen Literaturwert zum Vergleich; andere Umsetzungen derselben Namen können andere Zahlen liefern. | – |
| **Nutzen sind linear-additiv**, Punkte unabhängig, beide Seiten haben denselben Reservationsnutzen | Hängen Punkte voneinander ab oder hat eine Seite eine bessere Außenoption, ändern sich Grenze, Ziele und Rangfolge. Nicht gerechnet. | – |
| **Ein Gegner pro Aushandlung, zwei Seiten, kein Diskont** | Mehrere Gegner, Koalitionen und Zeitkosten (jede Runde kostet etwas) sind nicht modelliert; ohne Diskont kostet Warten nichts, das dürfte Boulware stärken (nicht gemessen). | – |
| **Der Reservationsnutzen des Gegners ist bekannt** (nur das Endspiel nutzt ihn, über das Modell geschätzt) | Ist er privat, wird das Endspiel riskanter; private Information über Reservationsnutzen ist nicht modelliert. | – |
| **Die Rangfolge der Spitze hängt am Pool**, an Frist, Gewichtsunterschied, Fensterbreite und Regel „letzte Runde“ | Die Tabellen zeigen, wie weit: bei längerer Frist ist die Spitzengruppe eine Gruppe, bei T = 6 führt Tit-for-Tat allein. Ein Sieger in der einen Zelle verliert in der nächsten. | – |
| **Die Zerlegung in Gebot, Annahme und Gegnermodell hängt an der Kennzahl und an den gewählten Stufen** | Mit anderen Stufen ändern sich die Anteile (E5 zeigt vier Stufenwahlen). Eine feste Rangfolge der Bausteine gilt in diesem Modell nicht. | – |
| **Der Lernende ist an seinen Trainingspool und den Domänengenerator gebunden** | Gegen andere Gegner oder andere Domänen ist nichts gemessen; die Übertragung ist an sechs Bedingungen geprüft, nicht an echten Verhandlern. | – |
| **Sprachmodelle sind nur eine Einordnung** (kein Aufruf, keine Zahl übernommen) | Hybride Sprachmodell-Verhandler koppeln ein Modell mit einem Löser (ASTRA: Löser als Werkzeug, Gegnermodell, Tit-for-Tat-Gegenseitigkeit; laut Abstract ein Machbarkeitsnachweis); AgenticPay ist laut Abstract ein Benchmark für Käufer–Verkäufer-Verhandlungen in natürlicher Sprache. Beide sind nur nach den Abstracts beschrieben, nicht im Volltext gelesen. | – |
| **Literatur nur nach Verzeichnis oder Auszug** | Faratin, Sierra und Jennings (1998), Baarslag u. a. (2014, Zerlegung), ANAC (seit 2010, Baarslag u. a.), GENIUS, RLBOA (Bakker u. a. 2019), NegMAS (Mohammad u. a.), ASTRA (Kwon u. a., arXiv 2503.07129) und AgenticPay (Liu u. a., arXiv 2602.06008) sind nicht im Volltext gelesen; die Demo rechnet das hier beschriebene Modell und behauptet keine ihrer Aussagen. | – |

Weitere Grenzen: Die Aussagen über Hartnäckigkeit und das Endspiel gelten für die gewählte Regel „letzte Runde“ (an) und für Annahmeregel „nächstes Gebot“; die Raster-Zellen haben nur 4 Punkte × 4 Werte (E6 prüft vier andere Größen). Das Häufigkeitsmodell schätzt auf einer eigenen Skala; der Vergleich mit dem Reservationsnutzen des Gegners im Endspiel ist eine grobe Annäherung. Der Lerner sieht nur Zeit, Nutzen des letzten Angebots und Trend (90 Zustände). Das Mittel über einen Pool ist eine Setzung des Pools: mit anderen Gegnern ändert sich die Rangfolge der Spitze (H2). Aus den Presets ist nichts zu schließen: sie sind einzelne Domänen (Zufalls-Seeds 35 bzw. 0).

## Literatur (nicht im Volltext gelesen)

Alle Angaben nach Verzeichnis oder Auszug, keine Zahl daraus übernommen. Faratin, Sierra, Jennings (1998), *Robotics and Autonomous Systems* 24(3–4), 159–182 (Formelfamilie der zeitabhängigen Zugeständnisse über Sekundärtexte, z. B. arXiv 1604.04728). Baarslag, Hindriks, Hendrikx, Dirkzwager, Jonker, „Decoupling Negotiating Agents…“ (Springer 2014): Zerlegung in Gebot, Gegnermodell und Annahme (in der Literatur laut Verzeichnis „BOA“ für Bidding strategy, Opponent model, Acceptance strategy; nicht zu verwechseln mit BOA* = Bi-Objective A* in `multicriteria-demo`); **die Rangfolge der Bausteine ist nicht bestätigt** (H8: in diesem Modell nicht reproduziert). ANAC 2010 (Baarslag, Hindriks, Jonker, Kraus, Lin; sieben Teams, GENIUS). GENIUS (Lin u. a., Baarslag u. a.), NegMAS (Mohammad, Nakadai, Greenwald), RLBOA (Bakker, Hammond, Bloembergen, Baarslag, AAMAS 2019: tabellarisches Q-Learning auf komprimiertem Zustand). Einordnung ohne Aufruf eines Sprachmodells: ASTRA (Kwon u. a., arXiv 2503.07129; laut Abstract LP-Löser als Werkzeug, Gegnermodell und Tit-for-Tat-Gegenseitigkeit, Machbarkeitsnachweis mit drei Gegenständen) und AgenticPay (Liu, Gu, Song, arXiv 2602.06008; laut Abstract ein Benchmark für Käufer–Verkäufer-Verhandlungen in natürlicher Sprache).

## Rechenzeit

Einzelmessungen auf einem Rechner (Windows, Python 3.14, ein Prozess), nicht in Tests festgeschrieben: eine Aushandlung (4 Punkte × 4 Werte, T = 20) rund 1 ms; am Reglermaximum (5 × 5 Werte = 3.125 Bündel, T = 100) 3 ms für Boulware gegen Tit-for-Tat und 26 ms für zwei Strategien mit Häufigkeitsmodell (eine davon mit Endspiel); die Auswertung der Hauptansicht rechnet am Maximum 0,04 s, das Modell entlang der Aushandlung 0,6 s. Das Pool-Turnier live (nach Klick): 6 Strategien mit 10 Domänen je Paar 0,2 s, mit 50 Domänen 1,3 s (7 Strategien 2,0 s); am Reglermaximum (5 × 5 Werte, T = 100) 3 s mit 10 Domänen (6 Strategien) und 30 s mit 50 Domänen (7 Strategien). Der Lerner live (5.000 Episoden, nach Klick) rund 8 s, mit dem Test am Reglermaximum 18 s. Die Domänenstatistik (400 Domänen je Zeile) 2 s. `tools/make_grid.py` rechnet alle 480 Aufträge der Messreihen parallel; die Dauer hängt von Rechner und Prozesszahl ab, die Trainingsläufe mit 50.000 Episoden dominieren.

## Tests

**229 Tests**, Laufzeit in der letzten Messung 6 Minuten 47 Sekunden seriell auf Windows mit Python 3.14 bei ausgelastetem Rechner, abhängig von der Last (den größten Teil brauchen die AppTest-Dateien und die Gegenprobe mit anderen Seeds; die Linux-CI mit Python 3.12 ist noch nicht gelaufen). Zusammensetzung:

- **Orakel (4, `test_oracle_verhandlungsagenten.py`):** Pareto-Grenze und Nash-Lösung gegen die Schleife über alle Paare (100 Zufallsdomänen) und gegen Maximierer gewichteter Summen (300 Fälle), Grenze gegen Dominanz aller Bündel, der Spezialfall m = 2, g = 1 gegen die Reihenfolge-Regel von G (360 Ecken).
- **Einheiten an der von Hand gerechneten Mini-Domäne (16 + 18 + 14 + 12 + 7, `test_domain.py`, `test_strategies.py`, `test_negotiation.py`, `test_learning.py`, `test_results.py`):** Nutzen, Grenze, Nash, Zielkurven, Tit-for-Tat, Wahl auf der Isolinie und ihre Rückfälle, Annahmeregeln, Endspiel, drei Handläufe mit T = 3, Häufigkeitsmodell, Rangkorrelation (gegen scipy, wenn vorhanden), Q-Lerner (Zustandsindex, TD-Schritt, ε-Plan, Determinismus), Strategiewerte, Rangfolgen und η² an Handbeispielen.
- **Eigenschaften (13, `test_eigenschaften_verhandlungsagenten.py`)** und **Zweige (15, `test_branches.py`):** Umnummerieren, Determinismus, gleichgerichtete Domäne, Rollentausch, Kennzahlen konsistent, Reihenfolge im Pool; jede abschaltbare Einheit ändert auf mindestens einer Instanz das Ergebnis, dokumentierte Gleichheit (Tit-for-Tat gegen Hartnäckig), der tote Zweig „Endspiel ohne Modell“.
- **Claims (26, `test_claims.py`):** jede Zahl der README und der Hilfetexte; die README-Tabellen werden mit denselben Zeilenbauern der App erzeugt und im Test als Text in der README gesucht; Hypothesenzeilen H1 bis H13; die Gegenprobe der Hauptzelle mit den Zufalls-Seeds 11 bis 14.
- **Messdatei (18, `test_measurements.py`):** Erzeuger-Hash, alle 480 Aufträge vorhanden, zehn Aufträge neu gerechnet und mit dem gespeicherten Ergebnis verglichen (Marge 10⁻⁵).
- **Verdikte, Voreinstellungen, Randfälle (15 + 14 + 11):** Zahlen im Text gegen die Rechnung, jede Voreinstellung mit ihrer erwarteten Verdikt-Zeile und den Zusagen ihres Hilfetexts, kleinste und größte Domäne, T = 2, kein rationales Bündel, vollständiger Gleichstand.
- **Kopie des Zufallsgenerators (5), Laufzeit-Importe (1), Oberfläche (15 + 23 + 2):** Layout (fester Kopf, Schnellstart, Hervorhebung, jeder Regler hebt sie auf), AppTest (jeder Knopf einschließlich der ▶️-Knöpfe, jeder Regler an beiden Rändern, alle Minima und alle Maxima zugleich, Permalink, PDF, fehlende Messdatei), Footer.
- Die Messreihen stehen nirgends exakt in einem Test der Rechnung, sondern als gespeicherte Werte (README-Zahl gleich dem gespeicherten Wert) und als Reproduktion mit Marge.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht (eine Aushandlung) und sechs Reiter mit den Experimenten E1 bis E7 |
| `data/messreihen.json` | vorgerechnete Messreihen (480 Aufträge) mit Hash der Quellen |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenprobe der Rangkorrelation) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/make_grid.py` | rechnet die Messreihen neu |
| `vha_constants.py` | Regler-Grenzen, Messparameter (im Erzeuger-Hash), Farben, Voreinstellungen |
| `vha_domain.py` | Domäne, Pareto-Grenze und Nash-Lösung durch Aufzählen, Seeds aus Bezeichnungen |
| `vha_strategies.py` | Bausteine: Zielkurve, Tit-for-Tat, Wahl auf der Isolinie, Annahmeregeln, Endspiel, der Pool |
| `vha_opponent.py` | Häufigkeitsmodell, Rangkorrelation |
| `vha_negotiation.py` | Ablauf einer Aushandlung, Agenten, Kennzahlen |
| `vha_tournament.py` | Turnier: Paarmatrizen, Strategiewerte, Rangfolgen |
| `vha_learning.py` | Q-Lerner |
| `vha_experiments.py` | Aufträge der Messreihen, Speichern und Laden mit Hash |
| `vha_results.py` | Mittel, Std, Rangfolgen und Varianzanteile aus den Messreihen |
| `vha_evaluation.py` | Live-Auswertung, Zeilenbauer der Tabellen, Verdikte |
| `vha_live.py` | Live-Turnier und Live-Training |
| `vha_presets.py`, `vha_scenario.py`, `vha_layout.py` | Permalink, Voreinstellungen, Einstellungen, fester Kopf |
| `vha_texts.py`, `vha_visualization.py`, `vha_pdf_export.py` | Texte, Abbildungen, PDF |
| `vha_rng.py` | SplitMix64, wörtliche Kopie |
| `tests/` | Einheiten, Orakel, Eigenschaften, Zweige, Claims, Messdateien, Oberfläche, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

Echte Verhandler und Sprachmodelle (nur die Einordnung in der App, kein Aufruf); Mehrparteien-Verhandlung und Koalitionen; Diskont und Zeitkosten je Runde; private Information über den Reservationsnutzen; verschiedene Reservationsnutzen der beiden Seiten; nichtlineare oder voneinander abhängige Nutzen; Referenzimplementierungen (GENIUS, NegMAS) und Literaturwerte; eine gemeinsame Domäne mit Stück G (stetige Stellung statt diskreter Bündel); der Endspiel-Ausnutzer mit Regel „letzte Runde“ aus.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer; die in der Streamlit Community Cloud eingestellte Python-Version ist beim Deploy zu notieren. Die Messreihen neu rechnen: `python tools/make_grid.py`.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
