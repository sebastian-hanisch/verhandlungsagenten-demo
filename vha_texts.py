"""Feste Texte der App, die auch in der README stehen (ein Test vergleicht beide Fassungen, DEMO-PLAYBOOK §5). Zahlen aus Messungen stehen hier nicht, sie kommen aus der Rechnung (vha_evaluation)."""

INTRO = """
**Zwei Programme handeln mehrere Punkte einer Tour aus und kennen die Nutzen des anderen nicht:** ein Verlader und ein Spediteur verhandeln über Preis, Lieferfenster, Zahlungsziel, Haftung und Rückladung
(3 bis 5 Punkte mit je 3 bis 5 Werten, Nutzen auf 0 bis 1 normiert). Gefragt ist, **welche Strategie gegen unbekannte Gegner trägt, was ein Gegnermodell lernt und wann beide scheitern.**

Stück der Linie „Regeln setzen, verhandeln, kontrollieren“ und Gegenstück zu [`verhandlung-demo`](https://github.com/sebastian-hanisch/verhandlung-demo): dort wird das Gleichgewicht eines Ablaufs mit
**einem** Anteil und bekannten Nutzen gerechnet, hier **spielen Programme** gegeneinander, die den Nutzen des Gegners nicht kennen. [`kuchen-vergroessern-demo`](https://github.com/sebastian-hanisch/kuchen-vergroessern-demo)
zeigt, dass es bei mehreren Punkten Gewinn gibt; hier wird gemessen, welche Strategie ihn gegen unbekannte Gegner hebt.

Abgrenzung: `axelrod-turnier-demo` lässt 16 Strategien im wiederholten Gefangenendilemma gegeneinander antreten (ein Spiel, kein Verhandlungsablauf mit Frist); `tit-for-tat-demo` zeigt Tit-for-Tat als Automaten im
Dilemma. **Gleicher Name, anderer Mechanismus:** hier ist „Tit-for-Tat (relativ)“ ein Nachgeben im Verhandlungsverlauf (ich gebe so viel nach, wie der andere mir entgegenkommt). `noregret-demo` lernt Strategien
in einem anderen Spiel. **Strategien, Nutzen und Fristen sind Modellannahmen, keine Aussagen über echte Verhandler oder Sprachmodelle.**

Weiter unten: sechs Reiter mit sieben **🔬 Experimenten** (E1 bis E7; die Erklärtabellen der Strategien, Kennzahlen und Regler stehen oben), die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

EXPLAINER = r"""
1. **Domäne.** Jede Seite bewertet jeden Wert jedes Punktes (0 bis 1, je Punkt der schlechteste Wert 0, der beste 1) und gewichtet die Punkte (Summe 1); der Nutzen eines **Bündels** (ein Wert je Punkt) ist die gewichtete Summe.
   Der **Gewichtsunterschied** $s$ regelt, wie verschieden beide Seiten gewichten (je verschiedener, desto größer ist im Mittel die Summe der Nutzen im Nash-Bündel, E1), der **Gegensatz** $g$, wie entgegengesetzt die Wertlisten sind.
2. **Ablauf.** Die Seiten bieten abwechselnd ein Bündel; der Antwortende nimmt an oder lehnt ab. Nach $T$ Runden ohne Einigung bekommt jede Seite ihren **Reservationsnutzen**. Wer in Runde 1 bietet, bietet bei geradem $T$ nicht zuletzt.
3. **Gebot.** Jede Strategie setzt sich ein **Ziel** für den eigenen Nutzen und bietet ein Bündel auf der **Isolinie** (Nutzen im Fenster von 0,05 über dem Ziel). Boulware hält das Ziel lange hoch, Conceder gibt früh nach.
4. **Annahme.** Ein Angebot wird angenommen, wenn es mindestens so viel bringt wie das nächste eigene Gebot (Regel „nächstes Gebot“); in der letzten Runde nimmt der Antwortende alles ab dem Reservationsnutzen an (Regel „letzte Runde“, wählbar).
5. **Gegnermodell.** Aus den Angeboten des Gegners schätzt das Häufigkeitsmodell, welche Werte und welche Punkte ihm wichtig sind, und wählt unter gleich guten Bündeln das, das dem Gegner am liebsten sein dürfte.
6. **Maßstäbe.** Eigener Nutzen, Summe der Nutzen, Abstand zur Pareto-Grenze, Einigungsrate und Nash-Produkt; die Pareto-Grenze und die Nash-Lösung werden durch Aufzählen aller Bündel gefunden.
"""

STRATEGY_TABLE = """
| Strategie | Gebot (Ziel in Runde $t$ von $T$) | Gegnermodell | Annahme |
|---|---|---|---|
| **Hartnäckig** | immer das eigene Maximum (Ziel 1) | keins | nächstes Gebot (bringt das Angebot so viel wie mein Maximum?) |
| **Boulware** | Ziel $1-(1-r)\\,\\tau^{1/\\beta}$ mit $\\beta = 0{,}2$: bleibt lange hoch, fällt am Ende | keins | nächstes Gebot |
| **Linear** | $\\beta = 1$: Ziel sinkt gleichmäßig von 1 auf $r$ | keins | nächstes Gebot |
| **Conceder** | $\\beta = 5$: Ziel fällt früh | keins | nächstes Gebot |
| **Tit-for-Tat (relativ)** | Ziel sinkt um den Zuwachs, den das neue Angebot des Gegners gegenüber seinem vorigen für mich brachte, nie unter $r$ | keins | nächstes Gebot |
| **Boulware + Modell** | wie Boulware | Häufigkeitsmodell: unter den Bündeln im Fenster das mit dem größten geschätzten Gegnernutzen | nächstes Gebot |
| **Boulware + Modell + Endspiel** | wie Boulware, aber das letzte Gebot ist das für mich beste Bündel, das der Gegner laut Modell noch annimmt | Häufigkeitsmodell | nächstes Gebot |

$\\tau = (t-1)/(T-1)$ ist der Zeitfortschritt, $r$ der Reservationsnutzen. Die Formelfamilie der Zeitkurven stammt laut Sekundärliteratur von Faratin u. a. (1998); die Arbeit selbst ist nicht im Volltext gelesen.
„Tit-for-Tat (relativ)“ ist nicht das Tit-for-Tat des wiederholten Gefangenendilemmas (`tit-for-tat-demo`).
"""

METRICS_TABLE = """
| Kennzahl | Bedeutung | Bezugsmenge |
|---|---|---|
| **Eigener Nutzen** | Nutzen, den die Strategie am Ende bekommt (Reservationsnutzen ohne Einigung) | Mittel über alle anderen Strategien des Pools, beide Positionen und alle Domänen einer Zelle |
| **Summe** | Nutzen beider Seiten zusammen | dieselbe Menge |
| **Abstand zur Pareto-Grenze** | kleinster Abstand des Ergebnisses zu einem Punkt der Grenze der individuell rationalen Bündel in der Nutzenebene | dieselbe Menge |
| **Einigungsrate** | Anteil der Aushandlungen mit Einigung | dieselbe Menge |
| **Nash-Produkt relativ** | Produkt der Gewinne über dem Reservationsnutzen, geteilt durch das der Nash-Lösung (0 ohne Einigung) | dieselbe Menge |
| **Std** | Standardabweichung (mit n − 1) der Zellenwerte über vier unabhängige Zufalls-Seeds (je 100 oder 50 Domänen) | vier Seeds |
"""

CONTROLS_TABLE = """
| Regler | Was er tut | Wirkung auf die Frage |
|---|---|---|
| **Punkte n, Werte m** | Wie viele Punkte verhandelt werden und wie viele Werte jeder hat; es gibt $m^n$ Bündel. | Bestimmt die Auswahl auf der Isolinie; die Messreihen laufen mit 4 × 4, E6 prüft vier andere Größen. |
| **Gewichtsunterschied s** | 0: beide Seiten gewichten die Punkte gleich; 1: unabhängig. | Je verschiedener die Gewichte, desto größer die Summe der Nutzen im Nash-Bündel (E1: 1,213 bei s = 0, 1,512 bei s = 1). |
| **Gegensatz g** | 0: die Wertlisten sind unabhängig; 1: genau gegenläufig (was der eine mag, mag der andere nicht). | E6 misst g = 0 und g = 1 gegen den Standard. |
| **Reservationsnutzen r** | Was jede Seite ohne Einigung bekommt. | E6 misst r = 0 und r = 0,5 gegen den Standard 0,2. |
| **Frist T** | Zahl der Runden; der Verlader bietet in den ungeraden, der Spediteur in den geraden. | Kurze und lange Frist belohnen verschiedene Strategien; die Parität entscheidet, wer zuletzt bietet (E2, E3). |
| **Fensterbreite** | Wie weit über dem Ziel das gebotene Bündel liegen darf. | Ein breiteres Fenster lässt dem Gegnermodell mehr Bündel zur Auswahl; E6 misst 0,02 und 0,10. |
| **Regel „letzte Runde“** | An oder aus: nimmt der Antwortende in Runde T alles ab dem Reservationsnutzen an? | Prägt Einigungsrate und Rang der Hartnäckigen (E3). |
| **Strategie des Verladers, des Spediteurs** | Welches Programm die Seite spielt. | Siehe die Erklärtabelle der Strategien. |
| **Zufalls-Seed der Domäne** | Welche Zufallsdomäne gezogen wird. | Eine Domäne ist eine Stichprobe; die Tabellen mitteln über viele. |
"""

ASSUMPTIONS_MARKDOWN = """
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
"""

MATH_MARKDOWN = r"""
**Lesehilfe.** Gegeben sind $n$ Punkte mit je $m$ Werten, Gewichte $w_{p,i}$ und Wertlisten $v_{p,i}(j) \in [0,1]$ der beiden Seiten $p \in \{0,1\}$, ein Reservationsnutzen $r$ und eine Frist $T$. Gesucht sind
die Ergebnisse, die Strategiepaare erzielen, und wie sie sich zur Pareto-Grenze verhalten.

| Symbol | Bedeutung |
|---|---|
| $b = (b_1, \dots, b_n)$ | Bündel: ein Wert $b_i \in \{0, \dots, m-1\}$ je Punkt; es gibt $m^n$ Bündel |
| $u_p(b)$ | Nutzen der Seite $p$ |
| $r$ | Reservationsnutzen (Ergebnis ohne Einigung) |
| $T$, $t$, $\tau$ | Frist, Runde, Zeitfortschritt $\tau = (t-1)/(T-1)$ |
| $\beta$ | Exponent der Zeitkurve |
| $\hat w$, $\hat v$, $\hat u$ | Schätzung des Gegners: Gewichte, Wertlisten, Nutzen |

**Nutzen.** $u_p(b) = \sum_{i=1}^{n} w_{p,i}\, v_{p,i}(b_i)$ mit $\sum_i w_{p,i} = 1$; jede Wertliste hat Minimum 0 und Maximum 1, also ist $\max_b u_p(b) = 1$.

**Pareto-Grenze und Nash.** Ein Bündel $b$ ist dominiert, wenn ein $b'$ beiden Seiten mindestens gleich viel und einer Seite mehr bringt. Unter den individuell rationalen Bündeln ($u_p \ge r$) maximiert die Nash-Lösung
$$b^{\mathrm{N}} = \arg\max_b\; (u_0(b) - r)\,(u_1(b) - r).$$

**Zielkurve.**
$$U(t) = 1 - (1 - r)\,\tau^{1/\beta}, \qquad \tau = \frac{t-1}{T-1}.$$
$\beta < 1$ (Boulware) hält das Ziel lange hoch, $\beta = 1$ ist linear, $\beta > 1$ (Conceder) gibt früh nach; Hartnäckig setzt $U \equiv 1$.

**Tit-for-Tat (relativ).** Mit $\Delta_t = u(o_t) - u(o_{t-1})$, dem Zuwachs des neuen Gegnerangebots $o_t$ für mich: $U_t = \max\{\,r,\; U_{t-1} - \max(\Delta_t, 0)\,\}$.

**Wahl auf der Isolinie.** Unter den Bündeln mit $U \le u(b) \le U + 0{,}05$ wird das mit dem größten geschätzten Gegnernutzen $\hat u(b)$ gewählt (ohne Modell: eine feste Zufallsreihenfolge je Domäne).

**Häufigkeitsmodell.** Mit $c_{i,j}$ der Zahl der Angebote des Gegners mit Wert $j$ an Punkt $i$ und $a_i$ der Zahl der Angebote, bei denen Punkt $i$ gegenüber dem vorigen gleich blieb:
$$\hat v_i(j) = \frac{c_{i,j} + 1}{\max_k c_{i,k} + 1}, \qquad \hat w_i = \frac{1 + a_i}{\sum_k (1 + a_k)}, \qquad \hat u(b) = \sum_i \hat w_i\, \hat v_i(b_i).$$

**Maßstäbe.** Abstand zur Pareto-Grenze: kleinster euklidischer Abstand des Nutzenpaars zur Grenze der rationalen Bündel. Nash-Produkt relativ: $(u_0 - r)(u_1 - r)$ geteilt durch das der Nash-Lösung.
**Varianzanteil** (E5): $\eta^2_F = \sum_\ell n_\ell\,(\bar x_\ell - \bar x)^2 \big/ \sum_a (x_a - \bar x)^2$ für einen Faktor $F$ mit den Stufen $\ell$ über die Agenten $a$.
"""
