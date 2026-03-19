Ein leistungsstarker, interaktiver 3D-Visualisierer für atomare Orbitale und Molekülorbitale (H₂), basierend auf dem Metropolis-Hastings-Algorithmus. 
Das Tool berechnet die Aufenthaltswahrscheinlichkeiten von Elektronen in Echtzeit und stellt diese als Punktwolken dar.

🚀 FeaturesInteraktive Atomorbitale: 
Visualisierung wasserstoffähnlicher Wellenfunktionen durch Eingabe der Quantenzahlen $n$, $l$ und $m$.
Molekülorbitale: Darstellung von bindenden und antibindenden Zuständen des Wasserstoff-Moleküls ($H_2$).
High Performance: Nutzung von Numba (JIT-Kompilierung) für blitzschnelle mathematische Berechnungen in Maschinencode.
Stochastische Simulation: Implementierung des Metropolis-Hastings-Algorithmus (Markov-Chain-Monte-Carlo), um die Wahrscheinlichkeitsdichte ($|\psi|^2$) effizient abzutasten.
Moderne GUI: Dunkles, benutzerfreundliches Interface mit PyQt6 und GPU-beschleunigtem Rendering via pyqtgraph.opengl.
