import sys
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QLabel, QSpinBox)
from numba import njit


# --- 1. Wahrscheinlichkeitsdichten (PDFs) mit Numba ---
# Durch @njit werden diese Funktionen in extrem schnellen Maschinencode übersetzt.

@njit
def pdf_1s(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.exp(-2 * r)


@njit
def pdf_2s(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (2 - r) ** 2 * np.exp(-r)


@njit
def pdf_2pz(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (z ** 2) * np.exp(-r)


@njit
def pdf_2px(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (x ** 2) * np.exp(-r)


@njit
def pdf_2py(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (y ** 2) * np.exp(-r)


@njit
def pdf_3dz2(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return ((3 * z ** 2 - r ** 2) ** 2) * np.exp(-2 * r / 3)


@njit
def pdf_3dx2y2(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return ((x ** 2 - y ** 2) ** 2) * np.exp(-2 * r / 3)


@njit
def pdf_h2_bonding(x, y, z):
    r1 = np.sqrt(x ** 2 + y ** 2 + (z - 0.7) ** 2)
    r2 = np.sqrt(x ** 2 + y ** 2 + (z + 0.7) ** 2)
    psi1 = np.exp(-r1)
    psi2 = np.exp(-r2)
    return (psi1 + psi2) ** 2


@njit
def pdf_h2_antibonding(x, y, z):
    r1 = np.sqrt(x ** 2 + y ** 2 + (z - 0.7) ** 2)
    r2 = np.sqrt(x ** 2 + y ** 2 + (z + 0.7) ** 2)
    psi1 = np.exp(-r1)
    psi2 = np.exp(-r2)
    return (psi1 - psi2) ** 2


# --- 2. Metropolis-Hastings Algorithmus mit Numba ---
@njit
def metropolis_hastings(pdf, num_points=30000, thin=5, step_size=1.0):
    total_steps = num_points * thin
    points = np.zeros((num_points, 3))

    # Skalare Variablen sind in Numba oft schneller als Arrays in Schleifen
    x, y, z = 1.0, 1.0, 1.0
    current_prob = pdf(x, y, z)

    accepted_points = 0

    for i in range(total_steps):
        # Einzelne Zufallszahlen generieren ist in Numba blitzschnell
        step_x = np.random.normal(0.0, step_size)
        step_y = np.random.normal(0.0, step_size)
        step_z = np.random.normal(0.0, step_size)

        prop_x = x + step_x
        prop_y = y + step_y
        prop_z = z + step_z

        proposal_prob = pdf(prop_x, prop_y, prop_z)

        if current_prob == 0.0:
            acceptance = 1.0
        else:
            acceptance = proposal_prob / current_prob

        if acceptance >= 1.0 or np.random.rand() < acceptance:
            x, y, z = prop_x, prop_y, prop_z
            current_prob = proposal_prob

        if i % thin == 0:
            points[accepted_points, 0] = x
            points[accepted_points, 1] = y
            points[accepted_points, 2] = z
            accepted_points += 1

    return points


# --- 3. Die Hauptanwendung ---
class OrbitalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Interaktive Wasserstoff-Orbitale & H2-Molekül (Numba Optimized)')
        self.setGeometry(100, 100, 1000, 800)

        self.setStyleSheet("""
                    QWidget { background-color: #0b0c10; color: #c5c6c7; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
                    QComboBox { background-color: #1f2833; border: 1px solid #45a29e; padding: 6px 12px; border-radius: 4px; color: #ffffff; }
                    QComboBox::drop-down { border: 0px; }
                    QComboBox:hover { border: 1px solid #66fcf1; }
                    QSpinBox { background-color: #1f2833; border: 1px solid #45a29e; padding: 6px 25px 6px 12px; border-radius: 4px; color: #ffffff; }
                    QSpinBox:hover { border: 1px solid #66fcf1; }
                    QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 20px; border-left: 1px solid #45a29e; background-color: #2b3746; border-top-right-radius: 3px; }
                    QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border-left: 1px solid #45a29e; background-color: #2b3746; border-bottom-right-radius: 3px; }
                    QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #45a29e; }
                """)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #111217; border-bottom: 1px solid #1f2833;")
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(20, 12, 20, 12)
        toolbar.setLayout(toolbar_layout)

        label_orbital = QLabel("Orbital:")
        label_orbital.setStyleSheet("font-weight: bold; border: none;")
        toolbar_layout.addWidget(label_orbital)

        self.combo = QComboBox()
        self.combo.addItems([
            "1s Orbital (Kugel)",
            "2s Orbital (Knotenfläche)",
            "2p_x Orbital (Hantel X-Achse)",
            "2p_y Orbital (Hantel Y-Achse)",
            "2p_z Orbital (Hantel Z-Achse)",
            "3d_z² Orbital (Hantel mit Donut)",
            "3d_x²-y² Orbital (Kleeblatt)",
            "H2 Molekül (bindendes σ-Orbital)",
            "H2 Molekül (antibindendes σ*-Orbital)"
        ])
        self.combo.currentIndexChanged.connect(self.update_orbital)
        toolbar_layout.addWidget(self.combo)

        label_points = QLabel("Punkte:")
        label_points.setStyleSheet("font-weight: bold; border: none; margin-left: 20px;")
        toolbar_layout.addWidget(label_points)

        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(1000, 50000000)
        self.points_spinbox.setSingleStep(10000)
        self.points_spinbox.setValue(40000)
        self.points_spinbox.setKeyboardTracking(False)
        self.points_spinbox.valueChanged.connect(self.update_orbital)
        toolbar_layout.addWidget(self.points_spinbox)

        self.status_label = QLabel("Status: Bereit")
        self.status_label.setStyleSheet("color: #45a29e; font-style: italic; border: none; margin-left: 20px;")
        toolbar_layout.addWidget(self.status_label)

        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=25, elevation=20, azimuth=45)
        main_layout.addWidget(self.view, stretch=1)

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        grid.setColor((80, 80, 80, 100))
        self.view.addItem(grid)

        self.scatter = gl.GLScatterPlotItem(size=2, pxMode=True)
        self.scatter.setGLOptions('additive')
        self.view.addItem(self.scatter)

        self.update_orbital()

    def update_orbital(self):
        self.status_label.setText("Status: Berechne... (Bitte warten)")
        self.status_label.setStyleSheet("color: #ffcc00; font-style: italic; border: none; margin-left: 20px;")
        QApplication.processEvents()

        selection = self.combo.currentText()
        num_points = self.points_spinbox.value()

        if "1s" in selection:
            pdf, color_base = pdf_1s, [0.1, 0.5, 1.0]
        elif "2s" in selection:
            pdf, color_base = pdf_2s, [1.0, 0.3, 0.1]
        elif "2p_x" in selection:
            pdf, color_base = pdf_2px, [0.8, 0.2, 0.5]
        elif "2p_y" in selection:
            pdf, color_base = pdf_2py, [0.8, 0.8, 0.2]
        elif "2p_z" in selection:
            pdf, color_base = pdf_2pz, [0.2, 0.9, 0.4]
        elif "3d_z²" in selection:
            pdf, color_base = pdf_3dz2, [0.6, 0.2, 1.0]
        elif "3d_x²-y²" in selection:
            pdf, color_base = pdf_3dx2y2, [0.2, 0.8, 0.8]
        elif "bindendes" in selection:
            pdf, color_base = pdf_h2_bonding, [0.2, 0.9, 0.9]
        elif "antibindendes" in selection:
            pdf, color_base = pdf_h2_antibonding, [0.9, 0.2, 0.2]

        step = 1.5 if ("3d" in selection or "H2" in selection) else 1.0

        # Der Metropolis-Aufruf. Numba kompiliert die Funktion beim jeweils ersten
        # Aufruf mit einer neuen PDF. Das dauert initial ca. 0.5 Sekunden,
        # danach läuft es in Echtzeit.
        positions = metropolis_hastings(pdf, num_points=num_points, thin=5, step_size=step)

        colors = np.ones((len(positions), 4))
        colors[:, 0] = color_base[0]
        colors[:, 1] = color_base[1]
        colors[:, 2] = color_base[2]
        colors[:, 3] = 0.06

        self.scatter.setData(pos=positions, color=colors)

        self.status_label.setText(f"Status: Fertig ({num_points:,} Punkte)".replace(',', '.'))
        self.status_label.setStyleSheet("color: #45a29e; font-style: italic; border: none; margin-left: 20px;")


def main():
    app = QApplication(sys.argv)
    window = OrbitalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()