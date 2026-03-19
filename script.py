import sys
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QLabel, QSpinBox)
from numba import njit


# --- 1. Allgemeine Mathematische Kernfunktionen (Numba) ---

@njit
def fact(n):
    """Berechnet die Fakultät für die Polynome."""
    if n <= 1: return 1.0
    res = 1.0
    for i in range(2, int(n) + 1):
        res *= float(i)
    return res


@njit
def assoc_laguerre(p, k, x):
    """Zugeordnete Laguerre-Polynome für den radialen Teil."""
    res = 0.0
    for i in range(p + 1):
        num = fact(p + k)
        den = fact(p - i) * fact(k + i)
        term = ((-1) ** i) * (num / den) * (x ** i) / fact(i)
        res += term
    return res


@njit
def assoc_legendre(l, m, x):
    """Zugeordnete Legendre-Polynome für den winkelabhängigen Teil."""
    m = abs(m)
    # Begrenzung zur Vermeidung von Floating-Point Ungenauigkeiten
    if x > 1.0: x = 1.0
    if x < -1.0: x = -1.0

    res = 0.0
    for k in range(int((l - m) / 2) + 1):
        num = fact(2 * l - 2 * k)
        den = (2 ** l) * fact(k) * fact(l - k) * fact(l - 2 * k - m)
        term = ((-1) ** k) * (num / den) * (x ** (l - 2 * k - m))
        res += term
    return ((-1) ** m) * (np.sqrt(1.0 - x ** 2) ** m) * res


@njit
def atomic_pdf(x, y, z, n, l, m):
    """Die universelle wasserstoffähnliche Wellenfunktion quadriert (PDF)."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if r < 1e-10:
        r = 1e-10  # Verhindert Division durch Null

    # Radialer Teil R(r)
    rho = 2.0 * r / n
    rad_part = np.exp(-r / n) * (rho ** l) * assoc_laguerre(n - l - 1, 2 * l + 1, rho)

    # Winkelabhängiger Teil Y(theta, phi)
    cos_theta = z / r
    phi = np.arctan2(y, x)

    p_lm = assoc_legendre(l, m, cos_theta)

    # Reelle sphärische Harmonische (für die Visualisierung in der Chemie üblich)
    if m == 0:
        ang_part = p_lm
    elif m > 0:
        ang_part = p_lm * np.cos(m * phi)
    else:
        ang_part = p_lm * np.sin(abs(m) * phi)

    return (rad_part * ang_part) ** 2


# --- Molekülfunktionen (Bleiben bestehen) ---
@njit
def pdf_h2_bonding(x, y, z):
    r1 = np.sqrt(x ** 2 + y ** 2 + (z - 0.7) ** 2)
    r2 = np.sqrt(x ** 2 + y ** 2 + (z + 0.7) ** 2)
    return (np.exp(-r1) + np.exp(-r2)) ** 2


@njit
def pdf_h2_antibonding(x, y, z):
    r1 = np.sqrt(x ** 2 + y ** 2 + (z - 0.7) ** 2)
    r2 = np.sqrt(x ** 2 + y ** 2 + (z + 0.7) ** 2)
    return (np.exp(-r1) - np.exp(-r2)) ** 2


# --- Weichensteller für Numba ---
@njit
def evaluate_pdf(x, y, z, orb_type, n, l, m):
    """Entscheidet innerhalb der kompilierten Schleife, was berechnet wird."""
    if orb_type == 0:
        return atomic_pdf(x, y, z, n, l, m)
    elif orb_type == 1:
        return pdf_h2_bonding(x, y, z)
    else:
        return pdf_h2_antibonding(x, y, z)


# --- 2. Universeller Metropolis-Hastings ---
@njit
def metropolis_hastings(orb_type, n, l, m, num_points=30000, thin=5, step_size=1.0):
    total_steps = num_points * thin
    points = np.zeros((num_points, 3))

    x, y, z = 1.0, 1.0, 1.0
    current_prob = evaluate_pdf(x, y, z, orb_type, n, l, m)

    accepted_points = 0

    for i in range(total_steps):
        step_x = np.random.normal(0.0, step_size)
        step_y = np.random.normal(0.0, step_size)
        step_z = np.random.normal(0.0, step_size)

        prop_x = x + step_x
        prop_y = y + step_y
        prop_z = z + step_z

        proposal_prob = evaluate_pdf(prop_x, prop_y, prop_z, orb_type, n, l, m)

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
        self.setWindowTitle('Universeller Orbital-Generator (n, l, m)')
        self.setGeometry(100, 100, 1100, 800)
        self.setStyleSheet("""
                    QWidget { 
                        background-color: #0b0c10; 
                        color: #c5c6c7; 
                        font-family: 'Segoe UI'; 
                        font-size: 14px; 
                    }
                    QComboBox, QSpinBox { 
                        background-color: #1f2833; 
                        border: 1px solid #45a29e; 
                        padding: 4px; 
                        border-radius: 4px; 
                        color: #ffffff; 
                    }
                    QLabel { 
                        font-weight: bold; 
                        margin-left: 10px; 
                    }

                    /* --- Fix für die QSpinBox Pfeile (Schaltflächen) --- */
                    QSpinBox::up-button {
                        subcontrol-origin: border;
                        subcontrol-position: top right;
                        width: 20px;
                        border-left: 1px solid #45a29e;
                        background-color: #1f2833;
                        border-top-right-radius: 3px;
                    }
                    QSpinBox::down-button {
                        subcontrol-origin: border;
                        subcontrol-position: bottom right;
                        width: 20px;
                        border-left: 1px solid #45a29e;
                        background-color: #1f2833;
                        border-bottom-right-radius: 3px;
                    }

                    /* Hover-Effekt für die Buttons */
                    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                        background-color: #45a29e;
                    }

                    /* CSS-Dreiecke als Pfeilersatz */
                    QSpinBox::up-arrow {
                        width: 0px; height: 0px;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-bottom: 5px solid #c5c6c7;
                    }
                    QSpinBox::down-arrow {
                        width: 0px; height: 0px;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-top: 5px solid #c5c6c7;
                    }

                    /* Pfeilfarbe ändert sich beim Hovern (Kontrast) */
                    QSpinBox::up-arrow:hover { border-bottom: 5px solid #0b0c10; }
                    QSpinBox::down-arrow:hover { border-top: 5px solid #0b0c10; }
                """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #111217; border-bottom: 1px solid #1f2833;")
        toolbar_layout = QHBoxLayout(toolbar)

        # Modus Auswahl
        toolbar_layout.addWidget(QLabel("Modus:"))
        self.combo = QComboBox()
        self.combo.addItems([
            "Atomares Orbital (via Quantenzahlen)",
            "H2 Molekül (bindend)",
            "H2 Molekül (antibindend)"
        ])
        self.combo.currentIndexChanged.connect(self.toggle_mode)
        toolbar_layout.addWidget(self.combo)

        # Quantenzahlen
        self.label_n = QLabel("n:")
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 15)
        self.spin_n.setValue(3)

        self.label_l = QLabel("l:")
        self.spin_l = QSpinBox()
        self.spin_l.setRange(0, 2)
        self.spin_l.setValue(2)

        self.label_m = QLabel("m:")
        self.spin_m = QSpinBox()
        self.spin_m.setRange(-2, 2)
        self.spin_m.setValue(0)

        for widget in [self.label_n, self.spin_n, self.label_l, self.spin_l, self.label_m, self.spin_m]:
            toolbar_layout.addWidget(widget)

        # Update-Logik für Quantenzahlen
        self.spin_n.valueChanged.connect(self.update_limits)
        self.spin_l.valueChanged.connect(self.update_limits)

        # Punkte Setup
        toolbar_layout.addWidget(QLabel("Punkte:"))
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(1000, 1000000)
        self.points_spinbox.setSingleStep(10000)
        self.points_spinbox.setValue(40000)
        self.points_spinbox.setKeyboardTracking(False)
        toolbar_layout.addWidget(self.points_spinbox)

        self.status_label = QLabel("Status: Bereit")
        toolbar_layout.addWidget(self.status_label)
        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        # Re-Render Auslöser
        self.spin_n.valueChanged.connect(self.update_orbital)
        self.spin_l.valueChanged.connect(self.update_orbital)
        self.spin_m.valueChanged.connect(self.update_orbital)
        self.points_spinbox.valueChanged.connect(self.update_orbital)

        # 3D View
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=30, elevation=20, azimuth=45)
        main_layout.addWidget(self.view, stretch=1)
        self.scatter = gl.GLScatterPlotItem(size=2, pxMode=True)
        self.scatter.setGLOptions('additive')
        self.view.addItem(self.scatter)

        self.update_orbital()

    def update_limits(self):
        """Passt die maximal/minimal erlaubten Werte für l und m basierend auf n an."""
        n = self.spin_n.value()
        self.spin_l.setMaximum(n - 1)

        l = self.spin_l.value()
        self.spin_m.setMinimum(-l)
        self.spin_m.setMaximum(l)

    def toggle_mode(self):
        is_atomic = (self.combo.currentIndex() == 0)
        for w in [self.label_n, self.spin_n, self.label_l, self.spin_l, self.label_m, self.spin_m]:
            w.setVisible(is_atomic)
        self.update_orbital()

    def update_orbital(self):
        self.status_label.setText("Status: Berechne...")
        self.status_label.setStyleSheet("color: #ffcc00;")
        QApplication.processEvents()

        orb_type = self.combo.currentIndex()
        n = self.spin_n.value()
        l = self.spin_l.value()
        m = self.spin_m.value()
        num_points = self.points_spinbox.value()

        # Automatische Anpassung der Schrittweite: Größere Orbitale (höheres n) brauchen größere Schritte
        step = n * 0.75 if orb_type == 0 else 1.5

        # Farbgebung dynamisch nach Orbitalform (l)
        if orb_type == 0:
            if l == 0:
                color_base = [0.1, 0.5, 1.0]  # s: Blau
            elif l == 1:
                color_base = [0.8, 0.2, 0.5]  # p: Pink/Rot
            elif l == 2:
                color_base = [0.2, 0.9, 0.4]  # d: Grün
            elif l == 3:
                color_base = [1.0, 0.6, 0.0]  # f: Orange
            else:
                color_base = [0.8, 0.8, 0.8]  # g, h...: Weiß/Grau
        elif orb_type == 1:
            color_base = [0.2, 0.9, 0.9]  # H2 bindend
        else:
            color_base = [0.9, 0.2, 0.2]  # H2 antibindend

        positions = metropolis_hastings(orb_type, n, l, m, num_points=num_points, thin=5, step_size=step)

        colors = np.ones((len(positions), 4))
        colors[:, :3] = color_base
        colors[:, 3] = 0.06  # Alpha-Transparenz

        self.scatter.setData(pos=positions, color=colors)

        # Kamera etwas herauszoomen, wenn das Orbital sehr groß ist
        if orb_type == 0:
            self.view.setCameraPosition(distance=max(25, n * 8))

        self.status_label.setText(f"Status: Fertig ({num_points:,} Punkte)")
        self.status_label.setStyleSheet("color: #45a29e;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OrbitalApp()
    window.show()
    sys.exit(app.exec())