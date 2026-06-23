"""
=============================================================
  Drei gekoppelte Pendel – Chaos und sensitive Abhängigkeit
=============================================================

Forschungsfrage:
  Wie verändert sich das Schwingungsverhalten von drei federgekoppelten
  Pendeln, wenn man Federstärke (k) oder Startwinkel minimal variiert?
  Zeigt das System sensitives Chaos?

Modell / Annahmen:
  - Drei masselose Stäbe mit Punktmassen am Ende (ideale Pendel)
  - Kleine Reibung (Dämpfung) vernachlässigt (konservatives System)
  - Federn verbinden die Pendelspitzen (Hooke'sches Gesetz: F = k * Δx)
  - Gravitation: g = 9.81 m/s²
  - Numerische Integration: Euler-Methode mit dt = 0.005 s

Vereinfachungen:
  - Keine Luftreibung
  - Federn haben keine Masse
  - Pivotpunkte sind fixiert (keine Bewegung der Aufhängung)
  - 2D-Bewegung (keine seitliche Auslenkung)

Parameter 1 (veränderbar): Federkonstante k
Parameter 2 (veränderbar): Startwinkel des mittleren Pendels

Grenzen des Modells:
  - Euler-Methode akkumuliert Fehler über Zeit (nicht energieerhaltend)
  - Reale Federn haben Masse und Eigenschwingung
  - Keine Dämpfung → realistischere Simulation würde Energie verlieren
"""

from math import sin, radians, degrees, cos, hypot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ============================================================
#  Klassen
# ============================================================

class Pendulum:
    def __init__(self, angle_deg, length_cm, pivot_x, mass=0.1):
        self.angle_deg = angle_deg      # aktueller Winkel in Grad
        self.angle_vel = 0.0            # Winkelgeschwindigkeit (Grad/s)
        self.length_m  = length_cm / 100
        self.pivot_x   = pivot_x
        self.mass      = mass           # Masse der Pendelkugel in kg
        self.g         = 9.81

    def tip_position(self):
        rad = radians(self.angle_deg)
        x = self.pivot_x + self.length_m * sin(rad)
        y = -self.length_m * cos(rad)
        return x, y

    def gravity_accel_deg(self):
        """Winkelbeschleunigung durch Schwerkraft (Grad/s²)"""
        rad   = radians(self.angle_deg)
        accel = -(self.g / self.length_m) * sin(rad)   # rad/s²
        return degrees(accel)


class Spring:
    def __init__(self, p1, p2, k, rest_length):
        self.p1          = p1
        self.p2          = p2
        self.k           = k            # Federkonstante (N/m)
        self.rest_length = rest_length

    def force_on_tips(self):
        """Gibt Kräfte auf beide Pendelspitzen zurück (fx1,fy1, fx2,fy2)."""
        x1, y1 = self.p1.tip_position()
        x2, y2 = self.p2.tip_position()
        dx, dy = x2 - x1, y2 - y1
        dist   = hypot(dx, dy)
        if dist == 0:
            return 0, 0, 0, 0
        stretch = dist - self.rest_length
        ux, uy  = dx / dist, dy / dist
        f       = self.k * stretch
        return f * ux, f * uy, -f * ux, -f * uy


# ============================================================
#  Physik-Schritt (Euler-Integration)
# ============================================================

def step(pendulums, springs, dt=0.005):
    accelerations = [p.gravity_accel_deg() for p in pendulums]

    for spring in springs:
        fx1, fy1, fx2, fy2 = spring.force_on_tips()
        for p, fx, fy, idx in [
            (spring.p1, fx1, fy1, pendulums.index(spring.p1)),
            (spring.p2, fx2, fy2, pendulums.index(spring.p2)),
        ]:
            rad       = radians(p.angle_deg)
            tan_x     = cos(rad)
            tan_y     = sin(rad)
            f_tan     = fx * tan_x + fy * tan_y
            a_rad     = f_tan / (p.mass * p.length_m)
            accelerations[idx] += degrees(a_rad)

    for p, accel in zip(pendulums, accelerations):
        p.angle_vel += accel * dt
        p.angle_deg += p.angle_vel * dt


# ============================================================
#  Hilfsfunktion: Simulation ohne Animation (für Graphen)
# ============================================================

def run_simulation(k_val, mid_angle_deg, steps=4000, dt=0.005):
    """
    Simuliert das System und gibt Zeitreihen der Winkel zurück.
    Parameter:
        k_val         – Federkonstante (N/m)
        mid_angle_deg – Startwinkel des mittleren Pendels (Grad)
    """
    pends = [
        Pendulum(angle_deg=45,           length_cm=100, pivot_x=-2),
        Pendulum(angle_deg=mid_angle_deg, length_cm=100, pivot_x=0),
        Pendulum(angle_deg=45,           length_cm=100, pivot_x=2),
    ]
    rest = abs(pends[1].pivot_x - pends[0].pivot_x)
    sprs = [
        Spring(pends[0], pends[1], k=k_val, rest_length=rest),
        Spring(pends[1], pends[2], k=k_val, rest_length=rest),
    ]

    t_arr = np.zeros(steps)
    a_arr = np.zeros((3, steps))

    for i in range(steps):
        t_arr[i]    = i * dt
        a_arr[0, i] = pends[0].angle_deg
        a_arr[1, i] = pends[1].angle_deg
        a_arr[2, i] = pends[2].angle_deg
        step(pends, sprs, dt)

    return t_arr, a_arr


# ============================================================
#  GRAPHEN – Parameter 1: Federkonstante k
# ============================================================

print("Erstelle Graphen …")

fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig1.suptitle("Parameter 1: Federkonstante k\n(Startwinkel Mitte = −15°)", fontsize=13)

k_values  = [1, 5, 15]
colors    = ["steelblue", "darkorange", "forestgreen"]
labels    = [f"k = {k} N/m" for k in k_values]

for k, color, label in zip(k_values, colors, labels):
    t, a = run_simulation(k_val=k, mid_angle_deg=-15)
    for ax_idx, ax in enumerate(axes1):
        ax.plot(t, a[ax_idx], color=color, label=label, alpha=0.8, linewidth=1)

for ax_idx, ax in enumerate(axes1):
    ax.set_ylabel(f"Pendel {ax_idx+1}\nWinkel (°)")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)

axes1[-1].set_xlabel("Zeit (s)")
fig1.tight_layout()
fig1.savefig("parameter_k.png", dpi=150)
print("  → parameter_k.png gespeichert")


# ============================================================
#  GRAPHEN – Parameter 2: Startwinkel (Chaos-Demonstration)
# ============================================================

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7))
fig2.suptitle("Parameter 2: Sensitive Abhängigkeit vom Startwinkel\n"
              "(Federkonstante k = 5 N/m)", fontsize=13)

angle_A = -15.0
angle_B = -15.1   # nur 0.1° Unterschied!

t_A, a_A = run_simulation(k_val=5, mid_angle_deg=angle_A, steps=6000)
t_B, a_B = run_simulation(k_val=5, mid_angle_deg=angle_B, steps=6000)

# Oben: Vergleich der Winkelverläufe von Pendel 2
ax = axes2[0]
ax.plot(t_A, a_A[1], label=f"Startwinkel = {angle_A}°", color="royalblue", linewidth=1)
ax.plot(t_B, a_B[1], label=f"Startwinkel = {angle_B}°", color="crimson",   linewidth=1)
ax.set_ylabel("Pendel 2 – Winkel (°)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title("Winkelverläufe: nahezu identische Startwerte divergieren")

# Unten: Absolute Differenz zwischen beiden Simulationen
diff = np.abs(a_A[1] - a_B[1])
ax2  = axes2[1]
ax2.plot(t_A, diff, color="purple", linewidth=1)
ax2.set_ylabel("|Differenz| (°)")
ax2.set_xlabel("Zeit (s)")
ax2.grid(True, alpha=0.3)
ax2.set_title("Wachstum der Abweichung über die Zeit (chaotisches Verhalten)")

fig2.tight_layout()
fig2.savefig("chaos_startwinkel.png", dpi=150)
print("  → chaos_startwinkel.png gespeichert")


# ============================================================
#  ANIMATION – Live-Visualisierung (Hauptsimulation)
# ============================================================

print("\nStarte Animation … (Fenster schließen zum Beenden)")

pendulums = [
    Pendulum(angle_deg=45,  length_cm=100, pivot_x=-2),
    Pendulum(angle_deg=-15, length_cm=100, pivot_x=0),
    Pendulum(angle_deg=45,  length_cm=100, pivot_x=2),
]

rest_len = abs(pendulums[1].pivot_x - pendulums[0].pivot_x)
springs  = [
    Spring(pendulums[0], pendulums[1], k=5, rest_length=rest_len),
    Spring(pendulums[1], pendulums[2], k=5, rest_length=rest_len),
]

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.set_xlim(-4, 4)
ax3.set_ylim(-1.5, 0.5)
ax3.set_aspect('equal')
ax3.set_title("Animation: Drei gekoppelte Pendel (k=5 N/m)")
ax3.set_xlabel("x (m)")
ax3.set_ylabel("y (m)")

# Aufhängepunkte
for p in pendulums:
    ax3.plot(p.pivot_x, 0, 'ks', markersize=6)

pendulum_lines = [
    ax3.plot([], [], 'o-', lw=2, markersize=10, label=f"Pendel {i+1}")[0]
    for i, p in enumerate(pendulums)
]
spring_line_1, = ax3.plot([], [], '--', color='orange', lw=2, label="Feder")
spring_line_2, = ax3.plot([], [], '--', color='orange', lw=2)
ax3.legend(loc="upper right", fontsize=8)


def update(frame):
    step(pendulums, springs, dt=0.005)
    tips = [p.tip_position() for p in pendulums]
    for i, p in enumerate(pendulums):
        tx, ty = tips[i]
        pendulum_lines[i].set_data([p.pivot_x, tx], [0, ty])
    spring_line_1.set_data([tips[0][0], tips[1][0]], [tips[0][1], tips[1][1]])
    spring_line_2.set_data([tips[1][0], tips[2][0]], [tips[1][1], tips[2][1]])
    return pendulum_lines + [spring_line_1, spring_line_2]


ani = animation.FuncAnimation(fig3, update, frames=5000, interval=10, blit=True)
plt.show()
