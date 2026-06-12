from math import sin, radians, degrees, cos, hypot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# A single pendulum
# -----------------------------
class Pendulum:
    def __init__(self, angle_deg, length_cm, pivot_x):
        self.angle_deg = angle_deg   # current angle (degrees)
        self.angle_vel = 0           # current angular speed (degrees/second)
        self.length_m = length_cm / 100
        self.pivot_x = pivot_x
        self.g = 9.81

    def tip_position(self):
        # (x, y) of the bob, in meters
        rad = radians(self.angle_deg)
        x = self.pivot_x + self.length_m * sin(rad)
        y = -self.length_m * cos(rad)
        return x, y

    def gravity_accel_deg(self):
        # angular acceleration from gravity alone (deg/s^2)
        rad = radians(self.angle_deg)
        accel = -(self.g / self.length_m) * sin(rad)  # rad/s^2
        return degrees(accel)


# -----------------------------
# A spring connecting two pendulum tips
# -----------------------------
class Spring:
    def __init__(self, p1, p2, k, rest_length):
        self.p1 = p1
        self.p2 = p2
        self.k = k                  # spring stiffness
        self.rest_length = rest_length

    def force_on_tips(self):
        # returns (fx1, fy1, fx2, fy2): force vectors on each tip
        x1, y1 = self.p1.tip_position()
        x2, y2 = self.p2.tip_position()

        dx = x2 - x1
        dy = y2 - y1
        dist = hypot(dx, dy)
        if dist == 0:
            return 0, 0, 0, 0

        # how much the spring is stretched/compressed
        stretch = dist - self.rest_length

        # unit vector pointing from p1 to p2
        ux, uy = dx / dist, dy / dist

        # spring force magnitude (Hooke's law: F = k * stretch)
        f = self.k * stretch

        # pull p1 toward p2, push/pull p2 accordingly (Newton's 3rd law)
        fx1, fy1 = f * ux, f * uy
        fx2, fy2 = -f * ux, -f * uy
        return fx1, fy1, fx2, fy2


# -----------------------------
# Create three pendulums, side by side
# -----------------------------
pendulums = [
    Pendulum(angle_deg=25, length_cm=100, pivot_x=-2),
    Pendulum(angle_deg=-15, length_cm=100, pivot_x=0),
    Pendulum(angle_deg=25, length_cm=100, pivot_x=2),
]

# rest length = distance between neighboring tips when pendulums hang straight down
rest_len = abs(pendulums[1].pivot_x - pendulums[0].pivot_x)

springs = [
    Spring(pendulums[0], pendulums[1], k=5, rest_length=rest_len),
    Spring(pendulums[1], pendulums[2], k=5, rest_length=rest_len),
]


# -----------------------------
# Physics step: combine gravity + spring forces
# -----------------------------
def step(dt=0.01):
    # 1. start with gravity-only angular acceleration for each pendulum
    accelerations = [p.gravity_accel_deg() for p in pendulums]

    # 2. add the effect of each spring
    for spring in springs:
        fx1, fy1, fx2, fy2 = spring.force_on_tips()

        # convert the force on each tip into an angular acceleration contribution.
        # we project the force onto the "tangential direction" of the pendulum's swing
        # (the direction the bob actually moves when the angle changes).
        for p, fx, fy, idx in [(spring.p1, fx1, fy1, pendulums.index(spring.p1)),
                                (spring.p2, fx2, fy2, pendulums.index(spring.p2))]:
            rad = radians(p.angle_deg)
            # tangential direction (perpendicular to the rod)
            tangent_x = cos(rad)
            tangent_y = sin(rad)

            # force component along the tangent
            f_tangent = fx * tangent_x + fy * tangent_y

            # torque = force * length -> angular accel = torque / (mass * length^2)
            # we assume mass = 1 for simplicity (cancels out nicely)
            angular_accel_rad = f_tangent / p.length_m
            accelerations[idx] += degrees(angular_accel_rad)

    # 3. update velocities and angles using the combined acceleration
    for p, accel_deg in zip(pendulums, accelerations):
        p.angle_vel += accel_deg * dt
        p.angle_deg += p.angle_vel * dt


# -----------------------------
# Set up the plot
# -----------------------------
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-1.5, 0.5)
ax.set_aspect('equal')

pendulum_lines = [ax.plot([], [], 'o-', lw=2, markersize=12)[0] for _ in pendulums]
spring_line_1, = ax.plot([], [], '--', color='orange', lw=2)
spring_line_2, = ax.plot([], [], '--', color='orange', lw=2)


def update(frame):
    step()

    tips = [p.tip_position() for p in pendulums]

    for i, p in enumerate(pendulums):
        tip_x, tip_y = tips[i]
        pendulum_lines[i].set_data([p.pivot_x, tip_x], [0, tip_y])

    spring_line_1.set_data([tips[0][0], tips[1][0]], [tips[0][1], tips[1][1]])
    spring_line_2.set_data([tips[1][0], tips[2][0]], [tips[1][1], tips[2][1]])

    return pendulum_lines + [spring_line_1, spring_line_2]


ani = animation.FuncAnimation(fig, update, frames=3000, interval=10, blit=True)
plt.show()