import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# 1. Define parameters
# -----------------------------
# For simplicity, we assume circular, coplanar orbits.
# Distances in Astronomical Units (AU), time in arbitrary units.

# Orbital radii (AU)
R_earth = 1.0
R_mars = 1.52

# Orbital periods (arbitrary, but ratio roughly Earth: 1, Mars: ~1.88)
T_earth = 1.0
T_mars = 1.88

# Angular velocities (radians per unit time)
omega_earth = 2 * np.pi / T_earth
omega_mars = 2 * np.pi / T_mars

# Total animation frames
frames = 600

# We will define the spacecraft travel in 3 segments:
# 1) Earth -> Mars
# 2) Wait at Mars
# 3) Mars -> Earth
# Let each segment be frames/3 in duration
segment_frames = frames // 3

# -----------------------------
# 2. Precompute orbits
# -----------------------------
# Time array for entire animation
t_vals = np.linspace(0, 2, frames)  # 2 Earth years total, for example

# Positions of Earth
earth_x = R_earth * np.cos(omega_earth * t_vals * T_earth)
earth_y = R_earth * np.sin(omega_earth * t_vals * T_earth)
earth_z = np.zeros_like(t_vals)  # Keep orbits in the XY plane for simplicity

# Positions of Mars
mars_x = R_mars * np.cos(omega_mars * t_vals * T_earth)
mars_y = R_mars * np.sin(omega_mars * t_vals * T_earth)
mars_z = np.zeros_like(t_vals)

# -----------------------------
# 3. Define spacecraft trajectory
# -----------------------------
# We'll break it into segments. We'll do a simple linear interpolation
# from Earth's position to Mars' position, hold at Mars, then back to Earth.

spacecraft_x = np.zeros(frames)
spacecraft_y = np.zeros(frames)
spacecraft_z = np.zeros(frames)

# Segment 1: Earth -> Mars
for i in range(segment_frames):
    alpha = i / (segment_frames - 1)  # goes from 0 to 1
    # Start at Earth's position at frame=0, end at Mars' position at frame=segment_frames
    spacecraft_x[i] = (1 - alpha) * earth_x[0] + alpha * mars_x[segment_frames]
    spacecraft_y[i] = (1 - alpha) * earth_y[0] + alpha * mars_y[segment_frames]
    spacecraft_z[i] = 0.0  # Keep it in the plane

# Segment 2: Wait at Mars
for i in range(segment_frames, 2 * segment_frames):
    # Just stay near Mars
    spacecraft_x[i] = mars_x[i]
    spacecraft_y[i] = mars_y[i]
    spacecraft_z[i] = 0.0

# Segment 3: Mars -> Earth
for i in range(2 * segment_frames, frames):
    # Linearly interpolate from Mars' position at the start of this segment
    # to Earth's position at the end of the entire timespan
    alpha = (i - 2 * segment_frames) / (segment_frames - 1)
    spacecraft_x[i] = (1 - alpha) * mars_x[2 * segment_frames] + alpha * earth_x[-1]
    spacecraft_y[i] = (1 - alpha) * mars_y[2 * segment_frames] + alpha * earth_y[-1]
    spacecraft_z[i] = 0.0

# -----------------------------
# 4. Create 3D figure and animation
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Set up plot limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-1, 1])
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Earth -> Mars -> Earth (Simplified 3D Animation)')

# Plot objects
sun_plot, = ax.plot([0], [0], [0], 'o', color='yellow', markersize=12, label='Sun')
earth_plot, = ax.plot([], [], [], 'o', color='blue', label='Earth')
mars_plot, = ax.plot([], [], [], 'o', color='red', label='Mars')
spacecraft_plot, = ax.plot([], [], [], 'o', color='green', label='Spacecraft')

# Trajectory lines (optional)
earth_orbit_line, = ax.plot([], [], [], '--', color='blue', alpha=0.5)
mars_orbit_line, = ax.plot([], [], [], '--', color='red', alpha=0.5)
spacecraft_traj_line, = ax.plot([], [], [], '-', color='green', alpha=0.5)

ax.legend()

def init():
    """Initialize animation objects."""
    earth_plot.set_data([], [])
    earth_plot.set_3d_properties([])
    mars_plot.set_data([], [])
    mars_plot.set_3d_properties([])
    spacecraft_plot.set_data([], [])
    spacecraft_plot.set_3d_properties([])
    
    earth_orbit_line.set_data([], [])
    earth_orbit_line.set_3d_properties([])
    mars_orbit_line.set_data([], [])
    mars_orbit_line.set_3d_properties([])
    spacecraft_traj_line.set_data([], [])
    spacecraft_traj_line.set_3d_properties([])
    
    return (earth_plot, mars_plot, spacecraft_plot,
            earth_orbit_line, mars_orbit_line, spacecraft_traj_line)

def update(frame):
    """Update the positions of Earth, Mars, and the spacecraft for each frame."""
    # Update Earth
    earth_plot.set_data([earth_x[frame]], [earth_y[frame]])
    earth_plot.set_3d_properties([earth_z[frame]])
    
    # Update Mars
    mars_plot.set_data([mars_x[frame]], [mars_y[frame]])
    mars_plot.set_3d_properties([mars_z[frame]])
    
    # Update spacecraft
    spacecraft_plot.set_data([spacecraft_x[frame]], [spacecraft_y[frame]])
    spacecraft_plot.set_3d_properties([spacecraft_z[frame]])
    
    # Update orbit lines (tracing out the entire orbit from 0->frame)
    earth_orbit_line.set_data(earth_x[:frame+1], earth_y[:frame+1])
    earth_orbit_line.set_3d_properties(earth_z[:frame+1])
    
    mars_orbit_line.set_data(mars_x[:frame+1], mars_y[:frame+1])
    mars_orbit_line.set_3d_properties(mars_z[:frame+1])
    
    # Update spacecraft trajectory line
    spacecraft_traj_line.set_data(spacecraft_x[:frame+1], spacecraft_y[:frame+1])
    spacecraft_traj_line.set_3d_properties(spacecraft_z[:frame+1])
    
    return (earth_plot, mars_plot, spacecraft_plot,
            earth_orbit_line, mars_orbit_line, spacecraft_traj_line)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=20)
plt.show()
