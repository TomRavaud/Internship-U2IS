import numpy as np
import matplotlib.pyplot as plt


def motion(x, u, dt):
    """
    Motion model (backward Euler method
    applied to differential drive kinematic model)
    """

    x[2] += u[1] * dt
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x

def predict_trajectory(x_init, v, omega, predict_time=3.0, dt=0.05):
    """
    Predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= predict_time:
        x = motion(x, [v, omega], dt)
        trajectory = np.vstack((trajectory, x))
        time += dt

    return trajectory

# Time for which the future trajectory is predicted
T = 3  # seconds

# Integration step
dt = 0.05  # seconds

# Set the list of angular velocities
omegas = [0.5, 0.3, 0., -0.3, -0.5]
colors = ['r', 'g', 'b']
velocities = [0.3, 0.6, 0.9]


ax = plt.subplot(111)
ax.axis('equal')

for i, omega in enumerate(omegas):
    
    for j, v in enumerate(velocities):
        
        trajectory = predict_trajectory([0., 0., 0., 0., 0.], v, omega, T, dt)

        ax.plot(trajectory[:, 1],
                trajectory[:, 0],
                label=f'vel = {v}',
                color=colors[j])

plt.legend()
plt.show()
