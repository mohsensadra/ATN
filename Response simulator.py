"""
A Signal Simulator for a Vertical Balancing Machine which it can
returna CSV file contains a random response for an unbalanced
shaft/disk/wheel/cylinderwith given Diameter, Weight, Width,
and RPM and undefined (random) mass (2 masses)which positions,
magnitudes, radiuses and planes of theirs are random 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
# Cylinder parameters
cylinder_diameter = 0.141 # m
cylinder_mass = 1.014 # kg
rotation_speed = 1000 # rpm

# Unbalance mass parameters
m1 = 0.001 #random.uniform(0.0, 0.02) # kg
m2 = 0.001 #random.uniform(0.0, 0.02) # kg
r1 = 0.005 #random.uniform(0.0, 0.01) # m
r2 = 0.005 #random.uniform(0.0, 0.01) # m
theta1 = 45 #random.uniform(0.0, 360.) # degrees
theta2 = -90 #random.uniform(0.0, 360.) # degrees

# Convert theta1 and theta2 to radians
theta1_rad = np.deg2rad(theta1)
theta2_rad = np.deg2rad(theta2)
print ('Theta1:',theta1,'Theta2:',theta2,' Phase difference:',theta2-theta1, ' deg')
# Sensor positions
sensor1_distance = 0.308 # m
sensor2_distance = sensor1_distance + 0.19 # m

# Conversion factors
rpm_to_rad_per_sec = 2 * np.pi / 60

# Simulation parameters
omega = rotation_speed * rpm_to_rad_per_sec  # Angular velocity (rad/s)
time = 1

# Simulation time (s)
sampling_frequency = 1000 # Hz
time_steps = int(sampling_frequency * time)

# Generate time array
t = np.linspace(0, time, time_steps)

# Calculate the net unbalance force
unbalance_force_1 = m1 * r1 * omega ** 2 * np.cos(omega * t + theta1_rad)
unbalance_force_2 = m2 * r2 * omega ** 2 * np.cos(omega * t + theta2_rad)
net_unbalance_force = unbalance_force_1 + unbalance_force_2

# Calculate acceleration at the sensors
amplitude1 = np.max(net_unbalance_force / sensor1_distance)
amplitude2 = np.max(net_unbalance_force / sensor2_distance)

# Generate sinusoidal signals with different amplitudes and phase shifts
signal1 = amplitude1 * np.sin(omega * t + theta1_rad)
signal2 = amplitude2 * np.sin(omega * t + theta2_rad)
#main_signal=amplitude1 *np.sin(omega*t)
# Compute phase difference between signals
complex_signal1 = signal1 + 1j * np.zeros_like(signal1)
complex_signal2 = signal2 + 1j * np.zeros_like(signal2)

# Create a DataFrame and save to CSV
data = {
    'Time': t,
    'Signal 1': signal1,
    'Signal 2': signal2,
}
df = pd.DataFrame(data)
df.to_csv('vibration_data.csv', index=False)

# Plot the signals
plt.figure(figsize=(10, 6))
plt.plot(t, signal1, label="Signal 1")
plt.plot(t, signal2, label="Signal 2")
#plt.plot(t, main_signal, label="Main Signal")
plt.title("Simulated Acceleration Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
