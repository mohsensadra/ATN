"""
The simple calculation using FFT (Fast Fourier Transform) for
Analyzing 2 signals from a CSV file contains Time, Signal 1,
Signal 2, Phase Difference come from "Response simulator.py"
and find the removing mass properties contain Angle (Theta),
Mass magnitude, and radius for a Balancing Machine which it
wants to remove some mass to reach out  to dynamic balancing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Read the CSV file
df = pd.read_csv('vibration_data.csv')

# Extract the data into numpy arrays
time = df['Time'].to_numpy()
signal1 = df['Signal 1'].to_numpy()
signal2 = df['Signal 2'].to_numpy()

# Calculate the RMS (Root Mean Square) values of the signals
rms1 = np.sqrt(np.mean(signal1**2))
rms2 = np.sqrt(np.mean(signal2**2))

# Calculate the cross-spectrum of the signals
signal1_fft = np.fft.fft(signal1)
signal2_fft = np.fft.fft(signal2)
cross_spectrum = signal1_fft * np.conjugate(signal2_fft)

# Find the frequency with the maximum cross-spectrum magnitude
max_index = np.argmax(np.abs(cross_spectrum))
frequency = max_index * (1 / time.size) * (1 / np.ptp(time))

# Compute the phase difference between signals
phase_difference = np.angle(cross_spectrum[max_index])

# Calculate the total unbalance mass, total angular position (theta), and total radius
m1 = np.sqrt((2 * np.abs(signal.hilbert(signal1))) / np.sqrt(2 * np.pi * frequency))
m2 = np.sqrt((2 * np.abs(signal.hilbert(signal2))) / np.sqrt(2 * np.pi * frequency))

r1 = np.sqrt((m1 * np.cos(phase_difference / 2)) / m1)
r2 = np.sqrt((m2 * np.cos((np.pi - phase_difference) / 2)) / m2)

total_radius = np.sqrt((r1 + r2) / 2)
mean_radius=np.mean(np.abs(total_radius))
total_mass = np.sqrt((m1 * r1 + m2 * r2) / total_radius)
mean_Mass = np.mean(np.abs(total_mass ))
# Convert phase difference from radians to degrees
phase_difference_degrees = np.rad2deg(phase_difference)
total_theta_degrees = (180 - phase_difference_degrees) / 2
"""
print(f"RMS of Signal 1: {rms1:.4f}")
print(f"RMS of Signal 2: {rms2:.4f}")
print(f"Phase difference: {phase_difference_degrees:.2f} degrees")
print(f"Total unbalance mass: {np.asscalar(mean_Mass):.4f} g")
print(f"Total angular position (theta): {total_theta_degrees:.2f} degrees")
#print(f"Total radius: {np.asscalar(total_radius):.4f} m")
"""

print(f"RMS of Signal 1: {rms1:.4f}")
print(f"RMS of Signal 2: {rms2:.4f}")
print(f"Phase difference: {phase_difference_degrees:.2f} degrees")
print(f"Total unbalance mass: {mean_Mass:.4f} g")
print(f"Total angular position (theta): {total_theta_degrees:.2f} degrees")
print(f"Total radius: {mean_radius:.4f} m")

# Plot the signals
plt.figure(figsize=(10, 6))
plt.plot(time, signal1, label="Signal 1")
plt.plot(time, signal2, label="Signal 2")
plt.title("Simulated Acceleration Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
