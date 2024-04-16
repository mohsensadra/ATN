import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import schedule
import time

#constant
rotation_speed=1000 # RPM
omega=rotation_speed*2*np.pi/60
time = 1
sampling_freq = 1000
sampling_frequency = 1000 # Hz
time_steps = int(sampling_frequency * time)
t = np.linspace(0, time, time_steps)
main_signal=np.sin(omega*t)






def phase_difference(signal1, signal2, sampling_freq):
       # Ensure both signals have the same length by truncating the longer one
       min_length = min(len(signal1), len(signal2))
       signal1 = signal1[:min_length]
       signal2 = signal2[:min_length]

       # Pad signals with zeros to the next power of 2
       next_power_of_2 = int(2 ** np.ceil(np.log2(min_length)))
       signal1 = np.pad(signal1, (0, next_power_of_2 - min_length), 'constant')
       signal2 = np.pad(signal2, (0, next_power_of_2 - min_length), 'constant')

       # Compute the FFT of the signals
       fft_signal1 = fft(signal1)
       fft_signal2 = fft(signal2)
           
       # Compute the phase of each signal at all frequencies
       phase1 = np.angle(fft_signal1)
       phase2 = np.angle(fft_signal2)

       # Compute the phase difference
       phase_difference_radians = np.abs(phase1 - phase2)
           
       # Convert phase difference from radians to degrees
       phase_difference_degrees = np.rad2deg(phase_difference_radians)

       return phase_difference_degrees,phase1


# Extract the data into numpy arrays]
df = pd.read_csv('vibration_data.csv')
time = df['Time'].to_numpy()
signal1 = df['Signal 1'].to_numpy()
signal2 = df['Signal 2'].to_numpy()

# Calculate the phase difference
phase_diff, phase1 = phase_difference(signal1, signal2, sampling_freq)
phase1,p1=phase_difference(main_signal, signal1, sampling_freq)
phase2=-phase1+phase_diff

# Calculate the total unbalance mass, total angular position (theta), and total radius
m1 = np.sqrt((2 * np.abs(signal.hilbert(signal1))) / np.sqrt(2 * np.pi * sampling_freq))
m2 = np.sqrt((2 * np.abs(signal.hilbert(signal2))) / np.sqrt(2 * np.pi * sampling_freq))
m1=np.mean(m1)
m2=np.mean(m2)
r1 = np.sqrt(np.abs(m1 * np.cos(phase_diff) / 2) / m1)
r2 = np.sqrt(np.abs(m2 * np.cos(phase_diff) / 2) / m2)
r1=np.mean(r1)
r2=np.mean(r2)
total_radius = np.sqrt(r1 + r2) / 2
mean_radius=np.mean(np.abs(total_radius))
total_mass = np.sqrt((m1 * r1 + m2 * r2) / total_radius)
mean_Mass = np.mean(np.abs(total_mass ))
total_theta_degrees = phase_diff
total_theta_degrees =np.mean(total_theta_degrees)

#print Calculation
print(f"Phase 1: {np.mean(phase1):.2f} degrees")
print(f"Phase 2: {np.mean(phase2):.2f} degrees")
print(f"Phase difference: {np.mean(phase_diff):.2f} degrees")
print(f"Total Removing Mass: {mean_Mass:.2f}  gram")
print(f"Total Theta: {total_theta_degrees:.2f}")

#Create separate polar plots for each signal
fig = plt.figure(figsize=(10, 6))
   
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
ax2 = fig.add_subplot(1, 2, 2, projection='polar')

ax1.set_theta_zero_location("N")
ax2.set_theta_zero_location("N")

ax1.set_title('PHASE.1')
ax2.set_title('PHASE.2')

ax1.set_theta_direction(1)
ax2.set_theta_direction(1)

ax1.axvline(np.mean(np.deg2rad(phase1)), color='orange', ls='--')
ax2.axvline(np.mean(np.deg2rad(phase2)), color='orange', ls='--')

plt.show()

