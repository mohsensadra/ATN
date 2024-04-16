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
sampling_frequency = 1000 # Hz
time_steps = int(sampling_frequency * time)
t = np.linspace(0, time, time_steps)
main_signal=np.cos(omega*t)
df = pd.read_csv('vibration_data.csv')


# Extract the data into numpy arrays
time = df['Time'].to_numpy()
signal1 = df['Signal 1'].to_numpy()
signal2 = df['Signal 2'].to_numpy()

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

       return phase_difference_degrees


sampling_freq = 1000
# Replace with your actual sampling frequency
# Calculate the phase difference
phase_diff = phase_difference(signal1, signal2, sampling_freq)
pahse1=phase_difference(main_signal, signal1, sampling_freq)


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
phase_diff_degrees = np.rad2deg(phase_diff)
total_theta_degrees = (180 - phase_diff_degrees) / 2

print(f"Phase difference: {np.mean(phase_diff):.2f} degrees")
print(f"Phase 1: {np.mean(pahse1):.2f} degrees")
#print(r1)
print("Total unbalance mass: ",mean_Mass, " g")
print("Total angular position: ", total_theta_degrees, " degrees")
