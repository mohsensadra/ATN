"""
The simple calculation using FFT (Fast Fourier Transform) for
Analyzing 2 signals from a CSV file contains Time, Signal 1,
Signal 2,  come from "Response simulator.py"
and it finds  Phase Difference , the removing mass properties
contain Angle (Theta),Mass magnitude, and radius for
a Balancing Machine which it wants to remove some mass to
reach out  to dynamic balancing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import schedule
import time
import sys

try:
    color = sys.stdout.shell
except AttributeError:
    raise RuntimeError("Use IDLE")

#constant
rotation_speed=1000 # RPM
omega=rotation_speed*2*np.pi/60
time = 1
sampling_frequency = 1000 # Hz
time_steps = int(sampling_frequency * time)
t = np.linspace(0, time, time_steps)
main_signal=np.sin(omega*t)

 # Create separate polar plots for each signal
fig = plt.figure(figsize=(10, 5))
   
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
ax2 = fig.add_subplot(1, 2, 2, projection='polar')

ax1.set_theta_zero_location("N")
ax2.set_theta_zero_location("N")
ax1.set_title('Phase.1, Correction')
ax2.set_title('Phase.2, Correction')

ax1.set_theta_direction(1)
ax2.set_theta_direction(1)



def Balancing(mainsignal,t):
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
    mainsignal_fft = np.fft.fft(mainsignal)
    cross_spectrum_main = mainsignal_fft * np.conjugate(signal1_fft)
    cross_spectrum = signal1_fft * np.conjugate(signal2_fft)
    
    # Find the frequency with the maximum cross-spectrum magnitude
    max_index = np.argmax(np.abs(cross_spectrum))
    frequency = max_index * (1 / time.size) * (1 / np.ptp(time))

    # Compute the phase difference between signals
    phase_difference = np.angle(cross_spectrum[max_index])
    theta1=np.angle(cross_spectrum_main[max_index])
    theta2=phase_difference+theta1
    
    # Calculate the total unbalance mass, total angular position (theta), and total radius
    m1 = np.sqrt((2 * np.abs(signal.hilbert(signal1))) / np.sqrt(2 * np.pi * frequency))
    m2 = np.sqrt((2 * np.abs(signal.hilbert(signal2))) / np.sqrt(2 * np.pi * frequency))

    r1 = np.sqrt(np.abs(m1 * np.cos(phase_difference) / 2) / m1)
    r2 = np.sqrt(np.abs(m2 * np.cos(phase_difference) / 2) / m2)

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
    correction_angle=(np.rad2deg(theta1)+np.rad2deg(theta2))/2
    color.write(f"Removing Correction Angle: {correction_angle:.1f} degrees\n","STRING ")
    color.write(f"Correction Weight: {mean_Mass:.4f} gr \n","STRING ")
    
    print(f"RMS of Signal 1: {rms1:.4f}")
    print(f"RMS of Signal 2: {rms2:.4f}")
    print(f"Theta1: {np.rad2deg(theta1):.2f} degrees")
    print(f"Theta2: {np.rad2deg(theta2):.2f} degrees")
    #print(f"Phase difference: {phase_difference_degrees:.2f} degrees")
    #print(f"Total unbalance mass: {mean_Mass:.4f} g")
    print(f"Total angular position (theta): {total_theta_degrees:.2f} degrees")
    print(f"Total radius: {mean_radius:.4f} m")
    
    """"
    # Create separate polar plots for each signal
    fig = plt.figure(figsize=(10, 5))
   
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')

    ax1.set_theta_zero_location("N")
    ax2.set_theta_zero_location("N")
    ax1.set_title('PHASE.1, PHASE.2')
    ax2.set_title('CORRECTION')

    ax1.set_theta_direction(1)
    ax2.set_theta_direction(1)
    """
    signal1_angle = np.unwrap(np.angle(signal1_fft.flatten()))
    signal2_angle = np.unwrap(np.angle(signal2_fft.flatten()))

    flattened_signal1 = signal1.flatten()
    flattened_signal2 = signal2.flatten()
    
    #x1=np.pi/2-phase_difference_degrees
    x1=theta1
    x2=theta2
    mean_line=correction_angle
    ax1.axvline(x1, color='orange', ls='--')
    ax1.axvline(np.deg2rad(mean_line), color='green', lw=2)
    ax2.axvline(x2, color='orange', ls='--')
    ax2.axvline(np.deg2rad(mean_line), color='green', lw=2)
    #print("1")
    #plt.show(block=False)
    #sleep.time(1)
    #plt.close()
    #for i in range(10):
    plt.show(block=False)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(2)
    
#while True:
    #Balancing(main_signal,t)
   

Balancing(main_signal,t)
