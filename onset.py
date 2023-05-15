import numpy as np 
from scipy import signal
import librosa
import sys
import matplotlib.pyplot as plt


x,sr = librosa.load(sys.argv[1] if len(sys.argv)>1 else 'test.wav')
x_duration = len(x)/sr
N = 2048
w = signal.hann(N)

# computing the local energy function
x_sq = x**2
energy_loc = np.convolve(x_sq,w**2,'same')
#plt.plot(x)
#plt.plot(energy_loc/1000,'r-')
#plt.xlabel('Frames')
#plt.legend(('Music as pressure wave','Local Energy Function, scaled down'),loc='upper right')
#plt.show()

# Discrete time differentiation and half-wave rectification
energy_loc_diff = np.concatenate((np.diff(energy_loc),np.array([0,])))
novelty_energy_func = np.copy(energy_loc_diff)
novelty_energy_func[energy_loc_diff < 0] = 0
plt.plot(range(len(novelty_energy_func)),novelty_energy_func,'b-')
#plt.xlabel('Frames')
#plt.legend(('Local Energy Function, scaled down','Novelty function, as energy diff'),loc='upper right')

# finding the peaks in the novelty function for onsets
peaks, props = signal.find_peaks(novelty_energy_func, prominence=0.010)
print(peaks/sr)
T_coef = np.arange(novelty_energy_func.shape[0])
peaks_sec = T_coef[peaks]
plt.plot(peaks_sec, novelty_energy_func[peaks], 'ro')
plt.xlabel('Frames')
plt.legend(('Novelty function, as energy diff','Peaks/ Onsets'),loc='upper right')
plt.show()