import sys
import numpy as np 
import librosa
from scipy import signal
import matplotlib.pyplot as plt

x,sr = librosa.load(sys.argv[1] if len(sys.argv)>1 
        else 'test.wav') # loading the audio data
x_duration = len(x)/sr 
N = 2048 # Frames we want to consider in a window
w = signal.hann(N) # Choosing the Hann window

# computing the local energy function
x_sq = x**2 
# The local energy formula given
energy_loc = np.convolve(x_sq,w**2,'same') 
plt.plot(x)
plt.plot(energy_loc/25,'r-')
plt.xlabel('Frames') 
plt.legend(('Audio wave','Local Energy Function, scaled down'),loc='upper right')
plt.show()

# Discrete time differentiation and half-wave rectification
energy_loc_diff = np.concatenate((np.diff(energy_loc),
                np.array([0,]))) 
# an ending zero for introducing the last 0-change
novelty_energy_func = np.copy(energy_loc_diff)
novelty_energy_func[energy_loc_diff < 0] = 0 
# Half-wave rectification
plt.plot(range(len(novelty_energy_func)),
            novelty_energy_func,'b-')

# finding the peaks in the novelty function for onsets
peaks, props = signal.find_peaks(novelty_energy_func, 
                prominence=0.012)
# finding the peaks, using scipy.signal, 
# tuning param prominence set to 0.012
T_coef = np.arange(novelty_energy_func.shape[0])
peaks_sec = T_coef[peaks]
plt.plot(peaks_sec, novelty_energy_func[peaks], 'ro')
plt.xlabel('Frames') 'Local Energy Function, scaled down',
plt.legend(('Novelty function, as energy diff','Peaks/ Note-onsets'),loc='upper right')
plt.show()

print('The onset times are : ')
print(*list(peaks_sec/sr))
