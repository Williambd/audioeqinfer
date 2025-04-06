from IPython.display import display, HTML
from scipy import stats
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pedalboard.io import AudioFile
from patsy import dmatrix,bs,build_design_matrices


def EQ_FILE(input_file, output_file, g, samplerate = 44100):
  '''
  Imports an audio file, applies the EQ function g to it, and writes the result to a new file.
    input_file: path to the input audio file
    output_file: path to the output audio file
    g: function that outputs the EQ coefficients for each frequency
    samplerate: sample rate of the audio file
  '''
  with AudioFile(input_file) as f:
    # Open an audio file to write to:
    with AudioFile(output_file, 'w', f.samplerate, 1) as o:
      # Read one second of audio at a time, until the file is empty:
      while f.tell() < f.frames:
          chunk = f.read(f.samplerate)[0]
          fft_result = np.fft.rfft(chunk)
          freq = np.fft.rfftfreq(chunk.size, d=1./samplerate)
          together = np.array(list(zip(freq,fft_result)))

          processed = together[:,1]*g(together[:,0]) #vectorized, very fast (hopefully)
          #processed = fft_result#*1
          newsignal = np.fft.irfft(processed)
          
          o.write(newsignal)
          #wow. this is so cool. wowowowow.

def fft_diagram(array, samplerate,**kwargs):
    '''
    This function computes the FFT of the audio signal and plots the spectrum
    file: audio signal (Make this short, computation is expensiver for longer signals)
    '''

    fft_result = np.fft.rfft(array)
    magnitude = np.abs(fft_result)#[:len(fft_result)//2]
    freq = np.fft.rfftfreq(array.size, d=1./samplerate)
    ## Visualize the spectrum
    fig, ax = plt.subplots()
    ax.set(**kwargs)
    ax.plot(freq, magnitude, linewidth = 0.01)
    ax.fill_between(freq, magnitude, alpha=1)
    ax.set_title("Audio Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    plt.show()
    #return magnitude, freq


def show_audio_with_controls(file_path):
    timestamp = int(time.time() * 1000)  # This is just a trick to ensure the jupyter reloads the audio file
    print(file_path)
    display(HTML(f"<audio controls><source src='{file_path}?t={timestamp}' type='audio/mpeg'></audio>"))

class SoundGenerator(object):
    """
    SoundGenerator class
    This class generates a sound signal from an arbitrary distribution
    Using rejection sampling
    """
    def __init__(self, func):
        self.function = func

    def RejectionSample(self, n, proposal = stats.uniform(0, 1), M = 1):
        """
        Rejection sampling algorithm
        n: number of samples
        proposal: proposal distribution (if not specified, normal distribution is used)
        M: constant such that f(x) <= M * g(x) for all x
        """
        samples = []
        while len(samples) < n:
            x = proposal.rvs()
            u = np.random.uniform(0, 1)
            if u < self.function(x) / (M * proposal.pdf(x)):
                samples.append(x)
        return samples
    
    def run(self, n, *args, **kwargs):
        return self.RejectionSample(n, *args, **kwargs)