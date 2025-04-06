import torch
from torch import optim
from nflows.flows import Flow
from nflows import distributions
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
from scipy.special import logsumexp
import numpy as np
from pedalboard.io import AudioFile

class f_X():
    def __init__(self, sr=44100, chunk_duration=0.02):
        features = (sr*chunk_duration)//2 +1 #because we have 44100 samples per second, and 20 milliseconds of audio, rrft size is (n/2+1)
        self.base_dist = distributions.StandardNormal(shape=[features])
        self.transforms = []
        for _ in range(3):
            self.transforms.append(MaskedAffineAutoregressiveTransform(features=features, hidden_features=features))
        self.transform = CompositeTransform(self.transforms)
        self.flow = Flow(self.transform, self.base_dist)
        self.optimizer = optim.Adam(self.flow.parameters(),lr=1e-4)
    
    # def __call__(self,a):
    #     return self.flow.log_prob(a) #returns our pdf evaluations

    def log_pdf(self, x):
        '''Returns the log pdf of the input x'''
        return self.flow.log_prob(x)

    def sample(self):
        '''Samples from the representation'''
        return self.flow.sample(1)
    
    def sample_n(self, n):
        '''Samples n samples from the representation'''
        return self.flow.sample(n)
    
    def audio_process(self, underlying_signal_file, chunk_duration = 0.02, process_block_duration = 5.0): #this function processes audio chunks
        '''
        Online processing of audio chunks for the flow model.
        Automatically trains the flow model on the chunks.

        underlying_signal_file: path to the audio file
        chunk_duration: duration of each chunk in seconds
        '''

        with AudioFile(underlying_signal_file) as f:
            # Open an audio file to write to:
            chunk_size_frames = int(f.samplerate * chunk_duration)  # 20ms chunks
            process_block_frames = int(process_block_duration * f.samplerate)

            while f.tell() < f.frames: #while we haven't read the entire file
                block = f.read(process_block_frames)[0] #read 5 seconds of audio, first channel
                X = []
                
                for i in range(0, len(block)-chunk_size_frames + 1, chunk_size_frames):
                    subchunk = block[i:i+chunk_size_frames]
                    fft_result = np.fft.rfft(subchunk)
                    magnitude = np.abs(fft_result)
                    #frequencies = np.fft.rfftfreq(chunk.size, d=1./samplerate)
                    X.append(magnitude)
                
                X = np.stack(X)
                self.train(X)
    
    def train(self, x, n_iterations=1, verbose=False): #This is a very simple training loop
        if len(x.shape) == 1:
        # If input is flat, reshape to (1, features)
            x = x.reshape(1, -1)

        x_tensor = torch.tensor(x).float()
        
        for i in range(n_iterations):
            self.optimizer.zero_grad()
            loss = -self.flow.log_prob(inputs=x_tensor).mean()
            loss.backward()
            self.optimizer.step()
            if verbose:
                print(f"Iteration {i}, Loss: {loss.item()}")