import numpy as np
from scipy.special import logsumexp
from .parameterfunction import ParameterFunction
from pedalboard.io import AudioFile
from patsy import build_design_matrices

class EQ_finder:
    def __init__(self, f_X, n_coefficients):
        '''
        f_X: Trained model approximating the distribution of audio chunks pretransformation
        n_coefficients: Number of coefficients for the basis spline representing EQ curve
        '''
        self.f_X = f_X
        self.n_coefficients = n_coefficients


    def find_eq_importance(self, sample, n_thetas_per_batch, n_audio_chunks, chunk_duration=0.02):
        '''
        Estimate EQ parameters via importance sampling across multiple audio chunks.
        This method is not recommended.
        '''
        pfunk = ParameterFunction(n_coefficients=self.n_coefficients)
        running_sum_log_numerator = np.full(self.n_coefficients, -np.inf)  # log(0) = -inf
        running_sum_log_denominator = -np.inf
        sr = None

        with AudioFile(sample) as f:
            total_frames = f.frames
            sr = f.samplerate
            total_chunks = int(total_frames / (sr * chunk_duration))  # 20ms chunks

            for n in range(n_audio_chunks):
                # Pick a random chunk
                chunk_start = np.random.randint(0, total_frames - int(sr * chunk_duration))
                f.seek(chunk_start)

                # Read a chunk of audio
                chunk = f.read(int(sr * chunk_duration))[0]
                results = self.fft_chunk(chunk, sr)  # FFT transform

                # Sample theta matrix
                thetamat = self.sample_theta(n_thetas_per_batch)  # (n_samples, n_coefficients)

                # Build basis matrix once
                basis_at_freqs = build_design_matrices([pfunk.design_info], {"x": results[:, 0]})[0]
                basis_at_freqs = np.asarray(basis_at_freqs)  # (n_freqs, n_basis_terms)

                # Predict all θs at once
                predicted_responses = basis_at_freqs @ thetamat.T  # (n_freqs, n_samples)
                predicted_responses = predicted_responses.T  # (n_samples, n_freqs)

                # g_inv for all θs
                g_inv_transformed = (1 / predicted_responses) * results[:, 1]  # (n_samples, n_freqs)
                g_inv_transformed = g_inv_transformed.astype(np.float32)

                # Compute log-pdfs for all transformed samples
                log_pdfs = self.f_X.log_pdf(g_inv_transformed).detach().cpu().numpy()  # (n_samples,)

                # Now, batch compute log(θ * p(X|θ))
                log_weighted_thetas = log_pdfs[:, None] + np.log(thetamat)  # (n_samples, n_coefficients)

                # Calculate logsumexp across θ samples for numerator and denominator
                log_batch_sum = logsumexp(log_weighted_thetas, axis=0)  # (n_coefficients,)
                log_batch_sum_denominator = logsumexp(log_pdfs)  # scalar

                # Update running sums
                running_sum_log_numerator = logsumexp(np.vstack([running_sum_log_numerator, log_batch_sum]), axis=0)
                running_sum_log_denominator = logsumexp([running_sum_log_denominator, log_batch_sum_denominator])

                print(f"Processed chunk {n + 1}/{n_audio_chunks}", end="\r")

        # After all chunks
        logE = running_sum_log_numerator - running_sum_log_denominator
        average_theta = np.exp(logE)

        return average_theta

    def find_eq_MH(self, sample, n_audio_chunks, burn_in = 10, keep_each = 1, std_dev = 0.01, chunk_duration=0.02):
        '''
        Estimate EQ parameters using a Metropolis-Hastings algorithm across random audio chunks.
        '''
        pfunk = ParameterFunction()
        sum_theta = np.zeros(self.n_coefficients)
        sr = None

        with AudioFile(sample) as f:
            total_frames = f.frames
            sr = f.samplerate
            
            #create initial theta
            theta_current = self.sample_theta(1)[0]
            #compute current logprob:
            pfunk.set_coefficients(theta_current)
            g_inv = lambda x: (1 / np.asarray(pfunk(x[:, 0]))) * x[:, 1]
            
            #find chunk
            chunk_start = np.random.randint(0, total_frames - int(sr * chunk_duration))
            f.seek(chunk_start)

            # Read a chunk of audio
            chunk = f.read(int(sr * chunk_duration))[0]
            results = self.fft_chunk(chunk, sr)

            transformed = g_inv(results)
            transformed = np.array([transformed])

            #calculate the pdf of the inverse eq at the audio chunk
            log_pdf_current = self.f_X.log_pdf(transformed.astype(np.float32)).item()

            for n in range(n_audio_chunks+burn_in):
                # Pick a random chunk
                chunk_start = np.random.randint(0, total_frames - int(sr * chunk_duration))
                f.seek(chunk_start)

                # Read a chunk of audio
                chunk = f.read(int(sr * chunk_duration))[0]
                results = self.fft_chunk(chunk, sr)

                # Create proposal theta
                theta_proposal = theta_current + np.random.normal(0, std_dev, self.n_coefficients)
                # Ensure proposal is valid (non-negative)
                theta_proposal = np.maximum(theta_proposal, 0)

                # Compute log pdf for proposal
                pfunk.set_coefficients(theta_proposal)
                g_inv = lambda x: (1 / np.asarray(pfunk(x[:, 0]))) * x[:, 1]

                # Transform x' into x
                transformed = g_inv(results)
                transformed = np.array([transformed])

                # Calculate the pdf of the inverse eq at the audio chunk
                log_pdf_proposal = self.f_X.log_pdf(transformed.astype(np.float32)).item()

                # Compute acceptance probability
                log_alpha = log_pdf_proposal - log_pdf_current
                alpha = np.exp(log_alpha)

                # Accept or reject the proposal
                if np.random.uniform(0, 1) < alpha:
                    theta_current = theta_proposal
                    log_pdf_current = log_pdf_proposal

                if n > burn_in and n % keep_each == 0:
                    sum_theta += theta_current
                
                print(f"Processed {n + 1}/{n_audio_chunks}", end="\r")
            return sum_theta / ((n_audio_chunks) / keep_each)

    def fft_chunk(self, chunk, samplerate):
        '''Compute FFT of an audio chunk'''
        fft_result = np.fft.rfft(chunk)
        freq = np.fft.rfftfreq(chunk.size, d=1./samplerate)
        magnitude = np.abs(fft_result)
        together = np.array(list(zip(freq,magnitude)))
        return together
    
    def sample_theta(self,n):
        '''Sample theta values uniformly in [0,1]'''
        return np.random.uniform(0,1,(n,self.n_coefficients))