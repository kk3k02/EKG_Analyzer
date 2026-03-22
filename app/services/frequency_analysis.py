import numpy as np

class FrequencyAnalysisService:
    @staticmethod
    def compute_fft(ecg_data: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
        num_samples = len(ecg_data)

        fft_values = np.fft.rfft(ecg_data)

        frequencies = np.fft.rfftfreq(num_samples, d=1.0 / sampling_rate)

        amplitudes = np.abs(fft_values) / num_samples
        amplitudes[1:-1] *= 2

        return frequencies, amplitudes
