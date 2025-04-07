import numpy as np
from typing import Dict, Any


class NoiseInjector:
    """
    Responsible for adding noise to the generated time series data.
    """
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_settings: Dict[str, Any]) -> np.ndarray:
        """
        Add noise to the given time series data.
        
        Args:
            data: Time series data array (rows are time points, columns are state variables)
            noise_settings: Dictionary with noise settings
                - noise_type: 'uniform' or 'gaussian'
                - noise_level: Noise level relative to signal RMS
                
        Returns:
            The noisy time series data
        """
        if not noise_settings.get('add_noise', False):
            return data
        
        noise_type = noise_settings.get('noise_type', 'gaussian')
        noise_level = noise_settings.get('noise_level', 0.05)
        
        # Calculate signal RMS (root mean square)
        signal_rms = np.sqrt(np.mean(data**2))
        target_noise_rms = signal_rms * noise_level
        
        # Generate noise with specified distribution
        if noise_type == 'uniform':
            # Uniform noise in [-a, a] has RMS = a/sqrt(3)
            # So we need a = target_noise_rms * sqrt(3)
            a = target_noise_rms * np.sqrt(3)
            noise = np.random.uniform(-a, a, size=data.shape)
        else:  # 'gaussian'
            # Gaussian noise with std = s has RMS = s
            # So we need s = target_noise_rms
            noise = np.random.normal(0, target_noise_rms, size=data.shape)
        
        # Add noise to data
        noisy_data = data + noise
        
        return noisy_data 