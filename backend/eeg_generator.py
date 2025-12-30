import numpy as np

class EEGDataGenerator:
    def __init__(self, fs=256, duration=4, n_channels=14):
        self.fs = fs
        self.duration = duration
        self.n_samples = fs * duration
        self.n_channels = n_channels
        
        self.freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 50)
        }

        # Updated recipes for "Positive" and "Negative" states
        self.state_patterns = {
            'positive': { # Simulates a calm, focused, resting state
                'delta': 1.0, 'theta': 0.8, 'alpha': 1.5, 'beta': 0.5, 'gamma': 0.2
            },
            'negative': { # Simulates a stressed, anxious, or frustrated state
                'delta': 0.5, 'theta': 1.0, 'alpha': 0.4, # Suppressed Alpha
                'beta': 1.8, 'gamma': 1.2 # Dominant Beta/Gamma
            }
        }
        print("✅ EEG Data Generator Initialized.")

    def _generate_single_wave(self, freq, amplitude, noise_level=0.1):
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        signal = amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
        noise = np.random.normal(0, noise_level * amplitude, self.n_samples)
        return signal + noise

    def generate_eeg_sample(self, state):
        pattern = self.state_patterns[state]
        eeg_data = np.zeros((self.n_channels, self.n_samples))

        for ch_idx in range(self.n_channels):
            channel_signal = np.zeros(self.n_samples)
            for band, (low_freq, high_freq) in self.freq_bands.items():
                base_amplitude = pattern[band]
                amplitude = base_amplitude * np.random.uniform(0.8, 1.2)
                freq = np.random.uniform(low_freq, high_freq)
                channel_signal += self._generate_single_wave(freq, amplitude)
            
            if np.random.rand() < 0.05:
                spike_start = np.random.randint(0, self.n_samples - 50)
                spike_amp = np.random.uniform(5, 15)
                eeg_data[ch_idx, spike_start:spike_start+25] += spike_amp
            
            eeg_data[ch_idx, :] = channel_signal
        return eeg_data

    def generate_dataset(self, n_samples_per_class=500):
        print(f"\nGenerating dataset with {n_samples_per_class} samples for each state...")
        X_list, y_list = [], []
        state_labels = {'positive': 0, 'negative': 1}

        for state, label in state_labels.items():
            print(f"  -> Generating '{state}' samples...")
            for _ in range(n_samples_per_class):
                X_list.append(self.generate_eeg_sample(state).flatten())
                y_list.append(label)
        
        print("\n✅ Dataset Generation Complete!")
        return np.array(X_list), np.array(y_list)