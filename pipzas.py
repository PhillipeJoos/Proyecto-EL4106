
import os
import torch
import torchaudio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

data_path = 'data'
felipeipe = os.path.join(data_path, 'pipsas_felipeipe.wav')
pipo = os.path.join(data_path, 'pipsas_pipo.wav')
# Crear eje de tiempo en segundos
wave_pipo, sample_pipo = torchaudio.load(pipo, normalize=True)
wave_felipeipe, sample_felipeipe = torchaudio.load(felipeipe, normalize=True)
def audio_plot(waveform, sample_rate):
    duration = len(waveform) / sample_rate
    time = np.linspace(0, duration, len(waveform))

    # Crear el plot
    plt.figure(figsize=(12, 4))
    plt.plot(time, waveform)
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def mfcc(waveform, sample_rate):
    mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
    mfcc = mfcc_transform(waveform)
    print(f"Shape of MFCC: {mfcc.shape}")
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc[0].detach().numpy(), cmap='hot', aspect='auto')
    plt.title("MFCC")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

audio_plot(wave_pipo, sample_pipo)
audio_plot(wave_felipeipe, sample_felipeipe)
mfcc(wave_pipo, sample_pipo)
mfcc(wave_felipeipe, sample_felipeipe)
