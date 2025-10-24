import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import torchaudio
import os

def plot_waveform(wf, sample_rate, label="", figname=None):
    """
    Muestra el waveform (izquierda) y los MFCCs (derecha) de una señal de audio.

    Parámetros:
        wf (Tensor): señal de audio [1, N] o [N]
        sample_rate (int): frecuencia de muestreo (Hz)
        label (str): etiqueta opcional para el título
        figname (str): ruta para guardar la figura (si es None, solo muestra)
    """
    if isinstance(wf, torch.Tensor):
        wf = wf.squeeze().cpu()

    # === Transformación MFCC ===
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 320, "hop_length": 160, "n_mels": 23},
        log_mels=True
    )
    mfcc = mfcc_transform(wf.unsqueeze(0)).squeeze().cpu().numpy()  # [n_mfcc, time]

    # === Crear figura con 2 subplots ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.set_style("whitegrid")

    # --- Waveform ---
    time = torch.arange(0, len(wf)) / sample_rate
    axes[0].plot(time, wf.numpy(), color="steelblue", linewidth=1.0)
    axes[0].set_title("Waveform", fontsize=12)
    axes[0].set_xlabel("Tiempo [s]")
    axes[0].set_ylabel("Amplitud")

    # --- MFCC ---
    sns.heatmap(mfcc, ax=axes[1], cmap="viridis", cbar=True)
    axes[1].set_title("MFCCs", fontsize=12)
    axes[1].set_xlabel("Tiempo (frames)")
    axes[1].set_ylabel("Coeficiente MFCC")

    fig.suptitle(f"Audio: {label}", fontsize=14, y=1.02)
    plt.tight_layout()

    # === Guardar o mostrar ===
    if figname:
        name = os.path.join('img', f'{figname}.pdf')
        plt.savefig(name, bbox_inches="tight")
        print(f"Figura guardada en {name}")
    else:
        plt.show()

    plt.close(fig)
