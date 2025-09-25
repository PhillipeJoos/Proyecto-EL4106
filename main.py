import os
import torch
import torchaudio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class CustomSpeechCommands(Dataset):
    def __init__(self, root, files_list, download=False): # cambiar esto a true si no se tiene el dataset
        # Descargar el dataset completo si no existe
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, 
            download=download
        )
        
        # Leer la lista de archivos de tu partición
        with open(files_list, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        
        print(f"Primeros 3 paths en {files_list}: {self.file_paths[:3]}")
        
        # Obtener todos los paths del dataset original y normalizarlos
        self.all_paths = []
        for item in self.dataset._walker:
            # Extraer solo la parte relativa del path
            full_path = item
            # Convertir a formato relativo (como en los archivos .txt)
            relative_path = os.path.relpath(full_path, start=os.path.join(root, "SpeechCommands", "speech_commands_v0.02"))
            # En Windows, convertir barras invertidas a normales
            self.all_paths.append(relative_path)
        
        print(f"Primeros 3 paths normalizados del dataset: {self.all_paths[:3]}")
        
        # Crear máscara para filtrar los archivos de tu partición
        self.indices = [
            i for i, path in enumerate(self.all_paths) 
            if path in self.file_paths
        ]
        
        print(f"Total archivos en dataset: {len(self.all_paths)}")
        print(f"Archivos en {files_list}: {len(self.file_paths)}")
        print(f"Archivos encontrados: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Obtener el índice original en el dataset completo
        original_idx = self.indices[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[original_idx]
        return waveform, sample_rate, label, speaker_id, utterance_number

# Configuración
ROOT_DIR = "data"
TRAIN_LIST = os.path.join(ROOT_DIR,"train_list.txt")
VAL_LIST = os.path.join(ROOT_DIR, "val_list.txt")
TEST_LIST = os.path.join(ROOT_DIR, "test_list.txt")

# Verificar que los archivos de lista existen
for file_list in [TRAIN_LIST, VAL_LIST, TEST_LIST]:
    if not os.path.exists(file_list):
        print(f"Error: No se encuentra {file_list}")
    else:
        print(f"Encontrado: {file_list}")

# Crear directorio si no existe
os.makedirs(ROOT_DIR, exist_ok=True)

# Crear datasets personalizados
try:
    train_dataset = CustomSpeechCommands(ROOT_DIR, TRAIN_LIST)
    val_dataset = CustomSpeechCommands(ROOT_DIR, VAL_LIST)
    test_dataset = CustomSpeechCommands(ROOT_DIR, TEST_LIST)
    
    print("¡Datasets cargados exitosamente!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Plotear la primera muestra del train dataset
    if len(train_dataset) > 0:
        waveform, sample_rate, label, speaker_id, utterance_number = train_dataset[2]
        
        # Convertir a numpy y eliminar la dimensión del canal
        wave = waveform.squeeze().numpy()  # .squeeze() elimina dimensiones de tamaño 1
        
        # Crear eje de tiempo en segundos
        duration = len(wave) / sample_rate
        time = np.linspace(0, duration, len(wave))
        
        # Crear el plot
        plt.figure(figsize=(12, 4))
        plt.plot(time, wave)
        plt.title(f'Forma de Onda - Label: "{label}" - Speaker: {speaker_id}')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Información adicional
        print(f"\nInformación del audio:")
        print(f"Label: {label}")
        print(f"Speaker ID: {speaker_id}")
        print(f"Duración: {duration:.2f} segundos")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Tamaño del waveform: {waveform.shape}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()