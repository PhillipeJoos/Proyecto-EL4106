import os
import torch
import torchaudio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
# Asegurarse de usar el backend correcto
torchaudio.set_audio_backend("sox_io")

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
            relative_path = relative_path.replace("\\", "/")
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

# Definir la transformación
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13, # número de coeficientes MFCC a extraer
    melkwargs={"n_fft": 320, "hop_length": 160, "n_mels": 23} # 320 = 20ms, 160 = 10ms, 23 = número de filtros mel
)

# Función para extraer características de un dataset
def extract_features(dataset):
    features = []
    labels = []
    for idx in range(len(dataset)):
        waveform, sample_rate, label, _, _ = dataset[idx]

        # aplicar transformación (ej: MFCC)
        feat = mfcc_transform(waveform).squeeze(0)  # (n_mfcc, time)
        
        features.append(feat.numpy())
        labels.append(label)
        
        if idx % 2000 == 0:  # solo para debug
            print(f"Procesado {idx}/{len(dataset)} muestras")
    
    return features, labels

# Crear datasets personalizados
train_dataset = CustomSpeechCommands(ROOT_DIR, TRAIN_LIST)
val_dataset = CustomSpeechCommands(ROOT_DIR, VAL_LIST)
test_dataset = CustomSpeechCommands(ROOT_DIR, TEST_LIST)

print("¡Datasets cargados exitosamente!")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Plotear la primera muestra del train dataset
if len(train_dataset) > 0:
    waveform, sample_rate, label, speaker_id, utterance_number = train_dataset[15000]
    
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

    # Aplicar transformación MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 320, "hop_length": 160, "n_mels": 23}
    )
    mfcc = mfcc_transform(waveform)
    print(f"Shape of MFCC: {mfcc.shape}")
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc[0].detach().numpy(), cmap='hot', aspect='auto')
    plt.title("MFCC")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Extraer características de los datasets
print("Extrayendo características del conjunto de entrenamiento...")
train_features, train_labels = extract_features(train_dataset)
print("Extrayendo características del conjunto de validación...")
val_features, val_labels = extract_features(val_dataset)
print("Extrayendo características del conjunto de prueba...")
test_features, test_labels = extract_features(test_dataset)
print("¡Extracción de características completada!")
print(f"Total características de entrenamiento: {len(train_features)}")
print(f"Total características de validación: {len(val_features)}")
print(f"Total características de prueba: {len(test_features)}")

# Guardar las características y etiquetas en archivos
np.save(os.path.join(ROOT_DIR, "train_features.npy"), train_features)
np.save(os.path.join(ROOT_DIR, "train_labels.npy"), train_labels)
np.save(os.path.join(ROOT_DIR, "val_features.npy"), val_features)
np.save(os.path.join(ROOT_DIR, "val_labels.npy"), val_labels)
np.save(os.path.join(ROOT_DIR, "test_features.npy"), test_features)
np.save(os.path.join(ROOT_DIR, "test_labels.npy"), test_labels)
print("¡Características y etiquetas guardadas en archivos .npy!")