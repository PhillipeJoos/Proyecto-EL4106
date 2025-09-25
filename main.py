import os
import torch
import torchaudio
from torch.utils.data import Dataset

class CustomSpeechCommands(Dataset):
    def __init__(self, root, files_list, download=True):
        # Descargir el dataset completo si no existe
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
            relative_path = relative_path.replace("\\", "/")
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
        
        # Si no se encuentran archivos, mostrar algunos ejemplos para debug
        if len(self.indices) == 0:
            print("\n=== DEBUG: Comparando paths ===")
            print("Ejemplos de paths en el archivo .txt:")
            for i in range(min(3, len(self.file_paths))):
                print(f"  {self.file_paths[i]}")
            print("Ejemplos de paths en el dataset:")
            for i in range(min(3, len(self.all_paths))):
                print(f"  {self.all_paths[i]}")
            print("¿Coinciden algunos paths?")
            matching = set(self.file_paths) & set(self.all_paths)
            print(f"Coincidencias encontradas: {len(matching)}")
            if matching:
                print(f"Ejemplos de coincidencias: {list(matching)[:3]}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Obtener el índice original en el dataset completo
        original_idx = self.indices[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[original_idx]
        return waveform, sample_rate, label, speaker_id, utterance_number

# Configuración
ROOT_DIR = "./speech_commands"
TRAIN_LIST = "./data/train_list.txt"
VAL_LIST = "./data/val_list.txt"
TEST_LIST = "./data/test_list.txt"

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
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()