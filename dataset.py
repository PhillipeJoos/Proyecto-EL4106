import os
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

class CustomSpeechCommands(Dataset):
    def __init__(self, root, files_list, download=False): # cambiar esto a true si no se tiene el dataset
        # Descargar el dataset completo si no existe
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, 
            download=download
        ) 
        self.indices = None
        self.splitter(files_list, root)

    def splitter(self, files_list, root):
        with open(files_list, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        
        print(f"Primeros 3 paths en {files_list}: {self.file_paths[:3]}")
        self.all_paths = []
        for item in self.dataset._walker:

            full_path = item
            relative_path = os.path.relpath(full_path, start=os.path.join(root, "SpeechCommands", "speech_commands_v0.02"))
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
    
    def extract_features(self, feature_extractor, device="cpu"):
        """
        Extrae características MFCC (u otras) del subset definido por self.indices.
        Devuelve dos tensores: features (N x C x T) y labels (lista de strings).
        """
        features = []
        labels = []

        # Pasar el extractor al dispositivo si es necesario
        feature_extractor.to(device)

        with torch.no_grad():
            for idx in tqdm(self.indices, desc="Extrayendo features"):
                waveform, sample_rate, label, _, _ = self.dataset[idx]
                waveform = waveform.to(device)

                feat = feature_extractor(waveform).squeeze(0).cpu()
                features.append(feat)
                labels.append(label)

        # Convertir la lista de features a tensor si tienen igual longitud temporal
        try:
            features = torch.stack(features)
        except RuntimeError:
            print("Advertencia: las longitudes temporales varían, manteniendo lista de tensores.")
        
        return features, labels
    def save_features(self, path):
        pass

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Obtener el índice original en el dataset completo
        original_idx = self.indices[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[original_idx]
        return waveform, sample_rate, label, speaker_id, utterance_number
