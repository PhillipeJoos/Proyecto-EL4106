import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomSpeechCommands(Dataset):
    def __init__(self, root, files_list, download=False, target_len=16000):
        """
        root: directorio raíz del dataset
        files_list: archivo con lista de paths (train/val/test)
        download: True para descargar dataset si no existe
        target_len: duración fija en muestras (16000 = 1s)
        """
        self.target_len = target_len
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, 
            download=download
        )
        self.indices = None
        self.splitter(files_list, root)
        self.OFFICIAL_CLASSES = [
        "yes", "no", "up", "down", "left", "right",
        "on", "off", "stop", "go"
        ]

    def splitter(self, files_list, root):
        with open(files_list, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        
        self.all_paths = []
        for item in tqdm(self.dataset._walker, desc=f"Splitting {files_list}"):
            full_path = item
            relative_path = os.path.relpath(
                full_path, 
                start=os.path.join(root, "SpeechCommands", "speech_commands_v0.02")
            )
            relative_path = relative_path.replace("\\", "/")
            self.all_paths.append(relative_path)

        self.indices = [
            i for i, path in enumerate(self.all_paths) 
            if path in self.file_paths
        ]

        print(f"Total archivos en dataset: {len(self.all_paths)}")
        print(f"Archivos en {files_list}: {len(self.file_paths)}")
        print(f"Archivos encontrados: {len(self.indices)}")

    def pad_waveform(self, waveform):
        """
        Aplica zero padding (o recorte) para dejar todas las señales del mismo largo.
        """
        length = waveform.shape[-1]
        if length < self.target_len:
            pad_amt = self.target_len - length
            waveform = F.pad(waveform, (0, pad_amt))
        elif length > self.target_len:
            waveform = waveform[:, :self.target_len]
        return waveform

    def extract_features(self, feature_extractor, device="cpu"):
        """
        Extrae características (MFCC u otras) aplicando zero padding en el waveform.
        Devuelve:
          - features: tensor N x C x T
          - labels: lista de strings
        """
        features = []
        labels = []

        feature_extractor.to(device)

        with torch.no_grad():
            for idx in tqdm(self.indices, desc="Extrayendo features"):
                waveform, sample_rate, label, _, _ = self.dataset[idx]

                # padding antes del extractor
                waveform = self.pad_waveform(waveform).to(device)

                feat = feature_extractor(waveform).squeeze(0).cpu()
                feat = feat.transpose(0, 1)  # [T, n_mfcc] → ahora input_size=13
                features.append(feat)
                labels.append(label)

        # Convertir a tensor (todas las secuencias tienen igual longitud ahora)
        features = torch.stack(features)
        print(f"Features tensor: {features.shape}")  # [N, n_mfcc, T]
        return features, labels


    def save_features(self, feature_extractor, save_path, device="cpu"):
        """
        Extrae y guarda features, reemplazando clases no oficiales por 'unknown'.
        """
        print(f"Guardando features en: {save_path}")
        try:
            features, labels = self.extract_features(feature_extractor, device=device)
            processed_labels = [
                label if label in self.OFFICIAL_CLASSES else "unknown"
                for label in labels
            ]
            torch.save({"features": features, "labels": processed_labels}, save_path)
            print(f"Features guardadas correctamente en {save_path}")
            print(f"Clases finales: {set(processed_labels)}")

        except Exception as e:
            print(f"Error al guardar features en {save_path}: {e}")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[original_idx]
        waveform = self.pad_waveform(waveform)
        return waveform, sample_rate, label, speaker_id, utterance_number

class FeaturesDataset(Dataset):
    def __init__(self, features_path):
        """
        Carga un archivo .pt con 'features' y 'labels' previamente guardados.

        features_path: ruta al archivo .pt (por ejemplo 'data/train.pt')
        """
        data = torch.load(features_path)
        self.features = data["features"]
        self.labels = data["labels"]

        # Crear diccionario para pasar de string a índice (útil para entrenar)
        self.label_to_idx = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.numeric_labels = torch.tensor([self.label_to_idx[l] for l in self.labels])

        print(f"Dataset cargado desde {features_path}")
        print(f" - {len(self.features)} ejemplos")
        print(f" - {len(self.label_to_idx)} clases")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.numeric_labels[idx]
        return feature, label

def main():
    ROOT_DIR = "data"
    TRAIN_LIST = os.path.join(ROOT_DIR,"train_list.txt")
    VAL_LIST = os.path.join(ROOT_DIR, "val_list.txt")
    TEST_LIST = os.path.join(ROOT_DIR, "test_list.txt")
    train_pt = os.path.join(ROOT_DIR, 'train.pt')
    val_pt = os.path.join(ROOT_DIR, 'val.pt')
    test_pt = os.path.join(ROOT_DIR, 'test.pt')

    # Se define extractor de features
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13, # número de coeficientes MFCC a extraer
        melkwargs={"n_fft": 320, "hop_length": 160, "n_mels": 23} # 320 = 20ms, 160 = 10ms, 23 = número de filtros mel
    )

    # Crear datasets personalizados
    train_dataset = CustomSpeechCommands(ROOT_DIR, TRAIN_LIST)
    val_dataset = CustomSpeechCommands(ROOT_DIR, VAL_LIST)
    test_dataset = CustomSpeechCommands(ROOT_DIR, TEST_LIST)

    train_dataset.save_features(mfcc_transform, train_pt)
    test_dataset.save_features(mfcc_transform, test_pt)
    val_dataset.save_features(mfcc_transform, val_pt)

if __name__ == '__main__':
    main()
