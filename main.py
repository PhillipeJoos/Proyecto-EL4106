# %%
import os
import torchaudio
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from random import randint

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os
import re



# %% [markdown]
# ## Datasets

# %%

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
        # self.OFFICIAL_CLASSES = [
        # "yes", "no", "up", "down", "left", "right",
        # "on", "off", "stop", "go"
        # ]

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
                feat = feat.transpose(0, 1)  # [T, n_mfcc] -> ahora input_size=13
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
            # processed_labels = [
            #     label if label in self.OFFICIAL_CLASSES else "unknown"
            #     for label in labels
            # ]
            torch.save({"features": features, "labels": labels}, save_path)
            print(f"Features guardadas correctamente en {save_path}")
            print(f"Clases finales: {set(labels)}")

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

# %% [markdown]
# ## Models

# %%
class RNNModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        n_input_channels,
        hidd_size=256,
        out_features = 35,
        num_layers=1,
    ):
        """
        Para utilizar una vanilla RNN entregue rnn_type="RNN"
        Para utilizar una LSTM entregue rnn_type="LSTM"
        Para utilizar una GRU entregue rnn_type="GRU"
        """
        super().__init__()

        self.rnn_type = rnn_type

        if rnn_type == "GRU":
            self.rnn_layer = nn.GRU(n_input_channels, hidd_size, batch_first=True, num_layers=num_layers)

        elif rnn_type == "LSTM":
            self.rnn_layer = nn.LSTM(n_input_channels, hidd_size, batch_first=True, num_layers=num_layers)

        elif rnn_type == "RNN":
            self.rnn_layer = nn.RNN(n_input_channels, hidd_size, batch_first=True, num_layers=num_layers, bidirectional=True)

        else:
            raise ValueError(f"rnn_type {rnn_type} not supported.")

        self.net = nn.Sequential(
            nn.Linear(hidd_size, out_features),
        )

        self.flatten_layer = nn.Flatten()

    def forward(self, x):
        if self.rnn_type == "GRU":
            out, h = self.rnn_layer(x)

        elif self.rnn_type == "LSTM":
            out, (h, c) = self.rnn_layer(x)

        elif self.rnn_type == "RNN":
            out, h = self.rnn_layer(x)

        out = h[-1]

        return self.net(out)

# %% [markdown]
# ## Trainers

# %%
def train_step(x_batch, y_batch, model, optimizer, criterion, use_gpu):
    # Predicción
    y_predicted = model(x_batch)

    # Cálculo de loss
    loss = criterion(y_predicted, y_batch)

    # Actualización de parámetros
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y_predicted, loss


def evaluate(val_loader, model, criterion, use_gpu):
    cumulative_loss = 0
    cumulative_predictions = 0
    data_count = 0

    for x_val, y_val in val_loader:
        if use_gpu:
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        y_predicted = model(x_val)

        loss = criterion(y_predicted, y_val)

        class_prediction = torch.argmax(y_predicted, axis=1).long()

        cumulative_predictions += (y_val == class_prediction).sum().item()
        cumulative_loss += loss.item() * y_val.shape[0]
        data_count += y_val.shape[0]

    val_acc = cumulative_predictions / data_count
    val_loss = cumulative_loss / data_count

    return val_acc, val_loss


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    criterion,
    batch_size,
    lr,
    n_evaluations_per_epoch=6,
    use_gpu=False,
):
    if use_gpu:
        model.cuda()

    # Definición de dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_gpu)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_gpu)

    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Listas para guardar curvas de entrenamiento
    curves = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    t0 = time.perf_counter()

    iteration = 0

    n_batches = len(train_loader)
    print(n_batches)

    for epoch in range(epochs):
        print(f"\rEpoch {epoch + 1}/{epochs}")
        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        examples_count = 0

        # Entrenamiento del modelo
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            if use_gpu:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_predicted, loss = train_step(x_batch, y_batch, model, optimizer, criterion, use_gpu)

            cumulative_train_loss += loss.item() * x_batch.shape[0]
            examples_count += y_batch.shape[0]

            # Calculamos número de aciertos
            class_prediction = torch.argmax(y_predicted, axis=1).long()
            cumulative_train_corrects += (y_batch == class_prediction).sum().item()

            if (i % (n_batches // n_evaluations_per_epoch) == 0) and (i > 0):
                train_loss = cumulative_train_loss / examples_count
                train_acc = cumulative_train_corrects / examples_count

                print(f"Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss}, Train acc: {train_acc}")

            iteration += 1

        model.eval()
        with torch.no_grad():
            val_acc, val_loss = evaluate(val_loader, model, criterion, use_gpu)

        print(f"Val loss: {val_loss}, Val acc: {val_acc}")

        train_loss = cumulative_train_loss / examples_count
        train_acc = cumulative_train_corrects / examples_count

        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

    print()
    total_time = time.perf_counter() - t0
    print(f"Tiempo total de entrenamiento: {total_time:.4f} [s]")

    model.cpu()

    return curves, total_time

def show_curves(all_curves, suptitle=''):
    final_curve_means = {k: np.mean([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}
    final_curve_stds = {k: np.std([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.set_facecolor('white')

    epochs = np.arange(len(final_curve_means["val_loss"])) + 1

    # ==== Plot de pérdidas ====
    ax[0].plot(epochs, final_curve_means['val_loss'], label='validation')
    ax[0].plot(epochs, final_curve_means['train_loss'], label='training')
    ax[0].fill_between(epochs, 
                       y1=final_curve_means["val_loss"] - final_curve_stds["val_loss"], 
                       y2=final_curve_means["val_loss"] + final_curve_stds["val_loss"], alpha=.5)
    ax[0].fill_between(epochs, 
                       y1=final_curve_means["train_loss"] - final_curve_stds["train_loss"], 
                       y2=final_curve_means["train_loss"] + final_curve_stds["train_loss"], alpha=.5)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss evolution during training')
    ax[0].legend()

    # ==== Plot de precisión ====
    ax[1].plot(epochs, final_curve_means['val_acc'], label='validation')
    ax[1].plot(epochs, final_curve_means['train_acc'], label='training')
    ax[1].fill_between(epochs, 
                       y1=final_curve_means["val_acc"] - final_curve_stds["val_acc"], 
                       y2=final_curve_means["val_acc"] + final_curve_stds["val_acc"], alpha=.5)
    ax[1].fill_between(epochs, 
                       y1=final_curve_means["train_acc"] - final_curve_stds["train_acc"], 
                       y2=final_curve_means["train_acc"] + final_curve_stds["train_acc"], alpha=.5)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy evolution during training')
    ax[1].legend()

    fig.suptitle(suptitle, fontsize=16, weight="bold")

    # ==== Guardar y cerrar ====
    filepath = os.path.join('img', f'{suptitle}.pdf')
    plt.savefig(filepath, bbox_inches='tight', format='pdf')
    plt.close(fig)  

def get_metrics_and_confusion_matrix(models, dataset, name=''):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(16, len(dataset)))

    # === Obtener etiquetas verdaderas ===
    y_true = []
    for _, y in dataloader:
        y_true.append(y)
    y_true = torch.cat(y_true)
    n_classes = len(torch.unique(y_true))

    # === Definir labels ===
    if hasattr(dataset, 'idx_to_label'):
        labels = [dataset.idx_to_label[i] for i in range(n_classes)]
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        labels = [str(i) for i in range(n_classes)]

    # === Calcular matrices de confusión ===
    cms = []
    for model in models:
        model.cpu()
        model.eval()
        y_pred = []
        for x, _ in dataloader:
            y_pred.append(model(x).argmax(dim=1))
        y_pred = torch.cat(y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes), normalize='true')
        cms.append(cm)

    cms = np.stack(cms)
    cm_mean = cms.mean(axis=0)
    cm_std = cms.std(axis=0)

    # === Accuracy promedio ===
    accs = []
    for model in models:
        y_pred = []
        for x, _ in dataloader:
            y_pred.append(model(x).argmax(dim=1))
        y_pred = torch.cat(y_pred)
        accs.append(accuracy_score(y_true, y_pred))

    acc_mean = np.mean(accs) * 100
    acc_std = np.std(accs) * 100

    # === Figura combinada ===
    os.makedirs('img', exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: medias ---
    im1 = axs[0].imshow(cm_mean, interpolation='nearest', cmap=plt.cm.Blues)
    axs[0].set_title('Mean Confusion Matrix')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_ylabel('True label')
    axs[0].set_xticks(np.arange(n_classes))
    axs[0].set_yticks(np.arange(n_classes))
    axs[0].set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    axs[0].set_yticklabels(labels)
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    # --- Subplot 2: desviaciones estándar ---
    im2 = axs[1].imshow(cm_std, interpolation='nearest', cmap=plt.cm.Oranges)
    axs[1].set_title('Standard Deviation')
    axs[1].set_xlabel('Predicted label')
    axs[1].set_ylabel('True label')
    axs[1].set_xticks(np.arange(n_classes))
    axs[1].set_yticks(np.arange(n_classes))
    axs[1].set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    axs[1].set_yticklabels(labels)
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    # --- Título general ---
    fig.suptitle(rf'{name}, mean acc = {acc_mean:.2f} ± {acc_std:.2f}%', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = os.path.join('img', f'conf_mat_{name}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    print(f"Combined confusion matrix (mean + std) saved to {filepath}")

def evaluate_with_std(model, dataloader, criterion, use_gpu=True):
    # jaja std
    if use_gpu:
        model.cuda()

    all_losses = []
    all_accuracies = []

    with torch.no_grad():
        for X, y in dataloader:
            if use_gpu:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)

            outputs = model(X)
            loss = criterion(outputs, y)
            all_losses.append(loss.item())

            preds = outputs.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            all_accuracies.append(acc)

    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)

    model.cpu()

    return mean_acc, std_acc, mean_loss, std_loss



def evaluate_models_metrics(models, dataloader, criterion, use_gpu=True):
    """
    Evalúa múltiples modelos y calcula métricas promedio y desviación estándar.
    Retorna un diccionario con accuracy, recall, precision y f1
    """

    # Diccionarios para guardar resultados
    all_metrics = {
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1": [],
    }

    for model in models:
        model.eval()
        if use_gpu:
            model.cuda()

        y_true = []
        y_pred = []
        losses = []

        with torch.no_grad():
            for X, y in dataloader:
                if use_gpu:
                    X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)

                outputs = model(X)
                loss = criterion(outputs, y)
                losses.append(loss.item())

                preds = outputs.argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Cálculo de métricas
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        loss_mean = np.mean(losses)

        # Guardar métricas
        all_metrics["accuracy"].append(acc)
        all_metrics["recall"].append(rec)
        all_metrics["precision"].append(prec)
        all_metrics["f1"].append(f1)

        if use_gpu:
            model.cuda()
        else:
            model.cpu()

    # Calcular medias y desviaciones estándar
    metrics_mean = {k: np.mean(v) for k, v in all_metrics.items()}
    metrics_std = {k: np.std(v) for k, v in all_metrics.items()}

    print("\n=== Resultados promedio sobre modelos ===")
    for metric in all_metrics.keys():
        print(f"{metric.capitalize():<10}: {metrics_mean[metric]:.4f} +/- {metrics_std[metric]:.4f}")

    print("\n=== Detalles por modelo ===")
    for i, model in enumerate(models):
        print(f"\n=== Modelo {i + 1} ({model.rnn_type}) ===")
        for metric in all_metrics.keys():
            print(f"{metric.capitalize():<10}: {all_metrics[metric][i]:.4f} +/- {metrics_std[metric]:.4f}")

    # return metrics_mean, metrics_std, all_metrics
    return

def nfft_hop_length_exp(n_trains, feature_xtractor, batch_size, lr, epochs, criterion, use_gpu = True):
    
    # ======== Estructuras de resultados ========
    results = {}  # {(nfft, hl): [accuracies]}
    times_of_training = {}
    models = {}

    # ======== Obtener combinaciones de archivos ========
    base_dir = os.path.join('data', 'petes')  # o 'data' si están en esa carpeta
    files = os.listdir(base_dir)

    # Extraer parámetros nfft y hop_length de los nombres
    pattern = re.compile(r'n(\d+)_hl(\d+)')
    pairs = sorted(list({pattern.search(f).groups() for f in files if pattern.search(f)}))

    # ======== Loop sobre combinaciones ========
    for nfft, hl in pairs:
        nfft = int(nfft)
        hl = int(hl)
        print(f"\n=== Entrenando para nfft={nfft}, hop_length={hl} ===")

        # Cargar datasets
        train_dataset = feature_xtractor(os.path.join(base_dir, f'train_n{nfft}_hl{hl}.pt'))
        test_dataset = feature_xtractor(os.path.join(base_dir, f'test_n{nfft}_hl{hl}.pt'))
        val_dataset = feature_xtractor(os.path.join(base_dir, f'val_n{nfft}_hl{hl}.pt'))

        accs = []
        train_times = []

        for k in range(n_trains):
            print(f'  Entrenando modelo {k+1}/{n_trains}')
            model = RNNModel(rnn_type='RNN', n_input_channels=13, hidd_size=128)
            all_curves, times = train_model(
                model, train_dataset, val_dataset, epochs, criterion,
                batch_size, lr, n_evaluations_per_epoch=3, use_gpu=use_gpu
            )

            val_acc = all_curves["val_acc"][-1]  # o la métrica final que uses
            accs.append(val_acc)
            train_times.append(times)
            models[(nfft, hl, k)] = model

        results[(nfft, hl)] = accs
        times_of_training[(nfft, hl)] = train_times

    # ======== Graficar resultados ========
    # === Procesar los resultados ===
    nfft_vals = sorted(set(k[0] for k in results.keys()))
    hl_vals   = sorted(set(k[1] for k in results.keys()))

    # Crear matrices de promedio y desviación estándar
    mean_matrix = np.zeros((len(hl_vals), len(nfft_vals)))
    std_matrix  = np.zeros((len(hl_vals), len(nfft_vals)))

    for i, hl in enumerate(hl_vals):
        for j, nfft in enumerate(nfft_vals):
            if (nfft, hl) in results:
                vals = np.array(results[(nfft, hl)])
                mean_matrix[i, j] = np.mean(vals)
                std_matrix[i, j]  = np.std(vals)
            else:
                mean_matrix[i, j] = np.nan
                std_matrix[i, j]  = np.nan

    # === Graficar el mapa de calor ===
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mean_matrix, cmap='viridis', origin='lower', aspect='auto')

    # Etiquetas
    plt.xticks(range(len(nfft_vals)), nfft_vals)
    plt.yticks(range(len(hl_vals)), hl_vals)
    plt.xlabel('n_fft')
    plt.ylabel('hop_length')
    plt.title('Accuracy promedio ± desviación (5 entrenamientos por configuración)')

    # Barra de color
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy promedio')

    # Mostrar valores promedio ± std en cada celda
    for i in range(len(hl_vals)):
        for j in range(len(nfft_vals)):
            mean_val = mean_matrix[i, j]
            std_val  = std_matrix[i, j]
            if not np.isnan(mean_val):
                color = 'white' if mean_val < 0.7 else 'black'
                plt.text(j, i, f"+/-{std_val:.4f}", 
                        ha='center', va='center', color=color, fontsize=6)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Visualization

# %%
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


# %% [markdown]
# ## Feature extraction

# %%
# ==== Paths ====
ROOT_DIR = 'data'
train_pt = os.path.join(ROOT_DIR, 'train.pt')
val_pt = os.path.join(ROOT_DIR, 'val.pt')
test_pt = os.path.join(ROOT_DIR, 'test.pt')
TRAIN_LIST = os.path.join(ROOT_DIR,"train_list.txt")
VAL_LIST = os.path.join(ROOT_DIR, "val_list.txt")
TEST_LIST = os.path.join(ROOT_DIR, "test_list.txt")

if not os.path.isfile(train_pt):
    train_raw = CustomSpeechCommands(ROOT_DIR, TRAIN_LIST)
    val_raw = CustomSpeechCommands(ROOT_DIR, VAL_LIST)
    test_raw = CustomSpeechCommands(ROOT_DIR, TEST_LIST)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13, # número de coeficientes MFCC a extraer
        melkwargs={"n_fft": 320, "hop_length": 160, "n_mels": 23}, # 320 = 20ms, 160 = 10ms, 23 = número de filtros mel
        log_mels = True
    )
    train_raw.save_features(mfcc_transform, train_pt)
    test_raw.save_features(mfcc_transform, test_pt)
    val_raw.save_features(mfcc_transform, val_pt)

train_dataset = FeaturesDataset(train_pt)
test_dataset = FeaturesDataset(test_pt)
val_dataset = FeaturesDataset(val_pt)


print("¡Datasets cargados exitosamente!")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# %%

# random_idx = randint(0, len(test_raw))
# waveform, sample_rate, label, *_ = test_raw[random_idx]

# plot_waveform(waveform, sample_rate, label, figname=f'{label}_waveform_and_MFCC')


# %% [markdown]
# # Entrenamiento

# %%
print(train_dataset.features.shape)

# %%
# Preliminary testing
lr = 5e-4
batch_size = 32
criterion = nn.CrossEntropyLoss()
n_trains = 2
epochs = 2

for arch in ['GRU', 'LSTM', 'RNN']:
    print(f'Entrenando Modelo {arch}')
    times_of_training = []
    models = []
    curves = []
    for k in range(n_trains):
        print(f'Entrenando modelo {k}/{n_trains}')
        model = RNNModel(rnn_type = arch, n_input_channels=13) # puede ser que sea util estudiar el hidden size, o sea reducirlo hasta que comience a afectar el rendimiento del modelo en val
        all_curves, times = train_model(model, train_dataset, val_dataset, epochs, criterion, batch_size, lr, n_evaluations_per_epoch=3, use_gpu=True)
        curves.append(all_curves)
        times_of_training.append(times)
        models.append(model)
    show_curves(curves, arch)
    get_metrics_and_confusion_matrix(models, test_dataset, arch)


# %%



