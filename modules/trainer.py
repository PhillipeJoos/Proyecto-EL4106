import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os
import re

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

    # === Definir labels de clases ===
    # Si el dataset tiene atributo label_to_idx o idx_to_label, lo usamos
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

    # === Plot ===
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_mean, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # === Texto: mean ± std ===
    fmt = lambda m, s: f"{m:.2f}\n±{s:.2f}"
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, fmt(cm_mean[i, j], cm_std[i, j]),
                    ha="center", va="center",
                    color="white" if cm_mean[i, j] > thresh else "black")

    # === Accuracy promedio ===
    accs = []
    for model in models:
        y_pred = []
        for x, _ in dataloader:
            y_pred.append(model(x).argmax(dim=1))
        y_pred = torch.cat(y_pred)
        accs.append(accuracy_score(y_true, y_pred))

    ax.set_title(rf'{name}, mean acc = {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%')
    plt.tight_layout()

    os.makedirs('img', exist_ok=True)
    filepath = os.path.join('img', f'conf_mat_{name}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix saved to {filepath}")

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

def nfft_hop_length_exp(n_trains, feature_xtractor):
    
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
                plt.text(j, i, f"{mean_val:.2f}\n+/-{std_val:.4f}", 
                        ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    plt.show()