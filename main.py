import torch
import os

def analyze_shapes(filepath):
    print(f"\n{'='*50}")
    print(f"📐 Analizando Arquitectura: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print(f"[ERROR] Archivo no encontrado.")
        return

    try:
        state_dict = torch.load(filepath, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if 'config' in state_dict:
            config = state_dict['config']
        
        # 1. Buscar la primera capa recurrente (input-to-hidden)
        # PyTorch suele nombrarlas como 'rnn_layer.weight_ih_l0' o similar
        input_size = None
        hidden_size = None
        layer_type = "Desconocido"
        
        for key, tensor in state_dict.items():
            # Detectar capa de entrada RNN/GRU/LSTM
            if 'weight_ih_l0' in key:
                shape = tensor.shape
                print(f"-> Capa de Entrada detectada ({key}): {shape}")
                
                # La forma es (hidden_size * num_gates, input_size)
                input_size = shape[1] 
                total_hidden = shape[0]
                
                # Deducir si es GRU (3 gates) o LSTM (4 gates) o RNN (1 gate)
                # Asumimos GRU por tu nombre de archivo, pero verificamos
                if total_hidden % 3 == 0:
                    hidden_size = total_hidden // 3
                    layer_type = "GRU (3 gates)"
                elif total_hidden % 4 == 0:
                    hidden_size = total_hidden // 4
                    layer_type = "LSTM (4 gates)"
                else:
                    hidden_size = total_hidden
                    layer_type = "RNN (1 gate)"

            # Detectar capa de salida (Linear)
            # Suele ser 'net.0.weight', 'fc.weight', 'classifier.weight'
            elif 'net.0.weight' in key or 'fc.weight' in key:
                print(f"-> Capa de Salida detectada ({key}): {tensor.shape}")
                num_classes = tensor.shape[0]
                last_layer_input = tensor.shape[1]
                print(f"   * Clases de salida: {num_classes}")

        print("-" * 30)
        if input_size and hidden_size:
            print(f"✅ DEDUCCIÓN DE ARQUITECTURA:")
            print(f"   - Tipo: {layer_type}")
            print(f"   - Input Features (entrada): {input_size}")
            print(f"   - Hidden Size (oculto):     {hidden_size}")
        else:
            print("⚠️ No se pudo deducir la arquitectura completa automáticamente.")
            print("   (Revisa los nombres de las capas impresos arriba)")

    except Exception as e:
        print(f"❌ Error al leer el archivo (¿Posible corrupción?): {e}")

# --- EJECUTAR ---
# file_1 = os.path.join('model_weights',"GRU_2025-11-21_02-49-31.pt")
# file_2 = os.path.join('SUS_MODELS_WEIGHTS',"GRU_2025-11-21_02-44-43.pt")
# files = [file_1, file_2]

for weight_dir in ['model_weights', 'old_model_weights', 'SUS_MODELS_WEIGHTS']:
    print(f"{weight_dir:^.10}")
    for f in os.listdir(weight_dir):
        filename = os.path.join(weight_dir, f)
        if os.path.isdir(filename):
            continue
        analyze_shapes(filename) 
