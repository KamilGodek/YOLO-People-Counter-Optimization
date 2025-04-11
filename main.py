
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Konfiguracja
models_paths = ['YOLO_wagi/yolo11n.pt', 'YOLO_wagi/yolo11s.pt', 'YOLO_wagi/yolo11m.pt', 'YOLO_wagi/yolo11l.pt', 'YOLO_wagi/yolo11x.pt']
input_folder = 'input_images'
output_root = 'results_tuned'
actual_people_counts = [15, 13, 31 , 29 , 15 , 10, 9 ,9 ,9 , 47 ,13, 44, 12 ,12 ,15 ,16, 16, 19, 88, 75]

# Parametry do iteracji
conf_thresholds = [0.06,0.07,0.08, 0.09, 0.13, 0.14, 0.15]
iou_thresholds = [0.26, 0.27, 0.28,0.29,0.3,0.31,0.32,0.33]

# Ustawienia przetwarzania
n_runs = 1 # Ustawiono na 1 dla szybszej iteracji parametrów
device = '0' if torch.cuda.is_available() else 'cpu'
inference_size = 3200 # Rozważ zmniejszenie dla szybszej iteracji
save_annotated_images = True # Czy zapisywać obrazy z detekcjami dla najlepszych parametrów

# --- Inicjalizacja ---
best_params_results = {
    'processing_times': {},
    'people_counts': {}
}
best_params_per_model = {}

# Tworzenie folderów i wczytywanie obrazów
os.makedirs(output_root, exist_ok=True)
try:
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if not image_files:
        raise ValueError(f"Brak plików obrazów w folderze: {input_folder}")
    num_actual_images = len(image_files) - 1
    if len(actual_people_counts) != num_actual_images:
        print(f"Ostrzeżenie: Liczba rzeczywistych zliczeń ({len(actual_people_counts)}) "
              f"nie pasuje do liczby obrazów do przetworzenia ({num_actual_images}). "
              f"Przycięto listę actual_people_counts do {num_actual_images} elementów.")
        actual_people_counts = actual_people_counts[:num_actual_images]

except FileNotFoundError:
    print(f"Błąd krytyczny: Folder wejściowy '{input_folder}' nie istnieje.")
    exit()
except ValueError as e:
    print(f"Błąd krytyczny: {e}")
    exit()

print(f"Rozpoczynanie wyszukiwania najlepszych parametrów (conf, iou) dla {len(models_paths)} modeli...")
print(f"Testowane wartości conf: {conf_thresholds}")
print(f"Testowane wartości iou: {iou_thresholds}")
print(f"Urządzenie: {device}")
print("-" * 30)

# Pętla wyszukiwania najlepszych parametrów
for model_path in models_paths:
    model_name_base = os.path.splitext(os.path.basename(model_path))[0]
    print(f"Przetwarzanie modelu: {model_name_base}")
    try:
        model = YOLO(model_path)
        print(f"  Model {model_name_base} załadowany.")
    except Exception as e:
        print(f"  Błąd ładowania modelu {model_path}: {e}. Pomijanie.")
        continue

    best_mae = float('inf')
    best_conf = None
    best_iou = None
    best_run_counts_for_model = None

    # Iteracja po parametrach
    for conf_thresh in conf_thresholds:
        for iou_thresh in iou_thresholds:
            print(f"  Testowanie: conf={conf_thresh}, iou={iou_thresh}...", end="")

            current_run_counts = []
            valid_run = True
            try:

                if image_files:
                    warmup_img_path = os.path.join(input_folder, image_files[0])
                    if os.path.exists(warmup_img_path):
                        frame_warmup = cv2.imread(warmup_img_path)
                        frame_warmup_resized = cv2.resize(frame_warmup, (640, 480))
                        _ = model(frame_warmup_resized, device=device, conf=conf_thresh, iou=iou_thresh, imgsz=inference_size, verbose=False, classes=[0])
                    # else: print("  Ostrzeżenie: Brak pliku rozgrzewkowego.") # Mniej istotne ostrzeżenie

                # Przetwarzanie obrazów (pomijając obraz 0)
                img_indices_to_process = range(1, len(image_files))
                if len(img_indices_to_process) != len(actual_people_counts):
                     print(f"\n    Ostrzeżenie: Niezgodność liczby obrazów ({len(img_indices_to_process)}) i liczby rzeczywistych zliczeń ({len(actual_people_counts)}). Sprawdź pliki i listę.")
                     valid_run = False
                     break

                for idx in img_indices_to_process:
                    img_file = image_files[idx]
                    img_path = os.path.join(input_folder, img_file)
                    if not os.path.exists(img_path):
                        print(f"\n    Ostrzeżenie: Plik obrazu {img_file} nie istnieje. Pomijanie.")
                        continue

                    frame = cv2.imread(img_path)
                    frame_resized = cv2.resize(frame, (960, 540))
                    detections = model(frame_resized, device=device, conf=conf_thresh, iou=iou_thresh, imgsz=inference_size, verbose=False, classes=[0])[0].boxes
                    run_count = sum(1 for box in detections if int(box.cls.item()) == 0)
                    current_run_counts.append(run_count)

            except Exception as e:
                print(f"\n    Błąd podczas przetwarzania z conf={conf_thresh}, iou={iou_thresh}: {e}")
                valid_run = False

            if not valid_run:
                print(" Przerwano.")
                continue # Przejdź do następnej kombinacji iou

            # Sprawdzenie, czy liczba wyników zgadza się z oczekiwaną
            if len(current_run_counts) != len(actual_people_counts):
                print(f" Niekompletne wyniki ({len(current_run_counts)}/{len(actual_people_counts)}).")
                continue # Przejdź do następnej kombinacji iou

            # Obliczanie błędu (MAE)
            mae = mean_absolute_error(actual_people_counts, current_run_counts)
            print(f" MAE={mae:.4f}", end="")

            # Aktualizacja najlepszych parametrów
            if mae < best_mae:
                best_mae = mae
                best_conf = conf_thresh
                best_iou = iou_thresh
                best_run_counts_for_model = current_run_counts
                print(" *Nowe najlepsze*")
            else:
                print("")

    # Zapisz najlepsze parametry dla bieżącego modelu
    if best_conf is not None and best_iou is not None:
        best_params_per_model[model_name_base] = {'conf': best_conf, 'iou': best_iou, 'mae': best_mae}
        print(f"\n=> Najlepsze parametry dla '{model_name_base}': conf={best_conf}, iou={best_iou} (osiągnięte MAE: {best_mae:.4f})\n")
    else:
        print(f"\n=> Nie udało się znaleźć optymalnych parametrów dla '{model_name_base}'.\n")
    print("-" * 30)

# Sprawdzenie, czy znaleziono jakiekolwiek parametry
if not best_params_per_model:
    print("Nie znaleziono optymalnych parametrów dla żadnego modelu. Zakończenie skryptu.")
    exit()

print("Zakończono wyszukiwanie.")
print("Rozpoczynanie przetwarzania końcowego z najlepszymi parametrami...")
print("-" * 30)

# Przetwarzanie końcowe z najlepszymi parametrami
for model_path in models_paths:
    model_name_base = os.path.splitext(os.path.basename(model_path))[0]

    if model_name_base not in best_params_per_model:
        print(f"Pominięto przetwarzanie końcowe dla '{model_name_base}' - brak optymalnych parametrów.")
        continue

    best_params = best_params_per_model[model_name_base]
    best_conf = best_params['conf']
    best_iou = best_params['iou']
    print(f"Przetwarzanie końcowe dla '{model_name_base}' (conf={best_conf}, iou={best_iou})")

    try:
        model = YOLO(model_path) # Ponowne załadowanie modelu
    except Exception as e:
        print(f"  Błąd ponownego ładowania modelu {model_path}: {e}. Pomijanie.")
        continue

    output_folder_tuned = os.path.join(output_root, model_name_base + "_best_results")
    os.makedirs(output_folder_tuned, exist_ok=True)

    final_processing_times = []
    final_people_counts = []
    try:

        if image_files:
            warmup_img_path = os.path.join(input_folder, image_files[0])
            if os.path.exists(warmup_img_path):
                frame_warmup = cv2.imread(warmup_img_path)
                frame_warmup_resized = cv2.resize(frame_warmup, (640, 480))
                _ = model(frame_warmup_resized, device=device, conf=best_conf, iou=best_iou, imgsz=inference_size, verbose=False, classes=[0])

        # Przetwarzanie obrazów
        for idx in range(1, len(image_files)): # Iteruj od 1 do końca
             img_file = image_files[idx]
             img_path = os.path.join(input_folder, img_file)
             if not os.path.exists(img_path): continue

             frame = cv2.imread(img_path)
             frame_resized = cv2.resize(frame, (960, 540))
             start_time = time.time()
             detections = model(frame_resized, device=device, conf=best_conf, iou=best_iou, imgsz=inference_size, verbose=False, classes=[0])[0].boxes
             processing_time = time.time() - start_time
             people_count = sum(1 for box in detections if int(box.cls.item()) == 0)

             final_processing_times.append(processing_time)
             final_people_counts.append(people_count)

             # Zapis obrazów z adnotacjami
             if save_annotated_images:
                 annotated_frame = frame_resized.copy()
                 for box in detections:
                     if int(box.cls.item()) == 0:
                         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                         conf_score = box.conf.item()
                         label = f'Osoba {conf_score:.2f}' # Etykieta po polsku
                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                         cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                 output_filename = f"{os.path.splitext(img_file)[0]}_annotated.jpg"
                 output_path = os.path.join(output_folder_tuned, output_filename)
                 cv2.imwrite(output_path, annotated_frame)

        # Sprawdzenie, czy przetworzono oczekiwaną liczbę obrazów
        if len(final_processing_times) == num_actual_images and len(final_people_counts) == num_actual_images:
             best_params_results['processing_times'][model_name_base] = final_processing_times
             best_params_results['people_counts'][model_name_base] = final_people_counts
             print(f"  Zakończono przetwarzanie końcowe dla '{model_name_base}'. Wyniki zapisane.")
        else:
             print(f"  Ostrzeżenie: Niezgodność liczby przetworzonych obrazów ({len(final_processing_times)}) z oczekiwaną ({num_actual_images}) dla '{model_name_base}'.")

    except Exception as e:
        print(f"  Błąd podczas przetwarzania końcowego dla '{model_name_base}': {e}")

# Dodaj rzeczywiste wartości do wyników
best_params_results['people_counts']['Rzeczywista'] = actual_people_counts

# Zapis do CSV dla najlepszych parametrów
if not best_params_results['processing_times']:
     print("\nBrak wyników z przetwarzania końcowego do zapisania w CSV i na wykresach.")
else:
    output_csv_path = os.path.join(output_root, 'final_reports_csv')
    os.makedirs(output_csv_path, exist_ok=True)
    print(f"\nZapisywanie raportów CSV do: {output_csv_path}")

    image_numbers = [str(i + 1) for i in range(num_actual_images)]
    models_list_best = list(best_params_results['processing_times'].keys())

    try:
        with open(os.path.join(output_csv_path,'processing_times_best.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model'] + image_numbers)
            for model_name in models_list_best:
                if model_name in best_params_results['processing_times'] and len(best_params_results['processing_times'][model_name]) == len(image_numbers):
                    row = [model_name] + [f"{t:.4f}" for t in best_params_results['processing_times'][model_name]]
                    writer.writerow(row)
    except IOError as e: print(f"Błąd zapisu processing_times_best.csv: {e}")

    try:
        with open(os.path.join(output_csv_path,'people_counts_best.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            people_counts_models = list(best_params_results['people_counts'].keys())
            writer.writerow(['Model'] + image_numbers)
            for model_name in people_counts_models:
                 if model_name in best_params_results['people_counts'] and len(best_params_results['people_counts'][model_name]) == len(image_numbers):
                    row = [model_name] + [str(int(c)) for c in best_params_results['people_counts'][model_name]]
                    writer.writerow(row)
    except IOError as e: print(f"Błąd zapisu people_counts_best.csv: {e}")

    try:
        with open(os.path.join(output_csv_path,'summary_best.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Najlepszy conf', 'Najlepszy iou', 'Finalne MAE', 'Całkowity czas [s]', 'Całkowita liczba wykrytych osób'])
            for model_name in models_list_best:
                 if model_name in best_params_per_model:
                    params = best_params_per_model[model_name]
                    total_time = np.sum(best_params_results['processing_times'].get(model_name, []))
                    total_people = np.sum(best_params_results['people_counts'].get(model_name, []))
                    writer.writerow([model_name, params['conf'], params['iou'], f"{params['mae']:.4f}", f"{total_time:.2f}", int(total_people)])
    except IOError as e: print(f"Błąd zapisu summary_best.csv: {e}")


    # Generowanie wykresów dla najlepszych parametrów
    print("\nGenerowanie wykresów z najlepszymi parametrami...")
    output_plots_path = os.path.join(output_root, 'final_plots')
    os.makedirs(output_plots_path, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid') # Użyj stylu z siatką dla lepszej czytelności
    # Użyj tab10 dla bardziej rozróżnialnych kolorów
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_list_best) + 1))
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', # Użyj czcionki obsługującej polskie znaki
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'figure.figsize': (18, 14),
        'figure.dpi': 100 # DPI dla szybszego podglądu, można zwiększyć do 300 dla finałowej wersji
    })

    #  Wykres główny
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) # Wspólna oś X
    n_images = num_actual_images
    x = np.arange(n_images)
    num_models_plot = len(models_list_best)
    total_bar_width = 0.85
    bar_width = total_bar_width / num_models_plot if num_models_plot > 0 else total_bar_width
    spacing = 0.01

    # Wykres czasów przetwarzania
    for i, (model_name, color) in enumerate(zip(models_list_best, colors)):
        offset = x + (i - num_models_plot / 2 + 0.5) * bar_width
        if model_name in best_params_results['processing_times']:
             times = best_params_results['processing_times'][model_name]
             if len(times) == n_images:
                 label_text = f"{model_name}\n(conf={best_params_per_model[model_name]['conf']}, iou={best_params_per_model[model_name]['iou']})"
                 bars = ax1.bar(offset, times, width=bar_width - spacing, label=label_text, color=color, edgecolor='black', linewidth=0.5)
             # else: print(f"Ostrzeżenie (Plot Time): Niezgodność danych dla {model_name}")

    ax1.set_title('Czas przetwarzania obrazu dla modeli z optymalnymi parametrami', pad=15) # Zmieniony tytuł
    ax1.set_ylabel('Czas [s]')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # Legenda z prawej strony, poza obszarem wykresu
    ax1.legend(title="Modele i użyte parametry", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title_fontsize=10)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False) # Ukryj znaczniki osi X na górnym wykresie


    # Wykres liczby osób
    all_models_people = list(best_params_results['people_counts'].keys()) # Zawiera 'Rzeczywista'
    num_bars_people = len(all_models_people)
    bar_width_people = total_bar_width / num_bars_people if num_bars_people > 0 else total_bar_width

    max_people = 0
    for model_name in all_models_people:
        if model_name in best_params_results['people_counts']:
             counts = best_params_results['people_counts'][model_name]
             if len(counts) == n_images:
                 current_max = max(counts) if counts else 0
                 if current_max > max_people: max_people = current_max

    # Ustawienie kolorów, wyróżnienie 'Rzeczywista'
    colors_people_map = {name: color for name, color in zip(models_list_best, colors)}
    colors_people_map['Rzeczywista'] = 'red' # Wyróżnienie czerwonym

    for i, model_name in enumerate(all_models_people):
        offset = x + (i - num_bars_people / 2 + 0.5) * bar_width_people
        if model_name in best_params_results['people_counts']:
            counts = best_params_results['people_counts'][model_name]
            if len(counts) == n_images:
                label_text = model_name if model_name == 'Rzeczywista' else f"{model_name} (MAE: {best_params_per_model.get(model_name, {}).get('mae', float('nan')):.2f})"
                color_to_use = colors_people_map.get(model_name, colors[i % len(colors)]) # Użyj mapy lub kolorów cyklicznie
                bars = ax2.bar(offset, counts, width=bar_width_people - spacing, label=label_text, color=color_to_use, edgecolor='black', linewidth=0.5)
                # Dodawanie etykiet na słupkach (tylko jeśli jest miejsce)
                if bar_width_people > 0.05:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{int(height)}', ha='center', va='bottom', rotation=90, fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Obraz {i + 1}' for i in range(n_images)], rotation=45, ha='right')
    ax2.set_xlabel('Numer obrazu wejściowego') # Dodano etykietę osi X
    ax2.set_yticks(np.arange(0, max_people + 10, 10 if max_people > 50 else 5))
    ax2.set_ylim(0, max_people + (max_people * 0.1) + 5) # Margines + stała
    ax2.set_title('Porównanie liczby wykrytych osób z wartością rzeczywistą (optymalne parametry)', pad=15) # Zmieniony tytuł
    ax2.set_ylabel('Liczba osób')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend(title="Modele (z MAE) i wartość rzeczywista", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title_fontsize=10) # Poprawiona legenda

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plot_filename = os.path.join(output_plots_path, 'comparison_best_params.png')
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres porównawczy: {plot_filename}")
    except Exception as e: print(f"Błąd zapisu wykresu porównawczego: {e}")
    plt.close(fig)


    # Wykres podsumowujący
    fig_summary, (ax_sum1, ax_sum2) = plt.subplots(1, 2, figsize=(17, 7)) # Nieco szerszy

    valid_models_summary = [model for model in models_list_best if model in best_params_results['processing_times'] and model in best_params_results['people_counts']]
    total_time_best = [np.sum(best_params_results['processing_times'][model]) for model in valid_models_summary]
    total_people_best = [np.sum(best_params_results['people_counts'][model]) for model in valid_models_summary]
    model_labels_summary = [f"{name}\n(MAE: {best_params_per_model[name]['mae']:.2f})" for name in valid_models_summary]

    # Podsumowanie czasów
    if total_time_best:
        bars_time = ax_sum1.bar(model_labels_summary, total_time_best, color=colors[:len(valid_models_summary)], edgecolor='black')
        ax_sum1.set_title('Łączny czas przetwarzania (optymalne parametry)', pad=15) # Poprawiony tytuł
        ax_sum1.set_ylabel('Całkowity czas [s]')
        ax_sum1.grid(axis='y', linestyle='--', alpha=0.7)
        ax_sum1.tick_params(axis='x', rotation=45, labelsize=9) # Obrót etykiet modeli

        for bar in bars_time:
            height = bar.get_height()
            ax_sum1.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    else:
        ax_sum1.text(0.5, 0.5, 'Brak danych czasowych', ha='center', va='center', fontsize=12)
        ax_sum1.set_title('Łączny czas przetwarzania (optymalne parametry)', pad=15)

    # Podsumowanie osób
    total_people_actual = np.sum(actual_people_counts)
    if total_people_best:
        bars_people = ax_sum2.bar(model_labels_summary, total_people_best, color=colors[:len(valid_models_summary)], edgecolor='black')
        ax_sum2.set_title('Łączna liczba wykrytych osób vs Rzeczywista (optymalne parametry)', pad=15) # Poprawiony tytuł
        ax_sum2.set_ylabel('Całkowita liczba osób')
        ax_sum2.grid(axis='y', linestyle='--', alpha=0.7)
        ax_sum2.tick_params(axis='x', rotation=45, labelsize=9) # Obrót etykiet modeli

        # Dodaj linię dla rzeczywistej sumy
        ax_sum2.axhline(total_people_actual, color='red', linestyle='--', linewidth=2, label=f'Rzeczywista suma: {int(total_people_actual)}')
        ax_sum2.legend(fontsize=10)

        # Ustaw granice osi Y, aby linia rzeczywista była dobrze widoczna
        max_y_limit_people = max(max(total_people_best) if total_people_best else 0, total_people_actual) * 1.1 + 5
        min_y_limit_people = 0 # min(min(total_people_best) if total_people_best else 0, total_people_actual) * 0.9 - 5
        ax_sum2.set_ylim(min_y_limit_people, max_y_limit_people)

        for bar in bars_people:
            height = bar.get_height()
            ax_sum2.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    else:
         ax_sum2.text(0.5, 0.5, 'Brak danych o liczbie osób', ha='center', va='center', fontsize=12)
         ax_sum2.set_title('Łączna liczba wykrytych osób vs Rzeczywista (optymalne parametry)', pad=15)


    plt.tight_layout(pad=2.0)
    summary_plot_filename = os.path.join(output_plots_path, 'summary_best_params.png')
    try:
        plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight') # Zwiększono DPI
        print(f"Zapisano wykres podsumowujący: {summary_plot_filename}")
    except Exception as e: print(f"Błąd zapisu wykresu podsumowującego: {e}")
    plt.close(fig_summary)

    # Końcowe podsumowanie
    print("-" * 30)
    print("Przetwarzanie zakończone.")
    print(f"Wyniki zapisano w folderze: {output_root}")
    print(f"  Raporty CSV: {output_csv_path}")
    print(f"  Wykresy: {output_plots_path}")
    if save_annotated_images: print(f"  Obrazy z adnotacjami w podfolderach *_best_results")

    print("\n--- Podsumowanie najlepszych znalezionych parametrów ---")
    if best_params_per_model:
        for model_name, params in best_params_per_model.items():
            print(f"- Model: {model_name}")
            print(f"  - Optymalny conf: {params['conf']}")
            print(f"  - Optymalny iou: {params['iou']}")
            print(f"  - Osiągnięte MAE (Mean Absolute Error): {params['mae']:.4f}")
    else: # Ten warunek jest już sprawdzany wcześniej, ale dla pewności
        print("Nie znaleziono optymalnych parametrów dla żadnego modelu.")