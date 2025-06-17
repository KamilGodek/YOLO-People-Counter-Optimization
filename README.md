# YOLO-People-Counter-Optimization
Automated parameter tuning for YOLO models in people counting tasks. Features model comparison, CSV reports, visualizations and annotated images.



## 🎯 Features

- Grid search over confidence and IoU thresholds  
- Evaluation of 5 YOLO variants (nano, small, medium, large, xlarge)  
- MAE (Mean Absolute Error) validation against ground truth counts  
- CSV reports of per-image processing times and counts  
- PNG comparison plots  
- Export of annotated images for the best parameter set  


## 🛠️ Requirements
- OS: Windows/Linux/macOS
- Python 3.8+ ([download](https://www.python.org/downloads/))
- NVIDIA GPU (optional but recommended)
- 9GB+ free disk space

## 📥 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your_username/yolo-people-counter.git
cd yolo-people-counter
```

2. Create Virtual Environment
```bash
python -m venv venv
# Activation:
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📂 Data Preparation
Folder Structure
```bash

├── YOLO_wagi/           # YOLO model weights
│    ├── yolo11n.pt
│    ├── yolo11s.pt
│    ├── yolo11m.pt
│    ├── yolo11l.pt
│    └── yolo11x.pt
├── input_images1/       # Test images (0.jpg, 1.jpg, 2.jpg, …)
└── results_tuned/       # Output folder for tuned results and reports

```

Steps:

**1. Download YOLO Models from Ultralytics and place in YOLO_weights/:**

yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
and place them in YOLO_wagi/.

**2. Test images**

- Formats: .jpg, .png, .jpeg
- Name sequentially starting at 0.jpg (used for warm-up), then 1.jpg, 2.jpg, …

**3. Edit Ground Truth Values in script:**

In people_counter_optimization.py, line 7:
actual_people_counts = [15, 13, 31, ...]  # True counts for images 1.jpg, 2.jpg,...

## ⚙️ Configuration
Modify the Configuration section at the top of YOLOv11_evaluation.py:
```bash
    # Paths to model weight files
    models_paths = [
        'YOLO_wagi/yolo11n.pt',
        'YOLO_wagi/yolo11s.pt',
        'YOLO_wagi/yolo11m.pt',
        'YOLO_wagi/yolo11l.pt',
        'YOLO_wagi/yolo11x.pt'
    ]
    
    # Input and output directories
    input_folder = 'input_images1'
    output_root  = 'results_tuned'
    
    # Number of repetitions per parameter combo (set to 1 for speed)
    n_runs = 1
    
    # Device selection
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Inference image size (e.g., 640, 1280, 3200)
    inference_size = 3200
    
    # Save annotated images for best parameters?
    save_annotated_images = True
    
    # Confidence thresholds to search
    conf_thresholds = [0.06, 0.07, 0.08, 0.09, 0.13, 0.14, 0.15]
    
    # IoU thresholds to search
    iou_thresholds  = [0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33]
```
Adjust these values and lists to suit your dataset size and desired parameter ranges.

##🚀 Usage
    ```bash
    python people_counter_optimization.py
    ```
    Workflow:
Warm up on 0.jpg
Test every (confidence, IoU) pair
Compute MAE against actual_people_counts
Record the best thresholds per model
Re-run inference with best parameters
Save CSV reports and PNG plots
(Optional) Save annotated images in results_tuned/<model>_best_results/

## 📊 Outputs
   
In results_tuned/ you will find:
- final_reports_csv/
    - processing_times_best.csv — per-image inference times
    - people_counts_best.csv — per-image detection counts
    - summary_best.csv — best thresholds, MAE, total time & counts
- final_plots/
    - comparison_best_params.png — bar charts of times and counts
    - summary_best_params.png — summary charts
- <model>_best_results/ (if save_annotated_images=True)
Annotated images showing detected people with bounding boxes and confidence labels.

## 🔍 Example Outputs
- MAE comparison table across all YOLO variants
- Bar charts visualizing processing time vs. count accuracy
- Sample annotated images for visual inspection







