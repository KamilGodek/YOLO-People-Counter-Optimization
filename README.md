# YOLO-People-Counter-Optimization
Automated parameter tuning for YOLO models in people counting tasks. Features model comparison, CSV reports, visualizations and annotated images.



## 🎯 Features
- Automated parameter combination testing (confidence/IoU)
- Comparison of 5 YOLO models (nano, small, medium, large, xlarge)
- CSV reports and PNG visualizations
- Export of annotated detection images
- MAE validation against ground truth


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

📂 Data Preparation
Folder Structure
```bash
├── YOLO_weights/        # YOLO model files
├── input_images/        # Test images
└── results_tuned/       # Generated results
```

Steps:

1. Download YOLO Models from Ultralytics and place in YOLO_weights/:

Required files: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

2. Prepare Test Images:

Place JPG/PNG images in input_images/

Name files sequentially: 0.jpg, 1.jpg, 2.jpg,...

Note: 0.jpg is used for model warm-up

3. Edit Ground Truth Values in script:

# In people_counter_optimization.py, line 7:
actual_people_counts = [15, 13, 31, ...]  # True counts for images 1.jpg, 2.jpg,...


🚀 Usage

```bash
python people_counter_optimization.py
```

Processing Pipeline:
Automatic device detection (GPU/CPU)

Testing 35 parameter combinations per model

Generating results in results_tuned/:

📊 Comparison plots (PNG)

📄 CSV reports with timings and counts

🖼️ Annotated images for best parameters

⚙️ Configuration
Edit parameters in the Configuration section of people_counter_optimization.py:

```bash
# --- Configuration ---
conf_thresholds = [0.06, 0.15]    # Confidence threshold range
iou_thresholds = [0.26, 0.33]     # IoU threshold range
inference_size = 3200              # Processing resolution (e.g., 640, 1280)
save_annotated_images = True       # Export detection visualizations
```
Adjust these lists to match the length of your test set and desired parameter ranges.


📊 Results
Sample Outputs
Model Comparison
Performance Summary

Output Files
File	Description
people_counts_best.csv	Per-image detection counts
processing_times_best.csv	Per-image processing times
summary_best.csv	MAE scores and total statistics
