# Video Data Preparation

# Mediapipe Feature Extraction

This project extracts **facial, pose, and hand features** from videos using **MediaPipe FaceLandmarker v2** and aggregates them into a single CSV.

---

## Setup

1. Install required packages:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Download the model file:
   - Go to [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
   - Download **face_landmarker_v2_with_blendshapes.task**
   - Place it in a folder named `model/`

---

## Steps

### 1. Extract Features
Run:
```
mediapipe.ipynb
```
This processes all `Front.mp4` videos and saves per-frame features to CSV files inside the `Output/` folder.

> **Note:** Update the `root_folder` path inside the notebook to point to the main video directory.

---

### 2. Aggregate Results
Run:
```
python aggregate.py
```
This merges all feature CSVs into:
```
Output/Aggregated/<tale_name>/AllFront_features.csv
```

> **Note:** Make sure to update the `root_folder` path inside `aggregate.py` to match the dataset path.

---
