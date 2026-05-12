# Video Data Preparation — MediaPipe Feature Extraction


Extracts facial, pose, and hand features from frontal videos using **MediaPipe FaceLandmarker v2** and aggregates them into a single CSV per dataset.

---


## Setup

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Download **face_landmarker_v2_with_blendshapes.task** from the [MediaPipe Face Landmarker page](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place it in a `model/` folder.

---

## Usage
 
### 1. Extract features
Run `mediapipe.ipynb`. Update `root_folder` inside the notebook to point to your video directory.

The notebook finds every `Front.mp4` nested in subfolders and produces three files per video under `Output/`:


| File | Contents |
|---|---|
| `*_mediapipe_data.csv` | Raw per-frame landmarks |
| `*_mediapipe_data_interpolated.csv` | Interpolated, with rotation angles, distances, and peak indicators |
| `*_features.csv` | Single-row summary (mean, std, peaks/sec) for ML |


### 2. Aggregate
Run:
```bash
python aggregate.py
```
Update `root_folder` in `aggregate.py` to match your dataset path. Merges all `*_features.csv` files into:
```
Output/Aggregated/<tale_name>/AllFront_features.csv
```

 

---

## Pipeline overview

Each video goes through these steps in order:

1. **Landmark extraction** — pose and hand keypoints via MediaPipe Holistic; face cropped from frame before running FaceLandmarker for blendshapes and head transform
2. **Interpolation** — linear fill for missing frames
3. **Derived features** — Euler head rotation, arm/torso angles, 8 pairwise distances, cumulative per-landmark motion
4. **Peak detection** — `scipy.find_peaks` on all signal columns, written back to the interpolated CSV
5. **Flattening** — mean, std, and peaks-per-second collapsed to one row per video

---

## Output feature groups

| Group | Examples |
|---|---|
| Pose | `pose_LEFT_SHOULDER_x/y/z`, `pose_NOSE_x/y/z` |
| Hands | `left_hand_WRIST_x/y/z`, `right_hand_INDEX_FINGER_TIP_x` |
| Face blendshapes | `eyeBlinkLeft`, `jawOpen`, `mouthSmileRight` (52 total) |
| Head rotation | `head_pitch_deg`, `head_yaw_deg`, `head_roll_deg` |
| Arm/torso angles | `left_arm_angle`, `torso_pitch`, `torso_roll`, `torso_yaw` |
| Distances | `dist_wrist_lr`, `dist_left_wrist_to_nose`, … |
| Cumulative motion | `L_WRIST_accum_dist`, `NOSE_accum_dist`, … |
| Summaries | `*_mean`, `*_std`, `*_peaks_per_s` |