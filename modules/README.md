# Video Freeze Detection — Modules

This folder contains all core modules (classes) for the video freeze detection system.

## Active Modules (used in main.py)

### `video_analyzer.py`
**Main class:** `VideoAnalyzer`

**Purpose:**
- Loads and validates video files
- Computes regular and edge-based frame differences
- Preprocesses frames with Canny edge detection
- Extracts frames by index

**Used in:** main.py, metrics_analyzer.py, save_central_diffs.py

### `freeze_detector_edge.py`
**Main class:** `FreezeDetectorEdge`

**Purpose:**
- Detects freezes using the edge_velocity_min_ratio metric
- Computes edge velocity metrics for each frame
- Identifies the most suspicious camera for each frame
- Sorts frames by suspicion score
- Generates analysis statistics

**Metric:** edge_velocity_min_ratio (85.7% accuracy on known data)

**Used in:** main.py

### `visualizer_edge.py`
**Main class:** `VisualizerEdge`

**Purpose:**
- Visualizes edge-based freeze detection results
- Creates images for previous/current frames and edge differences
- Adds metric annotations
- Saves results to files

**Used in:** main.py

## Legacy Modules (not used in main.py, kept for reference)

### `freeze_detector_jerk.py`
**Main class:** `FreezeDetectorJerk`

**Purpose:**
- Detects freezes using the 3rd derivative (jerk)
- Computes velocity, acceleration, and jerk for each frame
- Uses the MAX_3RD metric
- Detects sudden motion changes

**Metric:** jerk_max (3rd derivative)

**Status:** Legacy — replaced by edge-based approach

### `visualizer_jerk.py`
**Main class:** `VisualizerJerk`

**Purpose:**
- Visualizes jerk-based freeze detection results
- Creates images with jerk metrics and annotations
- Shows derivatives for each camera

**Status:** Legacy — replaced by edge-based visualization

## Dependency Structure

```
main.py
├── modules/video_analyzer.py
├── modules/freeze_detector_edge.py
└── modules/visualizer_edge.py

scripts/metrics_analyzer.py
└── modules/video_analyzer.py

scripts/save_central_diffs.py
└── (does not use modules)
```

- **Edge-based approach** gives the best results (85.7% accuracy)
- **Jerk-based approach** is kept for comparison and possible future use
- All modules follow PEP-8 and use type hints
- Documentation is in English

## 📚 Related Documentation

- See main `README.md` for project overview and usage
- See `scripts/README.md` for additional tools
- See `docs/METRICS_DOCUMENTATION.md` for detailed metric explanations
- See `docs/JERK_BASED_APPROACH.md` for switching to legacy jerk-based approach 