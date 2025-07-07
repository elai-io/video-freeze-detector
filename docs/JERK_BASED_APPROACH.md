# Video Freeze Detection — Jerk-Based Approach

This document explains how to switch from the current edge-based approach back to the jerk-based (3rd derivative) approach using the legacy modules.

## 🔄 Overview

The system currently uses `edge_velocity_min_ratio` as the main metric (85.7% accuracy). However, the legacy jerk-based approach using `jerk_max` (3rd derivative) is still available and can be useful in certain scenarios.

## 📊 Performance Comparison

| Approach | Best Metric | Detection Rate | Pros | Cons |
|----------|-------------|----------------|------|------|
| **Edge-based** | `edge_velocity_min_ratio` | 85.7% | Robust to noise, lighting invariant | More complex preprocessing |
| **Jerk-based** | `jerk_max` | 57.1% | Simple, detects sudden changes | Sensitive to noise, lighting changes |

## 🔧 Switching to Jerk-Based Approach

### Step 1: Update main.py imports

Replace the current imports in `main.py`:

```python
# Current (edge-based)
from modules.freeze_detector_edge import FreezeDetectorEdge
from modules.visualizer_edge import VisualizerEdge

# Change to (jerk-based)
from modules.freeze_detector_jerk import FreezeDetectorJerk
from modules.visualizer_jerk import VisualizerJerk
```

### Step 2: Update detector initialization

Replace the detector initialization:

```python
# Current (edge-based)
detector = FreezeDetectorEdge(frame_differences, edge_frame_differences, 
                            threshold_percent=args.freeze_threshold)

# Change to (jerk-based)
detector = FreezeDetectorJerk(frame_differences, threshold_percent=args.freeze_threshold)
```

### Step 3: Update visualizer initialization

Replace the visualizer initialization:

```python
# Current (edge-based)
visualizer = VisualizerEdge(analyzer, output_path)

# Change to (jerk-based)
visualizer = VisualizerJerk(analyzer, output_path)
```

### Step 4: Remove edge-based computations

Remove the edge-based frame difference computation:

```python
# Remove this line
edge_frame_differences = analyzer.compute_edge_frame_differences()
```

### Step 5: Update output messages

Update the console output messages to reflect jerk-based approach:

```python
# Change these messages
print(f"Using edge_velocity_min_ratio method - BEST METRIC")
print(f"Proven 85.7% accuracy on known freeze data")

# To
print(f"Using jerk_max method - 3rd derivative approach")
print(f"Proven 57.1% accuracy on known freeze data")
```



## 🔍 Key Differences

### Jerk-Based vs Edge-Based

| Aspect | Jerk-Based | Edge-Based |
|--------|------------|------------|
| **Preprocessing** | None (raw frames) | Canny edge detection |
| **Main Metric** | `jerk_max` | `edge_velocity_min_ratio` |
| **Accuracy** | 57.1% | 85.7% |
| **Noise Sensitivity** | High | Low |
| **Lighting Sensitivity** | High | Low |
| **Computational Cost** | Lower | Higher |
| **Detection Type** | Sudden changes | Relative freezes |

### Output Differences

**Jerk-based output:**
- `output_jerk/jerk_analysis_report.json`
- `output_jerk/XXX_XXXXXX_0.123456_camX_1.jpg` (previous frame)
- `output_jerk/XXX_XXXXXX_0.123456_camX_2.jpg` (current frame)
- `output_jerk/XXX_XXXXXX_0.123456_camX_3.jpg` (difference image)

**Edge-based output:**
- `output/edge_analysis_report.json`
- `output/XXX_XXXXXX_0.123456_camX_1.jpg` (previous frame)
- `output/XXX_XXXXXX_0.123456_camX_2.jpg` (current frame)
- `output/XXX_XXXXXX_0.123456_camX_3.jpg` (edge difference image)

## 🎯 When to Use Jerk-Based Approach

### Use jerk-based when:
- **Simple detection** is needed (no edge preprocessing)
- **Sudden motion changes** are the primary concern
- **Computational resources** are limited
- **Lighting conditions** are stable
- **Noise levels** are low

### Use edge-based when:
- **High accuracy** is required
- **Noise reduction** is important
- **Lighting conditions** vary
- **Robust detection** is needed
- **Research or production** use



## 📚 Related Documentation

- See `docs/METRICS_DOCUMENTATION.md` for detailed metric explanations
- See `modules/README.md` for module descriptions
- See main `README.md` for general usage

---

**Note: The jerk-based approach is kept for comparison and specific use cases where its characteristics are beneficial.** 