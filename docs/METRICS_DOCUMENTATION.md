# Video Freeze Detection — Metrics Documentation

This document provides detailed explanations of all metrics used in the video freeze detection system, including formulas, mathematical principles, and practical interpretations.

## 📊 Overview

The system analyzes 16 different metrics, divided into two categories:
- **Regular metrics** (8): Based on raw frame differences
- **Edge-based metrics** (8): Based on edge-detected frame differences

All metrics are computed across three synchronized cameras and ranked by their ability to detect known freeze frames.

## 🔍 Terminology

- **Frame differences (diff)**: The absolute difference between consecutive frames
- **Velocity**: Same as frame differences (1st derivative of pixel values)
- **Acceleration**: Difference of velocities (2nd derivative)
- **Jerk**: Difference of accelerations (3rd derivative)
- **Edge differences**: Frame differences computed on edge-detected images

## 📈 Regular Metrics

### 1. `velocity_min`
**Formula:** `min(|diff₁|, |diff₂|, |diff₃|)`

**Description:** Minimum velocity across all cameras. Lower values indicate potential freezes.

**Interpretation:** When one camera has very low movement (freeze), this metric will be small.

---

### 2. `velocity_min_ratio`
**Formula:** `min(diff₁, diff₂, diff₃) / mean(diff₁, diff₂, diff₃)`

**Description:** Ratio of minimum velocity to mean velocity across cameras.

**Interpretation:** Values close to 0 indicate one camera is significantly less active than others (freeze candidate).

---

### 3. `velocity_std_ratio`
**Formula:** `std(diff₁, diff₂, diff₃) / mean(diff₁, diff₂, diff₃)`

**Description:** Coefficient of variation for velocity (standard deviation divided by mean).

**Interpretation:** High values indicate uneven movement across cameras, potentially indicating a freeze.

---

### 4. `acceleration_min`
**Formula:** `min(acc₁, acc₂, acc₃)` where `accᵢ = diffᵢ[t] - diffᵢ[t-1]`

**Description:** Minimum acceleration across cameras. Negative values indicate deceleration.

**Interpretation:** Large negative values suggest sudden stops in movement (freeze events).

---

### 5. `acceleration_range`
**Formula:** `max(acc₁, acc₂, acc₃) - min(acc₁, acc₂, acc₃)`

**Description:** Range of acceleration values across cameras.

**Interpretation:** High values indicate significant variation in acceleration, often during freeze transitions.

---

### 6. `jerk_max`
**Formula:** `max(|jerk₁|, |jerk₂|, |jerk₃|)` where `jerkᵢ = accᵢ[t] - accᵢ[t-1]`

**Description:** Maximum absolute jerk across cameras.

**Interpretation:** High values indicate sudden changes in acceleration (freeze→unfreeze transitions).

---

### 7. `jerk_range`
**Formula:** `max(jerk₁, jerk₂, jerk₃) - min(jerk₁, jerk₂, jerk₃)`

**Description:** Range of jerk values across cameras.

**Interpretation:** High values indicate significant variation in jerk, often during freeze events.

---

### 8. `jerk_max_abs`
**Formula:** `max(|jerk₁|, |jerk₂|, |jerk₃|)`

**Description:** Maximum absolute jerk value (same as jerk_max, kept for consistency).

**Interpretation:** Identifies the most dramatic acceleration change across cameras.

---

## 🎯 Edge-Based Metrics

Edge-based metrics apply the same formulas but use edge-detected frame differences instead of raw frame differences. This preprocessing step:

1. **Reduces noise** by focusing on structural changes
2. **Improves robustness** to lighting variations
3. **Enhances detection** of meaningful motion changes

### Edge Detection Process:
1. Convert frame to grayscale
2. Apply Gaussian blur (5×5 kernel) to reduce noise
3. Apply Canny edge detection (thresholds: 50, 150)
4. Compute differences between edge images

### 9. `edge_velocity_min`
**Formula:** `min(|edge_diff₁|, |edge_diff₂|, |edge_diff₃|)`

**Description:** Minimum edge velocity across cameras.

**Interpretation:** Detects freezes based on structural changes rather than pixel-level changes.

---

### 10. `edge_velocity_min_ratio` ⭐ **BEST METRIC**
**Formula:** `min(edge_diff₁, edge_diff₂, edge_diff₃) / mean(edge_diff₁, edge_diff₂, edge_diff₃)`

**Description:** Ratio of minimum edge velocity to mean edge velocity.

**Performance:** 85.7% detection rate on known freeze frames.

**Interpretation:** Most effective at identifying relative freezes while being robust to noise and lighting changes.

---

### 11. `edge_velocity_std_ratio`
**Formula:** `std(edge_diff₁, edge_diff₂, edge_diff₃) / mean(edge_diff₁, edge_diff₂, edge_diff₃)`

**Description:** Coefficient of variation for edge velocity.

**Interpretation:** Measures uniformity of structural changes across cameras.

---

### 12. `edge_acceleration_min`
**Formula:** `min(edge_acc₁, edge_acc₂, edge_acc₃)` where `edge_accᵢ = edge_diffᵢ[t] - edge_diffᵢ[t-1]`

**Description:** Minimum edge acceleration across cameras.

**Interpretation:** Detects deceleration in structural changes.

---

### 13. `edge_acceleration_range`
**Formula:** `max(edge_acc₁, edge_acc₂, edge_acc₃) - min(edge_acc₁, edge_acc₂, edge_acc₃)`

**Description:** Range of edge acceleration values.

**Interpretation:** Measures variation in structural acceleration changes.

---

### 14. `edge_jerk_max`
**Formula:** `max(|edge_jerk₁|, |edge_jerk₂|, |edge_jerk₃|)` where `edge_jerkᵢ = edge_accᵢ[t] - edge_accᵢ[t-1]`

**Description:** Maximum absolute edge jerk across cameras.

**Interpretation:** Detects sudden changes in structural acceleration.

---

### 15. `edge_jerk_range`
**Formula:** `max(edge_jerk₁, edge_jerk₂, edge_jerk₃) - min(edge_jerk₁, edge_jerk₂, edge_jerk₃)`

**Description:** Range of edge jerk values.

**Interpretation:** Measures variation in structural jerk changes.

---

### 16. `edge_jerk_max_abs`
**Formula:** `max(|edge_jerk₁|, |edge_jerk₂|, |edge_jerk₃|)`

**Description:** Maximum absolute edge jerk (same as edge_jerk_max).

**Interpretation:** Identifies the most dramatic structural acceleration change.

---

## 🏆 Performance Ranking

Based on analysis of known freeze frames:

| Rank | Metric | Detection Rate | Type |
|------|--------|----------------|------|
| 1 | `edge_velocity_min_ratio` | 85.7% | Edge-based |
| 2 | `edge_jerk_max` | 71.4% | Edge-based |
| 3 | `edge_acceleration_min` | 64.3% | Edge-based |
| 4 | `edge_jerk_range` | 64.3% | Edge-based |
| 5 | `jerk_max` | 57.1% | Regular |
| 6 | `edge_jerk_max_abs` | 57.1% | Edge-based |

## 🔬 Mathematical Principles

### Why Edge-Based Metrics Work Better

1. **Noise Reduction**: Edge detection filters out pixel-level noise while preserving structural information
2. **Lighting Invariance**: Edge detection is less sensitive to lighting changes
3. **Focus on Structure**: Emphasizes meaningful motion changes over minor variations
4. **Relative Detection**: Ratio-based metrics (like `edge_velocity_min_ratio`) are more robust than absolute values

### Why `edge_velocity_min_ratio` is the Best

1. **Relative Measurement**: Compares minimum to mean, making it scale-invariant
2. **Edge Preprocessing**: Benefits from noise reduction and lighting invariance
3. **Simple and Robust**: Less sensitive to outliers than derivative-based metrics
4. **Intuitive**: Directly measures the relative "freeze-ness" of the least active camera

## 📊 Usage Examples

### Detecting Freezes
```python
# Lower values indicate more likely freezes
if edge_velocity_min_ratio < 0.1:
    print("High probability of freeze detected")
```

### Comparing Cameras
```python
# Find which camera is most likely frozen
frozen_camera = edge_velocities.index(min(edge_velocities))
```

### Setting Thresholds
```python
# Adaptive threshold based on historical data
threshold = np.percentile(edge_velocity_min_ratios, 5)  # Bottom 5%
```

## 🔧 Implementation Notes

- All metrics are computed for each frame across all three cameras
- Edge detection uses OpenCV's Canny algorithm with fixed parameters
- Frame differences are computed as absolute differences in grayscale
- Derivatives are computed using simple finite differences
- Results are cached to avoid recomputation

---

**For practical usage, see the main README and scripts documentation.** 