# Video Freeze Detection System

[![Version](https://img.shields.io/badge/version-0.5.2-blue.svg)](CHANGELOG.md)

A robust system for detecting video freezes in synchronized multi-camera recordings using edge-based frame difference analysis with comprehensive sequence detection.

📋 **[Changelog](CHANGELOG.md)** - See what's new in each version

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your videos

Put your three synchronized video files (MP4, AVI, MOV, MKV) in a folder.

### 3. Run freeze detection

```bash
python main.py path/to/videos
```

#### Optional arguments:
- `--output`, `-o` — Output folder for results (default: `output`)
- `--freeze-threshold`, `-t` — Freeze detection threshold (default: 0.25)
- `--visualization-count`, `-p` — Number of most suspicious frames to visualize (default: 5)
- `--verbose`, `-v` — Verbose output

Examples:
```bash
# Standard analysis
python main.py videos --freeze-threshold 0.25 --output results --verbose

# Analyze top 10 most suspicious frames
python main.py videos --visualization-count 10 --output analysis
```

## 📁 Project Structure

```
VideoFreezeDetector/
├── main.py                     # Main freeze detection script
├── requirements.txt            # Python dependencies
├── README.md                   # Main documentation (this file)
├── modules/                    # Core modules
│   ├── video_analyzer.py       # Video loading and frame analysis
│   ├── freeze_detector_edge.py # Edge-based freeze detection with sequences
│   ├── visualizer_edge.py      # Visualization and image creation
│   └── README.md               # Module documentation
├── scripts/                    # Additional analysis scripts
│   ├── metrics_analyzer.py      # Detailed metrics analysis
│   ├── combine_video_frames.py  # Combine frames (front/left/right), overlays, collages (full + 8K)
│   ├── head_pose_three_views.py # Head pose analysis from three views
│   └── README.md                # Scripts documentation
├── videos/                     # Input video files (example)
├── output/                     # Analysis results and visualizations
└── docs/                       # Additional documentation
```

## 🧠 How It Works

### Edge-based Analysis with Sequence Detection:

1. **Edge Detection:** Uses Canny edge detection to preprocess frames, making freeze detection robust to noise and lighting changes.

2. **Normalized Metrics:** For each frame, calculates normalized edge differences:
   - `edge_raw[frame][camera]` = raw edge difference values
   - `edge_normalized[frame][camera]` = normalized by max value per frame
   - `frame_metric[frame]` = minimum normalized value per frame

3. **Freeze Detection:** Uses configurable threshold (default 0.25) to identify freeze frames per camera.

4. **Sequence Analysis:** Finds longest consecutive freeze sequences for each camera individually.

5. **Image Quality Assessment:** Analyzes overall video quality using two metrics:
   - **Laplacian Variance** for sharpness measurement (focused on central face region)
   - **Tenengrad Variance** for focus quality assessment (focused on central face region)

6. **Multi-Camera Edge Differences Plot:** Creates comprehensive visualization showing edge differences over frames for all three cameras simultaneously.

7. **Quality Assessment:** Provides unified severity ratings based on freeze percentages and image quality.

### Key Features:

- **Per-camera sequence detection:** Identifies longest freeze sequences for each camera
- **Unified severity system:** CRITICAL/HIGH/MEDIUM/LOW based on percentage thresholds
- **Image quality analysis:** Evaluates sharpness and focus quality per camera (focused on central face region)
- **Multi-camera visualization:** Edge differences plot showing all cameras over time
- **Comprehensive reporting:** CSV, Excel, and JSON outputs with detailed frame data
- **Visual analysis:** Creates annotated images for most suspicious frames
- **Quality rating:** Overall system quality assessment including image quality metrics

## 📊 Analysis Metrics

### Severity Criteria (unified across all components):
- **CRITICAL 🔴**: ≥10% freeze rate
- **HIGH 🟠**: 5-9.9% freeze rate  
- **MEDIUM 🟡**: 2-4.9% freeze rate
- **LOW 🟢**: <2% freeze rate

### Quality Rating:
- **POOR 🔴**: ≥10% total freeze instances
- **BAD 🟠**: 5-9.9% total freeze instances
- **FAIR 🟡**: 2-4.9% total freeze instances  
- **GOOD 🟢**: <2% total freeze instances

### Image Quality Metrics:

The system analyzes the overall quality of each camera's video feed using two complementary metrics:

#### **Laplacian Variance (Sharpness ↑)**
- **Purpose**: Measures image sharpness and focus quality
- **Method**: Applies Laplacian edge detection and calculates variance on central face region (20% width, 50% height)
- **Interpretation**: Higher values = sharper images
- **Typical range**: 50-1000+ (depends on content and resolution)
- **Optimization**: Analyzes only 20 sample frames for efficiency

#### **Tenengrad Variance (Focus Quality ↑)**
- **Purpose**: Alternative focus quality measurement using gradient magnitude
- **Method**: Applies Sobel filters and calculates gradient variance on central face region (20% width, 50% height)
- **Interpretation**: Higher values = better focus
- **Typical range**: 100-5000+ (depends on content and resolution)
- **Optimization**: Analyzes only 20 sample frames for efficiency

**Note**: The ↑ symbol indicates that higher values represent better quality. These metrics help identify cameras with poor image quality that might affect freeze detection accuracy or indicate hardware issues.

## 📦 Output Files

All results are saved in the output folder:

### Data Files:
- `freeze_analysis.csv` — Frame-by-frame data with metrics and freeze flags
- `freeze_analysis.xlsx` — Excel version with multiple sheets:
  - Frame_Analysis: Detailed frame data
  - Statistics: Summary statistics
  - Longest_Sequences: Top freeze sequences by camera
- `freeze_analysis_report.json` — Complete analysis report with all statistics

### Visualization Images:
- `{rank}_{frame}_{metric}_cam{N}_1.jpg` — Previous frame (center region)
- `{rank}_{frame}_{metric}_cam{N}_2.jpg` — Current frame (center region)
- `{rank}_{frame}_{metric}_cam{N}_3.jpg` — Edge difference visualization
- `edge_differences_plot.png` — Multi-camera edge differences over time plot

## 📈 Analysis Output

### Console Output Example:
```
Top 2 Longest Freeze Sequences by Camera:
--------------------------------------------------------------------------------
   1. Camera 2 | Frames  46- 47 | Length:  2 | Avg metric: 0.1123
   2. Camera 1 | Frames  21- 23 | Length:  3 | Avg metric: 0.2341

Camera Analysis:
  Camera 1: 59 freezes (9.23%) of 639 frames | HIGH 🟠
  Camera 2: 66 freezes (10.33%) of 639 frames | CRITICAL 🔴
  Camera 3: 32 freezes (5.01%) of 639 frames | HIGH 🟠

Image Quality Analysis:
  Camera 1 Quality:
    Sharpness (Laplacian ↑): 423.1
    Focus (Tenengrad ↑): 1247.8
  Camera 2 Quality:
    Sharpness (Laplacian ↑): 156.3
    Focus (Tenengrad ↑): 892.4
  Camera 3 Quality:
    Sharpness (Laplacian ↑): 87.9
    Focus (Tenengrad ↑): 445.2

======================================================================
FINAL SUMMARY
======================================================================
📊 TOTAL FREEZE INSTANCES: 157 (8.19%)
📊 SUSPICIOUS FRAMES: 146 (22.85%)
⚠️  CRITICAL PATTERNS: 4 sequences (2+ frames)
🎯 QUALITY RATING: BAD 🟠
📁 IMAGES GENERATED: 32 sets
📄 DATA SAVED: CSV, Excel, and JSON reports
📊 PLOTS GENERATED: Multi-camera edge differences visualization
======================================================================
```

## 🛠️ Architecture

### Core Modules:
- **video_analyzer.py** — Video loading, frame extraction, edge-based frame differences
- **freeze_detector_edge.py** — Freeze detection, sequence analysis, statistics generation
- **visualizer_edge.py** — Image creation with annotations and metrics

### Analysis Flow:
1. Load and validate synchronized videos
2. Compute edge-based frame differences
3. Detect freezes using normalized threshold
4. Find longest sequences per camera
5. Generate comprehensive statistics
6. Create visualization images and multi-camera plots
7. Export data to CSV/Excel/JSON

---

**This system provides accurate, comprehensive analysis of video freeze events with detailed reporting and visualization capabilities.** 

# 🎯 Additional Analysis Scripts

## Video Frame Combination (`scripts/combine_video_frames.py`)

Combine frames from strict three-view sets into single images with overlays and build collages.

```
py scripts/combine_video_frames.py path/to/ID_folder --output results --frame-index 100 --verbose
```

Features:
- Strict structure: `front/`, `left/`, `right/` under each ID
- Triplet matching by unique suffix (e.g., `angry_high_001`)
- Center crop (default 40% of width), measurement scale, top/bottom lines
- Per-triplet PNGs named after the `front` file stem
- Collages: `collage_full.jpg` and `collage_8k.jpg` (12 columns, exact 8K width)

## Head Pose Analysis (Three Views)

This standalone utility leverages [SixDRepNet](https://github.com/thohemp/6DRepNet) to estimate **yaw** angles from three synchronised camera views (left, front, right).

```
python scripts/head_pose_three_views.py \
  --left  path/to/left_view.mp4 \
  --front path/to/front_view.mp4 \
  --right path/to/right_view.mp4
```

Output example:

```
HEAD POSE ANALYSIS (6DRepNet)
====================================================================================================
left:  43.1 ± 2.7
front:  0.3 ± 1.5
right: 30.2 ± 2.1
✅ Done
```

Arguments:

* `--sample-frames N` – Limit processing to *N* evenly-spaced frames (default **0** = use all)
* `--verbose`         – Show per-frame progress bars

Internally the script computes a 95 % confidence interval around the mean yaw value using a normal-approximation (z = 1.96) and prints the result in the concise `mean ± margin` form.

> NOTE: SixDRepNet weights are downloaded automatically on first run. 