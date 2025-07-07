# Video Freeze Detection System

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://github.com/elai-io/video-freeze-detector/releases/tag/v0.1.1)

A robust system for detecting video freezes in synchronized multi-camera recordings using edge-based frame difference analysis.

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your videos

Put your three synchronized video files (MP4, AVI, MOV, MKV) in the `videos/` folder.

### 3. Run the main freeze detector

```bash
py main.py videos
```

#### Optional arguments:
- `--output`, `-o` — Output folder for results (default: `output`)
- `--freeze-threshold`, `-t` — Percentage of most suspicious frames to analyze (default: 10)
- `--verbose`, `-v` — Verbose output

Example:
```bash
py main.py videos --freeze-threshold 5 --output output --verbose
```

## 📁 Project Structure

```
VideoFreezeDetector/
├── main.py                  # Main freeze detection script
├── requirements.txt         # Python dependencies
├── README.md                # Main documentation (this file)
├── modules/                 # Core modules (see modules/README.md)
├── scripts/                 # Additional scripts (see scripts/README.md)
├── videos/                  # Input video files
├── output/                  # Main output folder (results, images, reports)
├── metrics_analysis/        # Metrics analysis results (if used)
├── output_diffs/            # Frame difference images (if used)
└── docs/                    # Additional documentation
```

## 🧠 How It Works

- **Edge-based analysis:** Uses Canny edge detection to preprocess frames, making freeze detection robust to noise and lighting changes.
- **Metric:** The main metric is `edge_velocity_min_ratio` (minimum edge difference divided by mean edge difference across cameras), which achieved 85.7% accuracy on known freeze frames.
- **Visualization:** For each detected freeze, the system saves three images: previous frame, current frame, and their edge difference (center region, all cameras side-by-side).
- **Modular architecture:** All core logic is in the `modules/` folder for easy extension and testing.

## 📦 Output Files

All results are saved in the `output/` folder (or as specified by `--output`).

- `output/edge_analysis_report.json` — JSON report with analysis statistics
- `output/XXX_XXXXXX_0.123456_camX_1.jpg` — Previous frame (center region, all cameras)
- `output/XXX_XXXXXX_0.123456_camX_2.jpg` — Current frame (center region, all cameras)
- `output/XXX_XXXXXX_0.123456_camX_3.jpg` — Edge difference image (center region, all cameras)

## 🛠️ Architecture

- **main.py** — Entry point, orchestrates the detection pipeline
- **modules/video_analyzer.py** — Loads videos, computes regular and edge-based frame differences
- **modules/freeze_detector_edge.py** — Implements the edge-based freeze detection metric
- **modules/visualizer_edge.py** — Creates annotated output images

## 📚 More Documentation

- See `modules/README.md` for details on all modules and their roles.
- See `scripts/README.md` for all additional scripts (metrics analysis, frame diff visualization, etc).
- See `docs/METRICS_DOCUMENTATION.md` for detailed explanation of all metrics, formulas, and mathematical principles.
- See `docs/JERK_BASED_APPROACH.md` for switching to legacy 3rd derivative (jerk-based) approach.

---

**This project is designed for research and practical analysis of video freeze events.** 