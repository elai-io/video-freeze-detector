# Video Freeze Detection System

[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)](https://github.com/elai-io/video-freeze-detector)

A robust system for detecting video freezes in synchronized multi-camera recordings using edge-based frame difference analysis with comprehensive sequence detection.

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
- `--visualization-percent`, `-p` — Percentage of most suspicious frames to visualize (default: 5.0%)
- `--verbose`, `-v` — Verbose output

Examples:
```bash
# Standard analysis
python main.py videos --freeze-threshold 0.25 --output results --verbose

# Analyze top 10% most suspicious frames
python main.py videos --visualization-percent 10.0 --output analysis
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
│   ├── metrics_analyzer.py     # Detailed metrics analysis
│   └── README.md               # Scripts documentation
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

5. **Quality Assessment:** Provides unified severity ratings based on freeze percentages.

### Key Features:

- **Per-camera sequence detection:** Identifies longest freeze sequences for each camera
- **Unified severity system:** CRITICAL/HIGH/MEDIUM/LOW based on percentage thresholds
- **Comprehensive reporting:** CSV, Excel, and JSON outputs with detailed frame data
- **Visual analysis:** Creates annotated images for most suspicious frames
- **Quality rating:** Overall system quality assessment

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

======================================================================
FINAL SUMMARY
======================================================================
📊 TOTAL FREEZE INSTANCES: 157 (8.19%)
📊 SUSPICIOUS FRAMES: 146 (22.85%)
⚠️  CRITICAL PATTERNS: 4 sequences (2+ frames)
🎯 QUALITY RATING: BAD 🟠
📁 IMAGES GENERATED: 32 sets
📄 DATA SAVED: CSV, Excel, and JSON reports
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
6. Create visualization images
7. Export data to CSV/Excel/JSON

## 📚 Key Improvements

### Version 0.2.0 Features:
- **Unified severity system** across all components
- **Per-camera sequence detection** with proper counting
- **Comprehensive data export** with multiple formats
- **Quality-based rating system** with intuitive naming
- **Percentage-based thresholds** that scale with video length
- **Critical pattern detection** counting sequences per camera

### Removed Legacy Components:
- Jerk-based detection (3rd derivative approach)
- Comprehensive vs standard mode distinction
- Frame-based suspicious sequence counting
- Complex quality scoring algorithms

---

**This system provides accurate, comprehensive analysis of video freeze events with detailed reporting and visualization capabilities.** 