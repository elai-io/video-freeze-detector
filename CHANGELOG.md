# Changelog

All notable changes to the Video Freeze Detection System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5] - 2025-07-28

### Added
- **Enhanced video frame combination script** (`scripts/combine_video_frames.py`):
  - **Multi-format support**: Now supports MOV, AVI, MP4, MKV, WMV, FLV, WEBM formats
  - **Automatic video folder detection**: Finds folders with exactly 3 video files
  - **Center crop processing**: Extracts 50% center portion of each video frame
  - **Crosshair overlay**: Adds red crosshair in center of each processed frame
  - **Bottom line indicator**: Adds blue horizontal line at 15% from bottom
  - **Horizontal combination**: Combines 3 processed frames into single image
  - **Batch processing**: Processes multiple video folders automatically
  - **Configurable parameters**: Crop fraction and line position customization
- **Head pose analysis script** (`scripts/head_pose_three_views.py`):
  - **Multi-view head pose estimation**: Analyzes head pose from three camera angles
  - **3D head pose detection**: Uses computer vision for head orientation analysis
  - **Cross-camera consistency**: Validates head pose across different viewpoints
  - **Pose visualization**: Visual representation of head orientation
  - **Statistical analysis**: Comprehensive pose statistics and reporting

### Changed
- **Improved video format compatibility**: Extended support beyond MOV files to include common video formats
- **Enhanced folder processing**: More robust detection of video file combinations
- **Better error handling**: Improved handling of unsupported video formats and corrupted files

### Fixed
- **Video format detection**: Fixed case sensitivity issues in file extension matching
- **Path handling**: Improved cross-platform compatibility for file path operations

## [0.4.2] - 2025-07-22

### Changed
- **Visualization parameter change**: Changed from percentage to count (--visualization-count instead of --visualization-percent)
- **Default behavior**: Still defaults to 5 frames for visualization
- **Function signature update**: Updated `detect_freezes_for_visualization()` to accept integer count instead of float percentage
- **Documentation update**: Updated README.md examples and argument descriptions

## [0.4.1] - 2025-07-14

### Fixed
- **Aspect ratio preservation**: Fixed face region scaling in image quality analysis to maintain proper proportions instead of creating distorted square images
- **Face region extraction**: Corrected `extract_face_region()` function in `modules/video_analyzer.py` to properly scale extracted regions while preserving aspect ratios
- **Visualization consistency**: Ensured all face region visualizations maintain correct proportions across different scaling operations

---

## [0.4.0] - 2025-07-14

### Added
- **Enhanced Frame Difference Analysis Script** (`scripts/save_frame_diffs.py`):
  - **Side-by-side visualization**: Original crop (left) + frame difference (right)
  - **Dual analysis**: Regular frame differences + edge-based differences
  - **Canny edge detection** for structural change analysis
  - **Video generation** from all difference frames with configurable FPS
  - **Comprehensive plotting**: Two subplots showing both difference types
  - **Enhanced CSV export** with both regular and edge difference data
  - **Flexible cropping**: Configurable center crop fraction (default 50% width)
  - **Improved annotations**: Frame numbers, difference values, and labels

### Added
- **New CLI parameters** for frame difference analysis:
  - `--fps` - Output video FPS (default: 5.0)
  - `--crop-fraction` - Center crop fraction (default: 0.5)
  - `--no-video` - Skip video generation
  - `--no-plot` - Skip plotting
- **Edge-based analysis**:
  - Gaussian blur preprocessing for noise reduction
  - Canny edge detector (thresholds: 50, 150)
  - Edge difference computation between consecutive frames
  - Separate statistics for edge differences
- **Enhanced visualization**:
  - Combined plots with regular and edge differences
  - Separate statistics for each analysis type
  - Color-coded graphs (blue for regular, green for edges)
  - Comprehensive statistical information

### Changed
- **Improved frame correspondence**: Original frame now matches the difference frame
- **Better crop logic**: Crop fraction applies to width, creating square crops
- **Enhanced VS Code configuration** with new script parameters
- **Updated documentation** with new script features and usage examples

### Fixed
- **Frame alignment issue**: Original and difference frames now properly correspond
- **Crop calculation**: Fixed to use width-based cropping for consistent square crops

---

## [0.3.1] - 2025-07-11

### Added
- **Enhanced Video Quality Analyzer** with 7 comprehensive metrics:
  - **Sharpness (Laplacian) ↑** - Image sharpness measurement
  - **Focus (Tenengrad) ↑** - Focus quality assessment
  - **Noise ~** - Local variance noise estimate (expected to be similar across cameras)
  - **Brightness ↑** - Average pixel brightness (0-255)
  - **Contrast ↑** - Standard deviation of pixel values
  - **Saturation ↑** - Color saturation from HSV
  - **Color Balance ↓** - Deviation from neutral color balance

### Changed
- **Two display modes** for video quality analyzer:
  - **Compact mode** (default) - Shows only 3 main metrics (Sharpness, Focus, Noise)
  - **Detailed mode** (`--detailed` flag) - Shows all 7 metrics
- **Improved metric indicators**:
  - `↑` = Higher is better
  - `↓` = Lower is better
  - `~` = Expected to be similar (for noise level)
- **Dynamic table width** - Automatically adjusts to longest filename
- **No filename truncation** - Full filenames always displayed

### Added
- **New CLI arguments** for video quality analyzer:
  - `--detailed`, `-d` - Show detailed metrics
  - `--mode` - Analysis modes: fast (10 frames), normal (100 frames), full (all frames)
  - `--sample-frames` - Custom number of frames to sample
- **VS Code launch configurations** for all analysis modes
- **Comprehensive documentation** with usage examples

### Fixed
- **Better noise interpretation** - Low noise doesn't always mean better quality
- **Improved metric ordering** - Noise moved after Focus for better logical flow

---

## [0.3.0] - 2025-07-11

### Added
- **Video Quality Analyzer Script** (`scripts/video_quality_analyzer.py`):
  - Analyzes image quality metrics for multiple video files
  - Supports any number of video files (not limited to 3 cameras)
  - Outputs Laplacian and Tenengrad variance values
  - Useful for comparing video quality across multiple files
  - Flexible CLI with `--mode` presets (fast/normal/full)
  - Wide filename display (100 characters)

### Added
- **Image Quality Metrics** in main analysis:
  - **Laplacian Variance** for sharpness measurement
  - **Tenengrad Variance** for focus quality assessment
  - Quality indicators (↑ = higher is better)
  - Integration with main freeze detection pipeline

### Changed
- **Enhanced main analysis output** with image quality information
- **Updated documentation** with quality metrics explanation
- **Improved VS Code configurations** for new script

### Removed
- **NIQE metric** - Removed due to heavy dependencies (PyTorch)
- **Heavy dependencies** - Simplified to use only OpenCV-based metrics

---

## [0.2.1] - 2025-07-09

### Added
- **Image Quality Analysis** to main freeze detection pipeline
- **NIQE, Laplacian, and Tenengrad variance** metrics for video quality assessment
- **Quality indicators** in analysis output (↑ = higher is better)
- **Integration** of quality metrics with freeze detection results

### Changed
- **Enhanced analysis output** to include image quality information
- **Updated documentation** to explain quality metrics
- **Improved reporting** with quality assessment

### Fixed
- **Dependency issues** with heavy packages like PyTorch
- **Performance optimization** for quality metrics computation

---

## [0.2.0] - 2025-07-08

### Added
- **Edge-based Freeze Detection** with comprehensive sequence analysis
- **16 different metrics** for freeze detection (8 regular + 8 edge-based)
- **Metrics Analyzer Script** (`scripts/metrics_analyzer.py`):
  - Comprehensive analysis and benchmarking of all freeze detection metrics
  - Compares 16 metrics, ranks their performance
  - Evaluates detection accuracy against known freeze frames
  - Generates CSV, JSON, and Markdown documentation
- **Best metric identification**: `edge_velocity_min_ratio` (85.7% detection rate)

### Added
- **Sequence Detection**:
  - Finds longest consecutive freeze sequences for each camera
  - Per-camera sequence analysis
  - Critical pattern identification (2+ frame sequences)
- **Enhanced Statistics**:
  - Unified severity system (CRITICAL/HIGH/MEDIUM/LOW)
  - Overall quality rating based on freeze percentages
  - Camera balance analysis
- **Comprehensive Reporting**:
  - CSV, Excel, and JSON outputs
  - Detailed frame-by-frame data
  - Statistics and longest sequences
- **Visualization Improvements**:
  - Edge difference visualization
  - Annotated images with metrics
  - Previous/current frame comparisons

### Changed
- **Core Algorithm**: Switched from jerk-based to edge-based detection
- **Performance**: Improved accuracy and robustness
- **Output Format**: Enhanced reporting with multiple formats
- **Documentation**: Comprehensive metrics documentation

### Removed
- **Jerk-based approach** - Replaced with more robust edge-based detection
- **Old visualization methods** - Updated to edge-based visualization

---

## [0.1.0] - 2025-07-07

### Added
- **Initial Video Freeze Detection System**
- **Jerk-based Freeze Detection** using 3rd derivative analysis
- **Basic Video Analysis**:
  - Video loading and validation
  - Frame difference computation
  - Freeze detection with configurable threshold
- **Core Modules**:
  - `video_analyzer.py` - Video loading and frame analysis
  - `freeze_detector_jerk.py` - Jerk-based freeze detection
  - `visualizer_jerk.py` - Basic visualization
- **Basic CLI Interface**:
  - Input path specification
  - Output folder configuration
  - Freeze threshold adjustment
  - Verbose output option
- **Simple Reporting**:
  - Basic statistics output
  - Freeze count per camera
  - Simple quality assessment

### Added
- **Project Structure**:
  - Modular architecture
  - Scripts directory for additional tools
  - Documentation structure
  - VS Code configuration
- **Basic Dependencies**:
  - OpenCV for video processing
  - NumPy for numerical computations
  - Pandas for data handling
  - Matplotlib for visualization

 