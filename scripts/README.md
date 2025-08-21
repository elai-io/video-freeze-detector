# Video Freeze Detection — Additional Scripts

This folder contains extra scripts for advanced analysis, metrics evaluation, and visualization. These are not required for basic freeze detection, but provide powerful tools for research and diagnostics.

## 🎬 `concat_emotions_videos.py`

**Purpose:**
- Concatenates emotion videos in a fixed order with text overlays
- Selects random clips for each emotion (one "normal", one "high" strength)
- Automatically detects subject ID from path and generates output filename
- Optimized for speed with 720p downscaling and fast encoding

**File structure expected:**
```
ID_9/
  front/M001_<view>_angry_high_001.mov
  front/M001_<view>_angry_normal_002.mov
  front/M001_<view>_sad_high_003.mov
  front/M001_<view>_sad_normal_004.mov
  ...
```
Pattern: `<ID>_<view>_<emotion>_<strength>_<index>.mov`

**Usage:**
```bash
python3 scripts/concat_emotions_videos.py /fsx/dataset/unidata/ID_9/front --output-dir results/emotions_concat
```

**Arguments:**
- `input_dir` — Path to directory containing emotion video files
- `--output-dir` — Output directory (default: results/emotions_concat)
- `--seed` — Random seed for reproducible selection

**What it does:**
1. Detects subject ID from path (e.g., "ID_9" from "/fsx/dataset/unidata/ID_9/front")
2. Selects clips in order: Angry, Sad, Happy, Surprised, Confident, Confused, Disgust
3. For each emotion: picks one random "normal" and one random "high" strength clip
4. Transcodes each clip: 720p downscale, text overlay, H.264 fast preset, CRF 20, 60 fps
5. Overlays text: "{ID} - {Emotion} ({strength})" in top-left corner
6. Concatenates all segments into final video: `{ID}_emotions_concat.mp4`
7. Preserves audio and logs detection status

**Output:**
- `results/emotions_concat/ID_9_emotions_concat.mp4` — Final concatenated video
- Progress tracking: transcoding segments, concatenation, cleanup
- Audio detection logs for each segment and final output

**Features:**
- **Fixed emotion order**: Ensures consistent sequence across different subjects
- **Random selection**: Different clips each run (use `--seed` for reproducibility)
- **Speed optimized**: 720p downscaling, fast preset, efficient encoding
- **Audio preservation**: Maintains original audio with AAC 96k encoding
- **Progress feedback**: tqdm progress bars for all operations
- **Flexible input**: Works with any view type (frontal, left, right, etc.)

---

## 🖼️ `combine_video_frames.py`

**Purpose:**
- Combines frames from strict three-view sets into single composite images
- Supports strict folder layout per ID: `front/`, `left/`, `right/`
- Matches triplets by unique filename suffix (e.g., `angry_high_001`)
- Applies center cropping, measurement scale, and top/bottom reference lines
- Generates per-triplet PNGs and collages (full + 8K)

**Strict structure example:**
```
ID_6/
  front/M006_frontal_angry_high_001.mov
  left/ M006_left_angry_high_001.mov
  right/M006_right_angry_high_001.mov
```
Unique key: `angry_high_001` (everything after the first two underscore tokens)

**Usage:**
```bash
py scripts/combine_video_frames.py "path/to/ID_folder" --output results
```

**Arguments:**
- `input_dir` — Path to ID folder containing `front/left/right`
- `--output` — Output directory (default: results)
- `--crop-fraction` — Fraction of width for center crop (default: 0.4)
- `--line-position` — Position of top/bottom lines as fraction from edges (default: 0.15)
- `--frame-index` — Zero-based frame index to extract from each video (default: 0)
- `--no-collage` — Skip collages (by default both full and 8K are created)
- `--verbose`, `-v` — Verbose output

**What it does:**
1. Scans `front/left/right` and groups files by unique suffix
2. Extracts the requested frame from each video (default first frame)
3. Crops center region (default 40% of width)
4. Draws central horizontal measurement scale with ticks (±300px, every 50px; vertical 100px)
5. Draws blue reference lines at `line-position` from top and bottom
6. Combines as `left | front | right` and saves as PNG named after the `front` stem
7. Builds `collage_full.jpg` and `collage_8k.jpg` (12 columns per row, width-exact 8K)

## 📹 `video_quality_analyzer.py`

**Purpose:**
- Analyzes comprehensive image quality metrics for all video files in a specified directory
- Supports any number of video files (not limited to 3 cameras)
- Outputs 7 different quality metrics for each video file
- Useful for comparing camera settings and video quality across multiple files

**Usage:**
```bash
py scripts/video_quality_analyzer.py path/to/videos
```

**Arguments:**
- `input_dir` — Path to directory containing video files
- `--sample-frames` — Number of frames to sample per video (overrides --mode)
- `--mode` — Analysis mode: fast (10 frames), normal (100 frames), full (all frames) (default: fast)
- `--detailed`, `-d` — Show detailed metrics (brightness, contrast, saturation, color balance)
- `--verbose`, `-v` — Verbose output with per-file analysis details

**Usage Examples:**
```bash
# Basic usage (fast mode - 10 frames, main metrics only)
py scripts/video_quality_analyzer.py "C:\path\to\videos"

# Fast mode with detailed metrics
py scripts/video_quality_analyzer.py "C:\path\to\videos" --mode fast --detailed

# Normal mode (100 frames, main metrics only)
py scripts/video_quality_analyzer.py "C:\path\to\videos" --mode normal

# Normal mode with detailed metrics
py scripts/video_quality_analyzer.py "C:\path\to\videos" --mode normal --detailed

# Full analysis (all frames, main metrics only)
py scripts/video_quality_analyzer.py "C:\path\to\videos" --mode full

# Custom number of frames with detailed metrics
py scripts/video_quality_analyzer.py "C:\path\to\videos" --sample-frames 50 --detailed --verbose
```

**Example Output (Default - Main Metrics):**
```
📹 VIDEO QUALITY ANALYZER
====================================================================================================
Directory: C:\Videos\Test
Found 5 video files
Analysis mode: fast (10 frames)
----------------------------------------------------------------------------------------------------

========================================
QUALITY ANALYSIS RESULTS
========================================
Filename   Sharpness  Focus      Noise     
           Laplacian ↑ Tenengrad ↑ Level ~   
----------------------------------------
camera1.mp4    423.1      1247.8      2.1
camera2.avi    156.3      892.4       3.4
camera3.mov    87.9       445.2       1.8
backup_cam.mkv 312.7      1089.3      2.7
security_feed_with_very_long_filename_example.mp4 201.5 756.9 2.9
========================================
✅ Analyzed 5 video files
Note: ↑ = higher is better, ↓ = lower is better, ~ = expected to be similar
Use --detailed flag to see brightness, contrast, saturation, and color balance metrics
```

**Example Output (Detailed Mode):**
```
📹 VIDEO QUALITY ANALYZER
====================================================================================================
Directory: C:\Videos\Test
Found 5 video files
Analysis mode: fast (10 frames)
----------------------------------------------------------------------------------------------------

============================================================================================
QUALITY ANALYSIS RESULTS (DETAILED)
============================================================================================
Filename   Sharpness  Focus      Noise      Brightness Contrast   Saturation Color Balance
           Laplacian ↑ Tenengrad ↑ Level ~    Avg ↑      Std ↑      HSV-S ↑    Deviation ↓ 
--------------------------------------------------------------------------------------------
camera1.mp4    423.1      1247.8      2.1        156.3      42.1       98.5       0.089
camera2.avi    156.3      892.4       3.4        142.8      38.7       95.2       0.112
camera3.mov    87.9       445.2       1.8        148.9      45.3       102.1      0.134
backup_cam.mkv 312.7      1089.3      2.7        151.2      41.6       99.8       0.097
security_feed_with_very_long_filename_example.mp4 201.5 756.9 2.9 145.6 39.8 97.3 0.105
============================================================================================
✅ Analyzed 5 video files
Note: ↑ = higher is better, ↓ = lower is better, ~ = expected to be similar
Metrics: Sharpness (Laplacian), Focus (Tenengrad), Noise (local variance),
         Brightness (0-255), Contrast (Std Dev), Saturation (HSV-S), Color Balance (deviation)
```

**Features:**
- **Dynamic table width**: Automatically adjusts to the longest filename
- **No filename truncation**: Full filenames are always displayed
- **Compact default view**: Shows only main metrics (Sharpness, Focus, Noise) by default
- **Detailed mode**: Use `--detailed` flag to see all 7 metrics
- **Efficient layout**: Optimized for readability and screen space

**Metrics:**
- **Laplacian ↑** — Sharpness measurement (higher = sharper)
- **Tenengrad ↑** — Focus quality (higher = better focus)
- **Noise ~** — Local variance noise estimate (expected to be similar across cameras)
- **Brightness ↑** — Average pixel brightness (0-255, higher = brighter)
- **Contrast ↑** — Standard deviation of pixel values (higher = more contrast)
- **Saturation ↑** — Color saturation from HSV (higher = more saturated)
- **Color Balance ↓** — Deviation from neutral color balance (lower = more neutral)

---

## 📊 `metrics_analyzer.py`

**Purpose:**
- Comprehensive analysis and benchmarking of all freeze detection metrics (regular and edge-based)
- Compares 16 metrics, ranks their performance, and outputs detailed reports
- Evaluates detection accuracy against a list of known freeze frames
- Generates CSV, JSON, and Markdown documentation

**Usage:**
```bash
py scripts/metrics_analyzer.py videos --known-freezes 43 89 93 149 153 182 205 209 213 223 233 235 269 273
```

**Arguments:**
- `videos` — Path to folder with input videos
- `--known-freezes`, `-f` — List of known freeze frame numbers (space-separated)
- `--output`, `-o` — Output directory (default: `metrics_analysis`)
- `--recalculate`, `-r` — Force recomputation of frame differences
- `--cache-file`, `-c` — Path to cache file for frame differences

**Outputs:**
- `metrics_analysis/freeze_analysis_results.csv` — Main results table (metric, average rank, detection rate, etc)
- `metrics_analysis/full_metrics_table.csv` — All metric values for all frames
- `metrics_analysis/metrics_documentation.md` — Markdown documentation for all metrics
- `metrics_analysis/freeze_analysis_results.json` — Results in JSON format

**Best metric:** `edge_velocity_min_ratio` (85.7% detection rate)

---

## 🖼️ `save_frame_diffs.py`

**Purpose:**
- Advanced frame difference analysis for single video files
- Creates side-by-side visualizations (original crop + frame difference)
- Performs dual analysis: regular frame differences + edge-based differences
- Generates video from all difference frames with configurable FPS
- Comprehensive plotting with both difference types

**Usage:**
```bash
py scripts/save_frame_diffs.py "path/to/video.mp4" "output/directory"
```

**Arguments:**
- `video_file` — Path to input video file (MP4, AVI, MOV, MKV)
- `output_dir` — Output directory for difference images and analysis
- `--fps` — FPS for output video (default: 5.0)
- `--contrast` — Contrast factor for difference images (default: 2.0)
- `--crop-fraction` — Fraction of width for center crop (creates square, default: 0.5)
- `--no-video` — Skip creating output video
- `--no-plot` — Skip saving the plot of differences
- `--verbose`, `-v` — Verbose output with per-frame details

**Usage Examples:**
```bash
# Basic usage with default settings
py scripts/save_frame_diffs.py "C:\videos\test.mp4" "results\test_analysis"

# Custom FPS and contrast
py scripts/save_frame_diffs.py "C:\videos\test.mp4" "results\test_analysis" --fps 10 --contrast 3.0

# Large crop area (70% of width)
py scripts/save_frame_diffs.py "C:\videos\test.mp4" "results\test_analysis" --crop-fraction 0.7

# Skip video generation, only save images and plots
py scripts/save_frame_diffs.py "C:\videos\test.mp4" "results\test_analysis" --no-video

# Verbose output for debugging
py scripts/save_frame_diffs.py "C:\videos\test.mp4" "results\test_analysis" --verbose
```

**Outputs:**
- `diff_XXXX.png` — Side-by-side images (original crop + frame difference)
- `frame_differences_video.mp4` — Video compilation of all difference frames
- `frame_differences_plot.png` — Combined plot showing regular and edge differences
- `frame_differences_data.csv` — CSV data with both difference types

**Features:**
- **Side-by-side visualization**: Original crop (left) + frame difference (right)
- **Dual analysis**: Regular frame differences + Canny edge-based differences
- **Video generation**: Automatic creation of difference video with configurable FPS
- **Comprehensive plotting**: Two subplots showing both difference types
- **Enhanced annotations**: Frame numbers, difference values, and labels
- **Flexible cropping**: Configurable center crop fraction (default 50% width)

**Analysis Types:**
- **Regular differences**: Pixel-by-pixel differences between consecutive frames
- **Edge differences**: Differences in Canny edge maps (structural changes)
- **Combined statistics**: Separate analysis for each difference type

---

## 🖼️ `save_central_diffs.py`

**Purpose:**
- Generates and saves images of frame differences for the central region of each camera
- Useful for visual inspection and debugging

**Usage:**
```bash
py scripts/save_central_diffs.py
```

**Outputs:**
- `output_diffs/diff_XXXX.png` — Concatenated difference images for each frame (center region, all cameras)

---

## 📄 `METRICS_ANALYZER_README.md`

- Detailed documentation for `metrics_analyzer.py`, including metric formulas, interpretation, and usage tips.

## 📊 `docs/METRICS_DOCUMENTATION.md`

- Comprehensive documentation of all 16 metrics with formulas, mathematical principles, and performance rankings.

---

## 📚 Notes

- All scripts use modules from the `../modules/` folder.
- Results are saved in their respective output folders.
- Scripts can be run independently of `main.py`.
- Supported video formats: MP4, AVI, MOV, MKV.

---

**For details on the main freeze detection pipeline, see the main project README.**

## Dependency Structure

```
scripts/
├── concat_emotions_videos.py    # Emotion video concatenation
├── video_quality_analyzer.py    # Video file quality analysis
├── metrics_analyzer.py          # Metrics analysis
├── save_frame_diffs.py          # Advanced frame difference analysis
├── save_central_diffs.py        # Diff image creation
├── METRICS_ANALYZER_README.md   # Analyzer documentation
└── README.md                    # General documentation
```

## Notes

- All scripts use modules from the `../modules/` folder
- Results are saved in their respective folders
- Scripts can be run independently of main.py
- Same video formats are supported (MP4, AVI, MOV, MKV) 