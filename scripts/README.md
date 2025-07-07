# Video Freeze Detection — Additional Scripts

This folder contains extra scripts for advanced analysis, metrics evaluation, and visualization. These are not required for basic freeze detection, but provide powerful tools for research and diagnostics.

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

## Структура зависимостей

```
scripts/
├── metrics_analyzer.py          # Анализ метрик
├── save_central_diffs.py        # Создание diff изображений
├── METRICS_ANALYZER_README.md   # Документация анализатора
└── README.md                    # Общая документация
```

## Примечания

- Все скрипты используют модули из папки `../modules/`
- Результаты сохраняются в соответствующие папки
- Скрипты можно запускать независимо от main.py
- Поддерживаются те же форматы видео (MP4, AVI, MOV, MKV) 