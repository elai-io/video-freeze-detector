import argparse
import os
import random
import shlex
import subprocess
import tempfile
from glob import glob
from typing import List, Tuple, Union
from tqdm import tqdm


def find_default_font() -> Union[str, None]:
    """Try to locate a reasonable default system font for drawtext."""
    common_fonts: List[str] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for font_path in common_fonts:
        if os.path.isfile(font_path):
            return font_path
    return None


def escape_drawtext_text(text: str) -> str:
    """Escape text for ffmpeg drawtext filter."""
    # Escape characters that are special in drawtext
    # Reference: https://ffmpeg.org/ffmpeg-filters.html#drawtext-1
    escaped: str = text.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("[", "\\[").replace("]", "\\]")
    return escaped


def build_drawtext_filter(
    text: str, fontfile: Union[str, None], fontsize: int = 48
) -> str:
    """
    Build a drawtext filter string placing text near the bottom center with a semi-transparent box.
    """
    escaped_text: str = escape_drawtext_text(text)
    font_opt: str = f":fontfile={fontfile}" if fontfile else ""
    style: str = (
        f"drawtext=text='{escaped_text}'{font_opt}:"
        f"fontcolor=white:fontsize={fontsize}:"
        f"box=1:boxcolor=black@0.4:boxborderw=12:"
        f"borderw=2:bordercolor=black:"
        f"x=24:y=24"
    )
    return style


def run_ffmpeg(cmd: List[str]) -> None:
    """Run an ffmpeg command and raise on failure."""
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {process.returncode}\nCommand: {' '.join(shlex.quote(c) for c in cmd)}\n\nStderr:\n{process.stderr.decode('utf-8', errors='ignore')}"
        )


def select_random_clip(
    input_dir: str, token: str, strength: str, rng: random.Random
) -> str:
    """Select one random clip path for the given emotion token and strength."""
    pattern: str = os.path.join(
        input_dir, f"*_*_{token}_{strength}_*.mov"
    )
    matches: List[str] = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files found for pattern: {pattern}. Ensure files exist like '*_<view>_{token}_{strength}_NNN.mov'."
        )
    return rng.choice(matches)


def detect_subject_id_from_path(input_dir: str) -> str:
    """Extract subject ID from path like /fsx/dataset/unidata/ID_9/front -> ID_9."""
    # Get the parent directory name (e.g., "ID_9" from "/fsx/dataset/unidata/ID_9/front")
    parent_dir: str = os.path.basename(os.path.dirname(input_dir))
    if not parent_dir:
        raise ValueError(
            f"Could not extract subject ID from path: {input_dir}"
        )
    return parent_dir


def transcode_with_overlay(
    input_path: str,
    overlay_text: str,
    output_path: str,
    fontfile: Union[str, None],
    target_fps: int = 60,
    crf: int = 20,
    preset: str = "fast",
) -> None:
    """Transcode a single clip with overlaid text to a uniform format for safe concatenation."""
    drawtext_filter: str = build_drawtext_filter(overlay_text, fontfile)
    # Scale first for speed/consistency, then overlay text, finally ensure a widely compatible pixel format
    vf_chain: str = f"scale=-2:720,{drawtext_filter},format=yuv420p"
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf",
        vf_chain,
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-x264-params",
        "keyint=120:min-keyint=120:scenecut=0",
        "-threads",
        "0",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "96k",
        output_path,
    ]
    run_ffmpeg(cmd)


def segment_has_audio(file_path: str) -> bool:
    """Return True if the media file has at least one audio stream."""
    cmd: List[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "csv=p=0",
        file_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return False
    return proc.stdout.decode("utf-8", errors="ignore").strip() != ""


def concat_videos(inputs: List[str], output_path: str) -> None:
    """
    Concatenate videos using ffmpeg concat demuxer.

    If reencode is False, uses stream copy for speed (requires identical codecs and params).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
        for p in inputs:
            list_file.write(f"file '{p}'\n")
        list_path: str = list_file.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            output_path,
        ]
        run_ffmpeg(cmd)
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def build_sequence(
    input_dir: str, seed: Union[int, None]
) -> List[Tuple[str, str, str]]:
    """
    Build the ordered sequence of (path, EmotionName, strength) based on required logic:
    - Emotions in this order: Angry, Sad, Happy, Surprised, Confidence, Confused, Disgust
    - For each emotion: one random 'normal', then one random 'high'
    """
    rng: random.Random = random.Random(seed)

    # Map from display name to token used in filenames
    emotion_order: List[Tuple[str, str]] = [
        ("Angry", "angry"),
        ("Sad", "sad"),
        ("Happy", "happy"),
        ("Surprise", "surprise"),  # use token as label
        ("Confident", "confident"),  # use token as label
        ("Confused", "confused"),
        ("Disgust", "disgust"),
    ]

    strengths: List[str] = ["normal", "high"]

    sequence: List[Tuple[str, str, str]] = []
    for display_name, token in emotion_order:
        for strength in strengths:
            clip_path: str = select_random_clip(input_dir, token, strength, rng)
            sequence.append((clip_path, display_name, strength))
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate emotion clips in a fixed order, overlaying ID/emotion/strength text, using ffmpeg."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help=(
            "Directory containing clips with names like <ID>_frontal_angry_high_001.mov"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/emotions_concat",
        help="Directory to save the concatenated video. Filename is auto-generated as <ID>_emotions_concat.mp4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection",
    )
    # No extra tuning arguments; optimized defaults used for speed.

    args = parser.parse_args()

    input_dir: str = os.path.abspath(args.input_dir)
    subject_id: str = detect_subject_id_from_path(input_dir)
    output_dir: str = os.path.abspath(args.output_dir)
    output_path: str = os.path.join(output_dir, f"{subject_id}_emotions_concat.mp4")

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    fontfile: Union[str, None] = find_default_font()
    if fontfile is None:
        # Proceed without explicit font; ffmpeg may still find a default in some environments
        pass

    tqdm.write("Selecting random clips...")
    sequence: List[Tuple[str, str, str]] = build_sequence(input_dir, args.seed)

    temp_dir: str = tempfile.mkdtemp(prefix=f"{subject_id}_concat_")
    produced_paths: List[str] = []

    try:
        with tqdm(total=len(sequence), desc="Transcoding segments", unit="seg") as pbar:
            for idx, (clip_path, emotion_name, strength) in enumerate(sequence):
                overlay_text: str = f"{subject_id} - {emotion_name} ({strength})"
                seg_out: str = os.path.join(temp_dir, f"seg_{idx:02d}.mp4")
                original_abs: str = os.path.abspath(clip_path)
                original_has_audio: bool = segment_has_audio(original_abs)
                if not original_has_audio:
                    tqdm.write(f"Original: no audio detected in {original_abs}")

                transcode_with_overlay(
                    input_path=clip_path,
                    overlay_text=overlay_text,
                    output_path=seg_out,
                    fontfile=fontfile,
                )
                produced_paths.append(seg_out)
                seg_abs: str = os.path.abspath(seg_out)
                seg_has_audio: bool = segment_has_audio(seg_abs)
                if not seg_has_audio:
                    tqdm.write(
                        f"Segment: no audio detected in {seg_abs} (from {original_abs})"
                    )
                elif not original_has_audio and seg_has_audio:
                    tqdm.write(
                        f"Note: audio detected in segment {seg_abs} but original had no audio ({original_abs})"
                    )
                pbar.set_postfix_str(f"{emotion_name} ({strength})")
                pbar.update(1)

        tqdm.write("Concatenating segments...")
        concat_videos(produced_paths, output_path)
        if segment_has_audio(output_path):
            tqdm.write(f"Final video: audio stream detected in {os.path.abspath(output_path)}")
        else:
            tqdm.write(f"Final video: no audio stream detected in {os.path.abspath(output_path)}")
    finally:
        with tqdm(total=len(produced_paths), desc="Cleaning up", unit="file") as pbar:
            for p in produced_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
                pbar.update(1)
        try:
            os.rmdir(temp_dir)
        except OSError:
            # Directory may not be empty (e.g., if concat failed and list file remained)
            pass

    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()


