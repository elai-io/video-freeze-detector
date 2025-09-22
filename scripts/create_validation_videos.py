import argparse
import os
import sys
import re
import random
import tempfile
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from collections import defaultdict, Counter
import logging

import boto3
from botocore.exceptions import ClientError
import ffmpeg


class VideoMetadata(NamedTuple):
    """Container for parsed video metadata from filename."""
    sex: str
    individual_id: str
    view: str
    emotion: str
    emotion_level: str
    text_id: str
    filename: str
    s3_key: str


class VideoInfo(NamedTuple):
    """Container for video file information including metadata and technical specs."""
    metadata: VideoMetadata
    timecode: str
    duration: float
    fps: float
    width: int
    height: int
    local_path: str


class VideoTriplet(NamedTuple):
    """Container for a matched triplet of videos (left, frontal, right)."""
    left: VideoInfo
    frontal: VideoInfo
    right: VideoInfo
    
    @property
    def emotion_key(self) -> str:
        """Return emotion_emotion_level key for stratification."""
        return f'{self.left.metadata.emotion}_{self.left.metadata.emotion_level}'
    
    @property
    def text_id(self) -> str:
        """Return text_id for this triplet."""
        return self.left.metadata.text_id


def parse_filename(filename: str, s3_key: str) -> Optional[VideoMetadata]:
    """Parse video filename to extract metadata.
    
    Expected format: [SEX][INDIVIDUAL_ID]_[VIEW]_[EMOTION]_[EMOTION_LEVEL]_[TEXT_ID].mp4
    Example: M003_left_angry_high_029.mp4
    
    Args:
        filename: The video filename
        s3_key: Full S3 key path
        
    Returns:
        VideoMetadata object or None if parsing failed
    """
    # Remove file extension
    basename = os.path.splitext(filename)[0]
    
    # Pattern: [SEX][INDIVIDUAL_ID]_[VIEW]_[EMOTION]_[EMOTION_LEVEL]_[TEXT_ID]
    pattern = r'^([MF])(\d{3})_([a-z]+)_([a-z]+)_([a-z]+)_(\d{3})$'
    match = re.match(pattern, basename)
    
    if not match:
        logging.warning(f'Failed to parse filename: {filename}')
        return None
    
    sex, individual_id, view, emotion, emotion_level, text_id = match.groups()
    
    # Validate view
    if view not in ['left', 'frontal', 'right']:
        logging.warning(f'Invalid view "{view}" in filename: {filename}')
        return None
    
    return VideoMetadata(
        sex=sex,
        individual_id=individual_id,
        view=view,
        emotion=emotion,
        emotion_level=emotion_level,
        text_id=text_id,
        filename=filename,
        s3_key=s3_key
    )


def extract_video_info(video_path: str, metadata: VideoMetadata) -> Optional[VideoInfo]:
    """Extract comprehensive video information including timecode and technical specs.
    
    Args:
        video_path: Local path to the video file
        metadata: Parsed metadata from filename
        
    Returns:
        VideoInfo object with all video metadata or None if extraction failed
    """
    try:
        # Probe the video file to get stream information
        probe_data = ffmpeg.probe(video_path)
        
        # Find video stream
        video_stream = None
        for stream in probe_data['streams']:
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            logging.error(f'No video stream found in {video_path}')
            return None
        
        # Extract basic video info
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        duration = float(video_stream.get('duration', 0))
        
        # Calculate FPS
        fps_str = video_stream.get('r_frame_rate', '25/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 25.0
        else:
            fps = float(fps_str)
        
        # Look for timecode in data streams first (tmcd streams)
        timecode = None
        data_streams = [
            stream for stream in probe_data['streams'] 
            if stream.get('codec_tag_string') == 'tmcd'
        ]
        for stream in data_streams:
            tags = stream.get('tags', {})
            if 'timecode' in tags:
                timecode = tags['timecode']
                break
        
        # If no timecode found in data streams, check video stream
        if not timecode:
            tags = video_stream.get('tags', {})
            timecode = tags.get('timecode')
        
        if not timecode:
            logging.error(f'No timecode found in {video_path}')
            return None
        
        return VideoInfo(
            metadata=metadata,
            timecode=timecode,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            local_path=video_path
        )
        
    except ffmpeg.Error as e:
        logging.error(f'FFmpeg error extracting info from {video_path}: {e}')
        return None
    except Exception as e:
        logging.error(f'Unexpected error processing {video_path}: {e}')
        return None


def list_s3_videos(s3_client, bucket_name: str, prefix: str) -> Dict[str, List[VideoMetadata]]:
    """List all videos in S3 directory and organize by view.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        prefix: S3 prefix/directory path
        
    Returns:
        Dictionary with 'left', 'frontal', 'right' keys containing lists of VideoMetadata
    """
    videos_by_view = {'left': [], 'frontal': [], 'right': []}
    
    # Define directory mappings - frontal view can be in either 'frontal' or 'front' directories
    view_directories = {
        'left': ['left'],
        'frontal': ['frontal', 'front'],  # Check both 'frontal' and 'front' directories
        'right': ['right']
    }
    
    for view, directories in view_directories.items():
        for directory in directories:
            view_prefix = f'{prefix.rstrip("/")}/{directory}/'
            logging.info(f'Listing videos in s3://{bucket_name}/{view_prefix}')
            
            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=view_prefix)
                
                found_videos = 0
                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            filename = os.path.basename(key)
                            
                            # Only process .MOV and .mp4 files
                            if not filename.lower().endswith(('.mov', '.mp4')):
                                continue
                            
                            metadata = parse_filename(filename, key)
                            if metadata and metadata.view == view:
                                videos_by_view[view].append(metadata)
                                found_videos += 1
                            elif metadata:
                                logging.warning(f'View mismatch: file {filename} in {directory} directory but parsed as {metadata.view}')
                
                if found_videos > 0:
                    logging.info(f'Found {found_videos} videos in {directory} directory for {view} view')
                elif directory == 'frontal':
                    logging.info(f'No videos found in {directory} directory, will try "front" directory')
                
            except ClientError as e:
                logging.error(f'Error listing S3 objects in {view_prefix}: {e}')
    
    # Log final counts
    logging.info(f'Total videos by view: left={len(videos_by_view["left"])}, frontal={len(videos_by_view["frontal"])}, right={len(videos_by_view["right"])}')
            
    return videos_by_view


def find_video_triplets(videos_by_view: Dict[str, List[VideoMetadata]]) -> List[Tuple[VideoMetadata, VideoMetadata, VideoMetadata]]:
    """Find matching triplets of videos (left, frontal, right) based on metadata.
    
    Args:
        videos_by_view: Dictionary with video lists organized by view
        
    Returns:
        List of tuples (left_video, frontal_video, right_video)
    """
    # Create lookup dictionaries by unique identifier (excluding view)
    def make_key(meta: VideoMetadata) -> str:
        return f'{meta.sex}{meta.individual_id}_{meta.emotion}_{meta.emotion_level}_{meta.text_id}'
    
    left_lookup = {make_key(v): v for v in videos_by_view['left']}
    frontal_lookup = {make_key(v): v for v in videos_by_view['frontal']}
    right_lookup = {make_key(v): v for v in videos_by_view['right']}
    
    # Find intersection - videos that exist in all three views
    all_keys = set(left_lookup.keys()) & set(frontal_lookup.keys()) & set(right_lookup.keys())
    
    triplets = []
    for key in all_keys:
        triplet = (left_lookup[key], frontal_lookup[key], right_lookup[key])
        triplets.append(triplet)
    
    logging.info(f'Found {len(triplets)} complete video triplets out of {len(left_lookup)} left, {len(frontal_lookup)} frontal, {len(right_lookup)} right videos')
    
    return triplets


def stratified_sample_triplets(triplets: List[Tuple[VideoMetadata, VideoMetadata, VideoMetadata]], 
                               n_samples: int, 
                               n_team_members: int) -> Dict[str, List[Tuple[VideoMetadata, VideoMetadata, VideoMetadata]]]:
    """Perform stratified sampling to ensure balanced emotion/emotion_level distribution.
    
    Args:
        triplets: List of video triplets
        n_samples: Number of samples per emotion-emotion_level pair per team member
        n_team_members: Number of team members
        
    Returns:
        Dictionary mapping emotion_emotion_level -> list of triplets
    """
    # Group triplets by emotion-emotion_level
    triplets_by_emotion = defaultdict(list)
    for triplet in triplets:
        left, frontal, right = triplet
        emotion_key = f'{left.emotion}_{left.emotion_level}'
        triplets_by_emotion[emotion_key].append(triplet)
    
    # Log distribution
    logging.info('Triplet distribution by emotion-emotion_level:')
    for emotion_key, triplet_list in triplets_by_emotion.items():
        logging.info(f'  {emotion_key}: {len(triplet_list)} triplets')
    
    # Calculate required samples
    total_samples_needed = n_samples * n_team_members
    
    # Sample from each emotion group
    sampled_by_emotion = {}
    for emotion_key, triplet_list in triplets_by_emotion.items():
        available = len(triplet_list)
        needed = total_samples_needed
        
        if available < needed:
            logging.warning(f'Not enough triplets for {emotion_key}: need {needed}, have {available}')
            needed = available
        
        # Randomly sample without replacement
        sampled = random.sample(triplet_list, needed)
        sampled_by_emotion[emotion_key] = sampled
        logging.info(f'Sampled {len(sampled)} triplets for {emotion_key}')
    
    return sampled_by_emotion


def download_s3_file(s3_client, bucket_name: str, s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to local path.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        s3_key: S3 object key
        local_path: Local file path to download to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        logging.error(f'Error downloading s3://{bucket_name}/{s3_key} to {local_path}: {e}')
        return False
    except Exception as e:
        logging.error(f'Unexpected error downloading s3://{bucket_name}/{s3_key}: {e}')
        return False


def timecode_to_seconds(timecode: str, fps: float) -> float:
    """Convert timecode to seconds for precise alignment.
    
    Args:
        timecode: Timecode in HH:MM:SS:FF format
        fps: Frames per second
        
    Returns:
        Time in seconds as float
    """
    try:
        parts = timecode.split(':')
        if len(parts) != 4:
            return 0.0
        hours, minutes, seconds, frames = map(int, parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds + frames / fps
        return total_seconds
    except ValueError:
        return 0.0


def calculate_alignment_params(left_info: VideoInfo, 
                               frontal_info: VideoInfo, 
                               right_info: VideoInfo) -> Tuple[float, float, float, float]:
    """Calculate alignment parameters for three videos.
    
    Args:
        left_info: Left camera video info
        frontal_info: Frontal camera video info  
        right_info: Right camera video info
        
    Returns:
        Tuple of (start_offset_left, start_offset_frontal, start_offset_right, aligned_duration)
    """
    # Convert timecodes to seconds using the actual fps of each video
    left_start = timecode_to_seconds(left_info.timecode, left_info.fps)
    frontal_start = timecode_to_seconds(frontal_info.timecode, frontal_info.fps)
    right_start = timecode_to_seconds(right_info.timecode, right_info.fps)
    
    # Find the latest start time (all videos must start from this point)
    latest_start = max(left_start, frontal_start, right_start)
    
    # Calculate how much to trim from the beginning of each video
    left_trim = latest_start - left_start
    frontal_trim = latest_start - frontal_start  
    right_trim = latest_start - right_start
    
    # Calculate remaining duration for each video after trimming the start
    left_remaining = left_info.duration - left_trim
    frontal_remaining = frontal_info.duration - frontal_trim
    right_remaining = right_info.duration - right_trim
    
    # Final duration is the shortest remaining duration
    aligned_duration = min(left_remaining, frontal_remaining, right_remaining)
    
    return left_trim, frontal_trim, right_trim, aligned_duration


def process_and_concatenate_triplet(left_info: VideoInfo,
                                    frontal_info: VideoInfo, 
                                    right_info: VideoInfo,
                                    output_path: str,
                                    fps: float = 25.0) -> bool:
    """Process and concatenate a video triplet into a single side-by-side video.
    
    Scales each video to 854x480 (480p 16:9) and concatenates them horizontally
    to create a 2562x480 output video. Includes aligned audio from the first 
    available source in priority order: frontal > left > right.
    
    Args:
        left_info: Left camera video info
        frontal_info: Frontal camera video info
        right_info: Right camera video info
        output_path: Path for output concatenated video
        fps: Target fps for output (default: 25.0)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate alignment parameters
        left_trim, frontal_trim, right_trim, duration = calculate_alignment_params(
            left_info, frontal_info, right_info
        )
        
        if duration < 1.0:
            logging.warning(f'Aligned duration too short ({duration:.3f}s), skipping')
            return False
        
        # Create aligned inputs
        left_input = ffmpeg.input(left_info.local_path, ss=left_trim, t=duration)
        frontal_input = ffmpeg.input(frontal_info.local_path, ss=frontal_trim, t=duration)
        right_input = ffmpeg.input(right_info.local_path, ss=right_trim, t=duration)
        
        # Scale videos to 480p 16:9 (854x480) for each view
        # Final concatenated video will be 2562x480 (854*3 x 480)
        target_width = 854
        target_height = 480
        
        left_scaled = ffmpeg.filter(left_input, 'scale', target_width, target_height)
        frontal_scaled = ffmpeg.filter(frontal_input, 'scale', target_width, target_height)
        right_scaled = ffmpeg.filter(right_input, 'scale', target_width, target_height)
        
        # Concatenate horizontally (side by side)
        concatenated_video = ffmpeg.filter([left_scaled, frontal_scaled, right_scaled], 'hstack', inputs=3)

        # Add filename overlay at the top
        filename = f'{left_info.metadata.sex}{left_info.metadata.individual_id}_{left_info.metadata.emotion}_{left_info.metadata.emotion_level}_{left_info.metadata.text_id}'
        concatenated_video = ffmpeg.filter(
            concatenated_video,
            'drawtext',
            text=filename,
            fontsize=24,
            fontcolor='white',
            x='(w-text_w)/2',  # Center horizontally
            y='20',            # 20 pixels from top
            box='1',
            boxcolor='black@0.7',
            boxborderw='6'
        )
        
        # Handle audio - prefer frontal audio first, then left, then right
        audio_source = None
        audio_priority = [
            (frontal_info.local_path, frontal_trim, 'frontal'),
            (left_info.local_path, left_trim, 'left'),
            (right_info.local_path, right_trim, 'right')
        ]
        
        for video_path, trim_offset, view_name in audio_priority:
            try:
                # Check if this video has an audio stream
                probe_data = ffmpeg.probe(video_path)
                has_audio = any(stream.get('codec_type') == 'audio' for stream in probe_data['streams'])
                
                if has_audio:
                    audio_source = ffmpeg.input(video_path, ss=trim_offset, t=duration).audio
                    logging.info(f'Using audio from {view_name} view')
                    break
                    
            except Exception as e:
                logging.debug(f'No audio found in {view_name} view: {e}')
                continue
        
        # Configure output with NVIDIA hardware encoder
        if audio_source:
            output = ffmpeg.output(
                concatenated_video,
                audio_source,
                output_path,
                vcodec='h264_nvenc',
                preset='fast',
                crf=28,
                r=fps,
                acodec='aac',
                audio_bitrate='128k',
                loglevel='error'
            )
        else:
            logging.warning('No audio track found in any of the three videos, creating video-only output')
            output = ffmpeg.output(
                concatenated_video,
                output_path,
                vcodec='h264_nvenc',
                preset='fast',
                crf=28,
                r=fps,
                loglevel='error'
            )
        
        # Run ffmpeg
        ffmpeg.run(output, overwrite_output=True, quiet=False)
        return True
        
    except ffmpeg.Error as e:
        logging.error(f'FFmpeg error processing triplet: {e}')
        return False
    except Exception as e:
        logging.error(f'Unexpected error processing triplet: {e}')
        return False


def upload_to_s3(s3_client, local_path: str, bucket_name: str, s3_key: str) -> bool:
    """Upload a local file to S3.

    Args:
        s3_client: Boto3 S3 client
        local_path: Local file path
        bucket_name: S3 bucket name
        s3_key: S3 object key for upload

    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        # Extract teammate and actor_id from path for cleaner logging
        path_parts = s3_key.split('/')
        teammate = path_parts[-3] if len(path_parts) >= 3 else 'unknown'
        actor_id = path_parts[-2] if len(path_parts) >= 2 else 'unknown'
        filename = path_parts[-1]
        logging.info(f'Uploaded {filename} to s3://{bucket_name}/{teammate}/{actor_id}/')
        return True
    except ClientError as e:
        logging.error(f'Error uploading {local_path} to S3: {e}')
        return False
    except Exception as e:
        logging.error(f'Unexpected error uploading {local_path}: {e}')
        return False


def process_single_triplet(triplet_data: Tuple) -> Tuple[bool, str, str]:
    """Process a single video triplet (for parallel execution).

    Args:
        triplet_data: Tuple containing (left_meta, frontal_meta, right_meta, temp_dir, input_bucket, output_bucket, output_prefix, member, fps, aws_profile)

    Returns:
        Tuple of (success, text_id, error_message)
    """
    left_meta, frontal_meta, right_meta, temp_dir, input_bucket, output_bucket, output_prefix, member, fps, aws_profile = triplet_data
    text_id = left_meta.text_id

    # Create temporary directory for this triplet
    triplet_temp_dir = os.path.join(temp_dir, f'triplet_{text_id}_{random.randint(1000, 9999)}')
    os.makedirs(triplet_temp_dir, exist_ok=True)

    try:
        # Setup S3 client for this process
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                s3_client = session.client('s3')
            else:
                s3_client = boto3.client('s3')
        except Exception as e:
            return False, text_id, f'Error creating S3 client: {e}'

        # Download videos
        left_local = os.path.join(triplet_temp_dir, f'left_{text_id}.mov')
        frontal_local = os.path.join(triplet_temp_dir, f'frontal_{text_id}.mov')
        right_local = os.path.join(triplet_temp_dir, f'right_{text_id}.mov')

        downloads_successful = True
        downloads_successful &= download_s3_file(s3_client, input_bucket, left_meta.s3_key, left_local)
        downloads_successful &= download_s3_file(s3_client, input_bucket, frontal_meta.s3_key, frontal_local)
        downloads_successful &= download_s3_file(s3_client, input_bucket, right_meta.s3_key, right_local)

        if not downloads_successful:
            return False, text_id, 'Failed to download videos'

        # Extract video information
        left_info = extract_video_info(left_local, left_meta)
        frontal_info = extract_video_info(frontal_local, frontal_meta)
        right_info = extract_video_info(right_local, right_meta)

        if not (left_info and frontal_info and right_info):
            return False, text_id, 'Failed to extract video info'

        # Generate output filename and path
        output_filename = f'{left_meta.sex}{left_meta.individual_id}_{left_meta.emotion}_{left_meta.emotion_level}_{text_id}.mp4'
        output_local_path = os.path.join(triplet_temp_dir, output_filename)

        # Process and concatenate the triplet
        if not process_and_concatenate_triplet(
            left_info, frontal_info, right_info,
            output_local_path, fps
        ):
            return False, text_id, 'Failed to process triplet'

        # Upload to S3 (organized by teammate/actor_id/)
        actor_id = left_meta.individual_id
        output_s3_key = f'{output_prefix.rstrip("/")}/{member}/{actor_id}/{output_filename}'
        if not upload_to_s3(s3_client, output_local_path, output_bucket, output_s3_key):
            return False, text_id, 'Failed to upload result'

        return True, text_id, ''

    except Exception as e:
        return False, text_id, f'Unexpected error: {e}'
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(triplet_temp_dir)
        except Exception as e:
            logging.warning(f'Failed to clean up {triplet_temp_dir}: {e}')


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='Validate dataset by sampling, aligning, and distributing videos among team members. Output structure: {teammate}/{actor_id}/{video_files}'
    )
    parser.add_argument(
        's3_input_directory',
        help='S3 directory containing frontal/, left/, right/ subdirectories with .MOV files'
    )
    parser.add_argument(
        's3_output_directory',
        help='Base S3 directory for output (will create {teammate}/{actor_id}/ subdirectories)'
    )
    parser.add_argument(
        '--team_members',
        nargs='+',
        required=True,
        help='List of team member names (e.g., --team_members leonardo donatello raphael michelangelo)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='Number of TEXT_IDs to sample for each emotion-emotion_level pair (minimum 1, default 1)'
    )
    parser.add_argument(
        '--aws_profile',
        help='AWS profile name to use for S3 operations'
    )
    parser.add_argument(
        '--temp_dir',
        help='Local temporary directory for video processing (default: system temp)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=25.0,
        help='Target FPS for output videos (default: 25.0)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        help='Maximum number of parallel workers (default: min(CPU cores, 8))'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.n_samples < 1:
        parser.error('n_samples must be at least 1')
    
    setup_logging(args.debug)
    
    logging.info('Starting dataset validation script')
    logging.info(f'Input S3 directory: {args.s3_input_directory}')
    logging.info(f'Output S3 directory: {args.s3_output_directory}')
    logging.info(f'Team members: {args.team_members}')
    logging.info(f'Samples per emotion-level: {args.n_samples}')
    
    # Parse S3 paths
    input_parts = args.s3_input_directory.replace('s3://', '').split('/', 1)
    if len(input_parts) != 2:
        parser.error('s3_input_directory must be in format s3://bucket/prefix')
    input_bucket, input_prefix = input_parts
    
    output_parts = args.s3_output_directory.replace('s3://', '').split('/', 1)
    if len(output_parts) != 2:
        parser.error('s3_output_directory must be in format s3://bucket/prefix')
    output_bucket, output_prefix = output_parts
    
    # Setup S3 client
    try:
        if args.aws_profile:
            session = boto3.Session(profile_name=args.aws_profile)
            s3_client = session.client('s3')
        else:
            s3_client = boto3.client('s3')
    except Exception as e:
        logging.error(f'Error creating S3 client: {e}')
        sys.exit(1)
    
    # Setup temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix='validation_')
    
    try:
        # Step 1: List all videos in S3
        logging.info('Step 1: Listing videos from S3...')
        videos_by_view = list_s3_videos(s3_client, input_bucket, input_prefix)
        
        total_videos = sum(len(videos) for videos in videos_by_view.values())
        if total_videos == 0:
            logging.error('No videos found in S3 directory')
            sys.exit(1)
        
        # Step 2: Find complete triplets
        logging.info('Step 2: Finding complete video triplets...')
        triplets = find_video_triplets(videos_by_view)
        
        if not triplets:
            logging.error('No complete video triplets found')
            sys.exit(1)
        
        # Step 3: Perform stratified sampling
        logging.info('Step 3: Performing stratified sampling...')
        sampled_by_emotion = stratified_sample_triplets(
            triplets, args.n_samples, len(args.team_members)
        )
        
        # Flatten sampled triplets
        all_sampled_triplets = []
        for emotion_triplets in sampled_by_emotion.values():
            all_sampled_triplets.extend(emotion_triplets)
        
        if not all_sampled_triplets:
            logging.error('No triplets selected after sampling')
            sys.exit(1)
        
        logging.info(f'Selected {len(all_sampled_triplets)} total triplets for processing')
        
        # Step 4: Distribute triplets among team members
        logging.info('Step 4: Distributing triplets among team members...')
        random.shuffle(all_sampled_triplets)
        
        # Create balanced distribution
        team_distributions = {member: [] for member in args.team_members}
        
        # Distribute triplets by emotion to ensure each team member gets balanced samples
        for emotion_key, emotion_triplets in sampled_by_emotion.items():
            random.shuffle(emotion_triplets)
            
            # Distribute this emotion's triplets evenly among team members
            triplets_per_member = len(emotion_triplets) // len(args.team_members)
            remainder = len(emotion_triplets) % len(args.team_members)
            
            start_idx = 0
            for i, member in enumerate(args.team_members):
                # Give some members one extra triplet if there's a remainder
                count = triplets_per_member + (1 if i < remainder else 0)
                end_idx = start_idx + count
                team_distributions[member].extend(emotion_triplets[start_idx:end_idx])
                start_idx = end_idx
        
        # Log distribution
        logging.info('Team member distribution:')
        for member, member_triplets in team_distributions.items():
            emotion_counts = defaultdict(int)
            for triplet in member_triplets:
                left, _, _ = triplet
                emotion_key = f'{left.emotion}_{left.emotion_level}'
                emotion_counts[emotion_key] += 1
            
            logging.info(f'  {member}: {len(member_triplets)} triplets')
            for emotion_key, count in emotion_counts.items():
                logging.info(f'    {emotion_key}: {count}')
        
        # Step 5: Process videos for each team member (parallel processing)
        logging.info('Step 5: Processing videos for each team member...')

        # Determine optimal number of workers (limit to avoid overwhelming system)
        if args.max_workers:
            max_workers = args.max_workers
        else:
            max_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers or CPU count, whichever is smaller
        logging.info(f'Using {max_workers} parallel workers for processing')

        for member_idx, (member, member_triplets) in enumerate(team_distributions.items(), 1):
            if not member_triplets:
                logging.warning(f'No triplets assigned to {member}, skipping')
                continue

            logging.info(f'Processing {len(member_triplets)} triplets for {member} ({member_idx}/{len(args.team_members)})...')

            # Prepare triplet data for parallel processing
            triplet_tasks = []
            for left_meta, frontal_meta, right_meta in member_triplets:
                triplet_tasks.append((
                    left_meta, frontal_meta, right_meta,
                    temp_dir, input_bucket, output_bucket, output_prefix,
                    member, args.fps, args.aws_profile
                ))

            # Process triplets in parallel
            successful_videos = 0
            failed_videos = 0

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_triplet = {
                    executor.submit(process_single_triplet, triplet_data): triplet_data
                    for triplet_data in triplet_tasks
                }

                # Process completed tasks
                for future in as_completed(future_to_triplet):
                    triplet_data = future_to_triplet[future]
                    left_meta = triplet_data[0]
                    text_id = left_meta.text_id

                    try:
                        success, processed_text_id, error_msg = future.result()
                        if success:
                            successful_videos += 1
                            logging.info(f'  ✓ Successfully processed text_id {processed_text_id} for {member}')
                        else:
                            failed_videos += 1
                            logging.error(f'  ✗ Failed to process text_id {processed_text_id} for {member}: {error_msg}')
                    except Exception as e:
                        failed_videos += 1
                        logging.error(f'  ✗ Exception processing text_id {text_id} for {member}: {e}')

            logging.info(f'Completed processing for {member}: {successful_videos}/{len(member_triplets)} successful, {failed_videos} failed')
    
    finally:
        # Clean up temporary directory
        if not args.temp_dir:  # Only clean up if we created it
            try:
                shutil.rmtree(temp_dir)
                logging.info(f'Cleaned up temporary directory: {temp_dir}')
            except Exception as e:
                logging.warning(f'Failed to clean up temporary directory {temp_dir}: {e}')
    
    logging.info('Dataset validation completed successfully!')
    logging.info(f'Results uploaded to: s3://{output_bucket}/{output_prefix}')
    logging.info('Directory structure: {teammate}/{actor_id}/{video_files}')

    # Log final summary
    total_processed = sum(len(triplets) for triplets in team_distributions.values())
    logging.info(f'Summary: Processed {total_processed} triplets distributed among {len(args.team_members)} team members')


if __name__ == '__main__':
    main()
