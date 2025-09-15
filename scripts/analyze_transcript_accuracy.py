import os
import re
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from numpy.typing import NDArray
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, stripping whitespace,
    and removing all non-alphabetic characters.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string containing only lowercase letters and spaces
    """
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    
    # Remove all non-alphabetic characters except spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Replace multiple spaces with single space and strip again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_edit_distance(text1: str, text2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Edit distance as integer
    """
    if not text1:
        return len(text2)
    if not text2:
        return len(text1)
    
    # Create distance matrix
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                 dp[i][j-1],        # insertion
                                 dp[i-1][j-1])      # substitution
    
    return dp[m][n]


def calculate_word_distance(text1: str, text2: str) -> float:
    """
    Calculate word-level distance using simple word overlap.
    Since Whisper deals with complete words, we don't need complex frequency weighting.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Word distance as float (0 = identical words, 1 = no overlap)
    """
    if not text1 and not text2:
        return 0.0
    if not text1 or not text2:
        return 1.0
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        return 0.0
    if not words1 or not words2:
        return 1.0
    
    # Calculate Jaccard distance (1 - Jaccard similarity)
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


def calculate_combined_distance(text1: str, text2: str) -> float:
    """
    Calculate a combined distance metric that handles length differences properly.
    
    This combines:
    1. Normalized edit distance (handles character-level differences)
    2. Word overlap distance (handles vocabulary differences) 
    3. Length penalty (explicitly penalizes length mismatches)
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Combined distance as float (0 = identical, higher = more different)
    """
    if not text1 and not text2:
        return 0.0
    if not text1 or not text2:
        return 1.0
    
    # 1. Normalized edit distance (0 to 1)
    edit_dist = calculate_edit_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    normalized_edit = edit_dist / max_len if max_len > 0 else 0.0
    
    # 2. Word-level distance (0 to 1, based on word overlap)
    word_dist = calculate_word_distance(text1, text2)
    
    # 3. Length ratio penalty (0 to 1)
    len1, len2 = len(text1), len(text2)
    if len1 == 0 and len2 == 0:
        length_penalty = 0.0
    elif len1 == 0 or len2 == 0:
        length_penalty = 1.0
    else:
        length_ratio = min(len1, len2) / max(len1, len2)
        length_penalty = 1.0 - length_ratio
    
    # Combine metrics with weights
    # Edit distance gets highest weight (most reliable)
    # Word distance helps with semantic content
    # Length penalty ensures length differences are penalized
    combined_distance = (
        0.6 * normalized_edit +     # Primary metric
        0.3 * word_dist +          # Secondary semantic metric  
        0.1 * length_penalty       # Length consideration
    )
    
    return float(combined_distance)


def extract_text_id_from_filename(filename: str) -> str:
    """
    Extract text ID from transcript filename.
    Expected format: *_[TEXT_ID].txt
    
    Args:
        filename: Transcript filename
        
    Returns:
        Extracted text ID as string
    """
    # Remove .txt extension and get the last part after underscore
    stem = Path(filename).stem
    parts = stem.split('_')
    
    # Look for 3-digit number at the end (TEXT_ID format)
    for part in reversed(parts):
        if re.match(r'^\d{3}$', part):
            return part
    
    # Fallback: return last part
    return parts[-1] if parts else stem


def extract_identity_from_filename(filename: str) -> str:
    """
    Extract identity (SEX + INDIVIDUAL_ID) from transcript filename.
    Expected format: [SEX][INDIVIDUAL_ID]_[VIEW]_[EMOTION]_[EMOTION_LEVEL]_[TEXT_ID].txt
    
    Args:
        filename: Transcript filename
        
    Returns:
        Extracted identity as string (e.g., 'M003', 'F015')
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if parts:
        # Look for pattern like M003, F015, etc.
        identity_match = re.match(r'^([MF]\d{3})$', parts[0])
        if identity_match:
            return identity_match.group(1)
    
    # Fallback: return first part
    return parts[0] if parts else stem


def load_ground_truth_texts(texts_path: str) -> Dict[str, str]:
    """
    Load ground truth sentences from text file.
    Expected format: one sentence per line, no markup.
    Line numbers become text IDs (001, 002, etc.)
    
    Args:
        texts_path: Path to ground truth text file
        
    Returns:
        Dictionary mapping text IDs to normalized sentences
    """
    texts = {}
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Use line number as text ID (1-indexed, padded to 3 digits)
            text_id = str(line_num).zfill(3)
            texts[text_id] = normalize_text(line)
    
    return texts


def process_single_transcript(transcript_file: str, ground_truth: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a single transcript file and calculate distances to all ground truth sentences.
    This function is designed to be used with multiprocessing.
    
    Args:
        transcript_file: Path to transcript file
        ground_truth: Dictionary of ground truth sentences
        
    Returns:
        Dictionary containing all processing results for this file
    """
    filename = Path(transcript_file).name
    
    # Extract expected text ID and identity from filename
    expected_id = extract_text_id_from_filename(filename)
    identity = extract_identity_from_filename(filename)
    
    # Load and normalize transcript
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_text = normalize_text(f.read())
    
    # Calculate distance to all ground truth sentences
    distances = {}
    best_match_id = None
    best_distance = float('inf')
    
    for text_id, ground_truth_text in ground_truth.items():
        distance = calculate_combined_distance(transcript_text, ground_truth_text)
        distances[text_id] = distance
        
        if distance < best_distance:
            best_distance = distance
            best_match_id = text_id
    
    expected_distance = distances.get(expected_id, float('inf'))
    
    # Calculate ranking of expected match
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    expected_rank = next((i+1 for i, (tid, _) in enumerate(sorted_distances) if tid == expected_id), len(sorted_distances))
    
    # Determine if this is a mismatch
    is_mismatch = best_match_id != expected_id
    
    result = {
        'filename': filename,
        'expected_id': expected_id,
        'best_match_id': best_match_id,
        'best_distance': best_distance,
        'expected_distance': expected_distance,
        'transcript': transcript_text,
        'distances': distances,
        'identity': identity,
        'expected_rank': expected_rank,
        'is_mismatch': is_mismatch
    }
    
    # Add mismatch details if needed
    if is_mismatch:
        result['expected_text'] = ground_truth.get(expected_id, 'N/A')
        result['best_match_text'] = ground_truth.get(best_match_id, 'N/A')
    
    return result


def analyze_transcripts(transcripts_path: str, texts_path: str, output_dir: str, num_workers: int = None) -> None:
    """
    Main analysis function that processes all transcripts and generates reports.
    
    Args:
        transcripts_path: Glob pattern for transcript files
        texts_path: Path to ground truth text file
        output_dir: Output directory for results
        num_workers: Number of worker processes (None = auto-detect CPU count)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('Loading ground truth texts...')
    ground_truth = load_ground_truth_texts(texts_path)
    print(f'Loaded {len(ground_truth)} ground truth sentences')
    
    print('Finding transcript files...')
    transcript_files = glob.glob(transcripts_path)
    print(f'Found {len(transcript_files)} transcript files')
    
    if not transcript_files:
        print(f'No files found matching pattern: {transcripts_path}')
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(transcript_files))
    print(f'Using {num_workers} worker processes')
    
    # Process transcripts using multiprocessing
    print('Processing transcripts...')
    worker_func = partial(process_single_transcript, ground_truth=ground_truth)
    
    if num_workers == 1:
        # Single-threaded processing (useful for debugging)
        all_results = []
        for transcript_file in tqdm(transcript_files, desc='Processing transcripts'):
            result = worker_func(transcript_file)
            all_results.append(result)
    else:
        # Multi-threaded processing
        with mp.Pool(processes=num_workers) as pool:
            all_results = list(tqdm(
                pool.imap(worker_func, transcript_files),
                total=len(transcript_files),
                desc='Processing transcripts'
            ))
    
    # Aggregate results
    results = {}
    problematic_files = []
    correct_matches = 0
    total_files = len(transcript_files)
    identity_stats = {}
    
    print('Aggregating results...')
    for result in tqdm(all_results, desc='Aggregating results'):
        filename = result['filename']
        identity = result['identity']
        expected_distance = result['expected_distance']
        best_distance = result['best_distance']
        
        # Initialize identity stats if needed
        if identity not in identity_stats:
            identity_stats[identity] = {
                'total_files': 0,
                'correct_matches': 0,
                'problematic_files': [],
                'avg_expected_distance': 0.0,
                'avg_best_distance': 0.0
            }
        
        # Update identity statistics
        identity_stats[identity]['total_files'] += 1
        identity_stats[identity]['avg_expected_distance'] += expected_distance
        identity_stats[identity]['avg_best_distance'] += best_distance
        
        # Store results
        results[filename] = {
            'expected_id': result['expected_id'],
            'best_match_id': result['best_match_id'],
            'best_distance': best_distance,
            'expected_distance': expected_distance,
            'transcript': result['transcript'],
            'distances': result['distances'],
            'identity': identity
        }
        
        # Check if match is correct
        if result['is_mismatch']:
            # Print mismatch information
            expected_text = result['expected_text']
            best_match_text = result['best_match_text']
            expected_rank = result['expected_rank']
            
            print(f'⚠️  MISMATCH: {filename}')
            print(f'    🎯 Expected ({int(result["expected_id"]):03d}):   "{expected_text}"')
            print(f'    🤖 Transcript:       "{result["transcript"]}"') 
            print(f'    ✅ Best match ({int(result["best_match_id"]):03d}): "{best_match_text}"')
            print(f'    📊 Expected distance: {expected_distance:.4f} (rank: {expected_rank}/{len(result["distances"])})')
            print(f'    📊 Best distance:     {best_distance:.4f}')
            print(f'    💡 Improvement:       {expected_distance - best_distance:.4f}')
            print('')
            
            problematic_file_data = {
                'filename': filename,
                'expected_id': result['expected_id'],
                'expected_text': expected_text,
                'best_match_id': result['best_match_id'],
                'best_match_text': best_match_text,
                'transcript': result['transcript'],
                'expected_distance': expected_distance,
                'best_distance': best_distance,
                'expected_rank': expected_rank,
                'identity': identity
            }
            
            problematic_files.append(problematic_file_data)
            identity_stats[identity]['problematic_files'].append(problematic_file_data)
        else:
            correct_matches += 1
            identity_stats[identity]['correct_matches'] += 1
    
    # Finalize identity statistics (compute averages)
    for identity, stats in identity_stats.items():
        if stats['total_files'] > 0:
            stats['avg_expected_distance'] /= stats['total_files']
            stats['avg_best_distance'] /= stats['total_files']
            stats['accuracy'] = (stats['correct_matches'] / stats['total_files']) * 100
    
    # Generate reports
    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    
    # Overall statistics
    accuracy = (correct_matches / total_files) * 100
    print(f'Total files processed: {total_files}')
    print(f'Correct matches: {correct_matches}')
    print(f'Problematic files: {len(problematic_files)}')
    print(f'Overall accuracy: {accuracy:.2f}%')
    print(f'Unique identities: {len(identity_stats)}')
    
    # Identity-based statistics
    print('\n' + '='*80)
    print('IDENTITY-BASED STATISTICS')
    print('='*80)
    
    # Sort identities by accuracy (worst first for attention)
    sorted_identities = sorted(identity_stats.items(), key=lambda x: x[1]['accuracy'])
    
    for identity, stats in sorted_identities:
        problematic_count = len(stats['problematic_files'])
        print(f"\n👤 {identity}:")
        print(f"    📊 Files: {stats['total_files']}, Correct: {stats['correct_matches']}, Problematic: {problematic_count}")
        print(f"    🎯 Accuracy: {stats['accuracy']:.2f}%")
        print(f"    📏 Avg expected distance: {stats['avg_expected_distance']:.4f}")
        print(f"    📏 Avg best distance: {stats['avg_best_distance']:.4f}")
        
        if problematic_count > 0:
            print(f"    ⚠️  Worst files: {', '.join([pf['filename'] for pf in stats['problematic_files'][:3]])}")
            if problematic_count > 3:
                print(f"        ... and {problematic_count - 3} more")
    
    # Summary statistics by sex
    male_stats = {'total': 0, 'correct': 0, 'problematic': 0}
    female_stats = {'total': 0, 'correct': 0, 'problematic': 0}
    
    for identity, stats in identity_stats.items():
        if identity.startswith('M'):
            male_stats['total'] += stats['total_files']
            male_stats['correct'] += stats['correct_matches']
            male_stats['problematic'] += len(stats['problematic_files'])
        elif identity.startswith('F'):
            female_stats['total'] += stats['total_files']
            female_stats['correct'] += stats['correct_matches']
            female_stats['problematic'] += len(stats['problematic_files'])
    
    print(f"\n📊 SUMMARY BY SEX:")
    if male_stats['total'] > 0:
        male_accuracy = (male_stats['correct'] / male_stats['total']) * 100
        print(f"    👨 Males: {male_stats['total']} files, {male_accuracy:.2f}% accuracy, {male_stats['problematic']} problematic")
    
    if female_stats['total'] > 0:
        female_accuracy = (female_stats['correct'] / female_stats['total']) * 100
        print(f"    👩 Females: {female_stats['total']} files, {female_accuracy:.2f}% accuracy, {female_stats['problematic']} problematic")
    
    # Save detailed results
    results_path = output_dir / 'detailed_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nDetailed results saved to: {results_path}')
    
    # Save problematic files as JSON
    problematic_files_json_path = output_dir / 'problematic_files.json'
    with open(problematic_files_json_path, 'w', encoding='utf-8') as f:
        json.dump(problematic_files, f, indent=2, ensure_ascii=False)
    print(f'Problematic files JSON saved to: {problematic_files_json_path}')
    
    # Save identity statistics
    identity_stats_path = output_dir / 'identity_statistics.json'
    # Create a clean version for JSON serialization
    identity_stats_clean = {}
    for identity, stats in identity_stats.items():
        identity_stats_clean[identity] = {
            'total_files': stats['total_files'],
            'correct_matches': stats['correct_matches'],
            'accuracy': stats['accuracy'],
            'avg_expected_distance': stats['avg_expected_distance'],
            'avg_best_distance': stats['avg_best_distance'],
            'problematic_filenames': [pf['filename'] for pf in stats['problematic_files']]
        }
    
    with open(identity_stats_path, 'w', encoding='utf-8') as f:
        json.dump(identity_stats_clean, f, indent=2, ensure_ascii=False)
    print(f'Identity statistics saved to: {identity_stats_path}')
    
    # Generate proposed re-indexing
    proposed_reindex = {}
    for item in problematic_files:
        # Remove .txt extension from filename
        base_filename = item['filename'].rsplit('.', 1)[0]

        # The filename already contains the text ID, so we need to replace it
        # Split on '_' and replace the last part (the text ID)
        parts = base_filename.rsplit('_', 1)
        if len(parts) == 2:
            # Replace the text ID part with the best match ID
            old_key = base_filename
            new_key = f"{parts[0]}_{item['best_match_id']}"
            proposed_reindex[old_key] = new_key
    
    reindex_path = output_dir / 'proposed_reindexing.json'
    with open(reindex_path, 'w', encoding='utf-8') as f:
        json.dump(proposed_reindex, f, indent=2, ensure_ascii=False)
    print(f'Proposed re-indexing saved to: {reindex_path}')
    
    # Create detailed problematic files report
    if problematic_files:
        print('\n' + '='*80)
        print('PROBLEMATIC FILES REPORT')
        print('='*80)
        
        report_lines = []
        for item in problematic_files:
            report_lines.append(f"📁 File: {item['filename']} ({item['identity']})")
            report_lines.append(f"🎯 Expected ({int(item['expected_id']):03d}):   \"{item['expected_text']}\"")
            report_lines.append(f"🤖 Transcript:       \"{item['transcript']}\"")
            report_lines.append(f"✅ Best match ({int(item['best_match_id']):03d}): \"{item['best_match_text']}\"")
            report_lines.append(f"📊 Expected distance: {item['expected_distance']:.4f} (rank: {item['expected_rank']}/{len(ground_truth)})")
            report_lines.append(f"📊 Best distance:     {item['best_distance']:.4f}")
            report_lines.append(f"💡 Improvement:       {item['expected_distance'] - item['best_distance']:.4f}")
            report_lines.append("-" * 80)
            report_lines.append("")
        
        report_path = output_dir / 'problematic_files_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f'Detailed problematic files report saved to: {report_path}')
        
        # Display first few problematic files
        print('\nFirst 3 problematic files:')
        for i, item in enumerate(problematic_files[:3], 1):
            print(f"\n📁 File {i}: {item['filename']} ({item['identity']})")
            print(f"    🎯 Expected ({int(item['expected_id']):03d}):   \"{item['expected_text']}\"")
            print(f"    🤖 Transcript:       \"{item['transcript']}\"")
            print(f"    ✅ Best match ({int(item['best_match_id']):03d}): \"{item['best_match_text']}\"")
            print(f"    📊 Expected distance: {item['expected_distance']:.4f} (rank: {item['expected_rank']}/{len(ground_truth)})")
            print(f"    📊 Best distance:     {item['best_distance']:.4f}")
            print(f"    💡 Improvement:       {item['expected_distance'] - item['best_distance']:.4f}")
    
    print(f'\n📁 All results saved to: {output_dir}')
    print(f'   • detailed_results.json - Complete analysis data')
    print(f'   • problematic_files.json - Mismatched files data')
    print(f'   • identity_statistics.json - Per-identity statistics')
    print(f'   • proposed_reindexing.json - Suggested filename corrections')
    print(f'   • problematic_files_report.txt - Human-readable report')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze transcript accuracy using combined distance metrics (edit distance + word overlap + length penalty)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  pip install tqdm scipy numpy

Examples:
  python analyze_transcript_accuracy.py \\
    --transcripts_path "/path/to/transcripts/*.txt" \\
    --texts_path "/path/to/ground_truth.txt" \\
    --output_dir "/path/to/output"
  
  # Use 8 worker processes for faster processing
  python analyze_transcript_accuracy.py \\
    --transcripts_path "/path/to/transcripts/*.txt" \\
    --texts_path "/path/to/ground_truth.txt" \\
    --output_dir "/path/to/output" \\
    --workers 8
  
  # Use single process (useful for debugging)
  python analyze_transcript_accuracy.py \\
    --transcripts_path "/path/to/transcripts/*.txt" \\
    --texts_path "/path/to/ground_truth.txt" \\
    --output_dir "/path/to/output" \\
    --workers 1
        """
    )
    
    parser.add_argument(
        '--transcripts_path',
        required=True,
        help='Glob pattern for transcript files (e.g., "/path/to/*.txt")'
    )
    
    parser.add_argument(
        '--texts_path',
        required=True,
        help='Path to ground truth text file (one sentence per line, no markup)'
    )
    
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of worker processes (default: auto-detect based on CPU count)'
    )
    
    args = parser.parse_args()
    
    analyze_transcripts(
        transcripts_path=args.transcripts_path,
        texts_path=args.texts_path,
        output_dir=args.output_dir,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
