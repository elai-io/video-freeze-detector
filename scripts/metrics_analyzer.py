"""
Video Freeze Detection Metrics Analyzer

This script analyzes various metrics for video freeze detection and provides
comprehensive analysis including all frame metrics, ranking analysis, and 
performance evaluation against known freeze frames.

Author: Video Freeze Detection System
Version: 2.0
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional
import json
import argparse
import csv
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.video_analyzer import VideoAnalyzer


class MetricsDocumentation:
    """Documentation for all metrics used in freeze detection"""
    
    METRICS_INFO = {
        'velocity_min': {
            'name': 'Velocity Minimum',
            'formula': 'min(|diff|) across cameras',
            'description': 'Minimum velocity (frame differences) - detects freeze frames',
            'order': 'velocity'
        },
        'velocity_min_ratio': {
            'name': 'Velocity Minimum Ratio',
            'formula': 'min(diff) / mean(diff) across cameras',
            'description': 'Ratio of minimum velocity to mean - detects relative freezes',
            'order': 'velocity'
        },
        'velocity_std_ratio': {
            'name': 'Velocity Standard Deviation Ratio',
            'formula': 'std(diff) / mean(diff) across cameras',
            'description': 'Coefficient of variation for velocity - detects uniformity',
            'order': 'velocity'
        },
        'acceleration_min': {
            'name': 'Acceleration Minimum',
            'formula': 'min(diff[i] - diff[i-1]) across cameras',
            'description': 'Minimum acceleration (with negative values) - detects deceleration/freeze',
            'order': 'acceleration'
        },
        'acceleration_range': {
            'name': 'Acceleration Range',
            'formula': 'max(acceleration) - min(acceleration) across cameras',
            'description': 'Range of acceleration values - detects variation during freeze',
            'order': 'acceleration'
        },
        'jerk_max': {
            'name': 'Jerk Maximum (MAX_3RD)',
            'formula': 'max(|acc[i] - acc[i-1]|) across cameras',
            'description': 'Maximum jerk (difference of accelerations) across cameras - BEST METRIC',
            'order': 'jerk'
        },
        'jerk_range': {
            'name': 'Jerk Range',
            'formula': 'max(jerk) - min(jerk) across cameras',
            'description': 'Range of jerk values across cameras',
            'order': 'jerk'
        },
        'jerk_max_abs': {
            'name': 'Jerk Maximum Absolute',
            'formula': 'max(|jerk|) across cameras',
            'description': 'Maximum absolute jerk across cameras',
            'order': 'jerk'
        },
        
        # Edge-based metrics (same as above but with edge detection preprocessing)
        'edge_velocity_min': {
            'name': 'Edge Velocity Minimum',
            'formula': 'min(|edge_diff|) across cameras',
            'description': 'Minimum edge velocity - detects freeze frames (edge-based)',
            'order': 'edge_velocity'
        },
        'edge_velocity_min_ratio': {
            'name': 'Edge Velocity Minimum Ratio',
            'formula': 'min(edge_diff) / mean(edge_diff) across cameras',
            'description': 'Ratio of minimum edge velocity to mean - detects relative freezes (edge-based)',
            'order': 'edge_velocity'
        },
        'edge_velocity_std_ratio': {
            'name': 'Edge Velocity Standard Deviation Ratio',
            'formula': 'std(edge_diff) / mean(edge_diff) across cameras',
            'description': 'Coefficient of variation for edge velocity - detects uniformity (edge-based)',
            'order': 'edge_velocity'
        },
        'edge_acceleration_min': {
            'name': 'Edge Acceleration Minimum',
            'formula': 'min(edge_diff[i] - edge_diff[i-1]) across cameras',
            'description': 'Minimum edge acceleration (with negative values) - detects deceleration/freeze (edge-based)',
            'order': 'edge_acceleration'
        },
        'edge_acceleration_range': {
            'name': 'Edge Acceleration Range',
            'formula': 'max(edge_acceleration) - min(edge_acceleration) across cameras',
            'description': 'Range of edge acceleration values - detects variation during freeze (edge-based)',
            'order': 'edge_acceleration'
        },
        'edge_jerk_max': {
            'name': 'Edge Jerk Maximum',
            'formula': 'max(|edge_acc[i] - edge_acc[i-1]|) across cameras',
            'description': 'Maximum edge jerk (difference of edge accelerations) across cameras (edge-based)',
            'order': 'edge_jerk'
        },
        'edge_jerk_range': {
            'name': 'Edge Jerk Range',
            'formula': 'max(edge_jerk) - min(edge_jerk) across cameras',
            'description': 'Range of edge jerk values across cameras (edge-based)',
            'order': 'edge_jerk'
        },
        'edge_jerk_max_abs': {
            'name': 'Edge Jerk Maximum Absolute',
            'formula': 'max(|edge_jerk|) across cameras',
            'description': 'Maximum absolute edge jerk across cameras (edge-based)',
            'order': 'edge_jerk'
        },

    }
    
    @classmethod
    def print_documentation(cls):
        """Print comprehensive metrics documentation"""
        print("📚 METRICS DOCUMENTATION")
        print("=" * 80)
        print()
        print("TERMINOLOGY:")
        print("  • Frame differences (diff) = velocity (same thing)")
        print("  • Difference of velocities = acceleration")
        print("  • Difference of accelerations = jerk")
        print()
        print("METRICS FORMULAS:")
        print("-" * 80)
        
        for metric_key, info in cls.METRICS_INFO.items():
            print(f"• {info['name']} ({info['order']}):")
            print(f"  Formula: {info['formula']}")
            print(f"  Description: {info['description']}")
            print()
    
    @classmethod
    def save_documentation(cls, output_path: str):
        """Save documentation to file"""
        doc_file = os.path.join(output_path, 'metrics_documentation.md')
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write("# Video Freeze Detection Metrics Documentation\n\n")
            f.write("## Terminology\n\n")
            f.write("- **Frame differences (diff)**: velocity (same thing)\n")
            f.write("- **Difference of velocities**: acceleration\n")
            f.write("- **Difference of accelerations**: jerk\n\n")
            f.write("## Metrics Formulas\n\n")
            
            for metric_key, info in cls.METRICS_INFO.items():
                f.write(f"### {info['name']} ({info['order']})\n\n")
                f.write(f"**Formula**: `{info['formula']}`\n\n")
                f.write(f"**Description**: {info['description']}\n\n")
        
        print(f"Documentation saved to: {doc_file}")


class MetricsAnalyzer:
    """Comprehensive metrics analyzer for freeze detection"""
    
    def __init__(self, input_path: str, output_path: str = 'metrics_analysis', 
                 cache_file: str = None, known_freezes: List[int] = None):
        self.input_path = input_path
        self.output_path = output_path
        self.cache_file = cache_file or os.path.join(output_path, 'frame_differences_cache.json')
        self.known_freezes = known_freezes or []
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize data containers
        self.frame_differences = None
        self.edge_frame_differences = None
        self.derivatives = {}
        self.edge_derivatives = {}
        self.all_metrics = {}
        self.full_metrics_table = []
        
    def load_or_compute_frame_differences(self, recalculate: bool = False):
        """Load frame differences from cache or compute them"""
        if not recalculate and os.path.exists(self.cache_file):
            print("Loading frame differences from cache...")
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Handle both old and new cache formats
                if isinstance(cache_data, dict) and 'regular' in cache_data:
                    self.frame_differences = cache_data['regular']
                    self.edge_frame_differences = cache_data['edges']
                else:
                    # Old cache format - only regular differences
                    self.frame_differences = cache_data
                    self.edge_frame_differences = None
                
                print(f"Frame differences loaded from: {self.cache_file}")
                if self.edge_frame_differences is None:
                    print("Edge differences not in cache, will compute them...")
                    analyzer = VideoAnalyzer(self.input_path, verbose=True)
                    analyzer.load_videos()
                    self.edge_frame_differences = analyzer.compute_edge_frame_differences()
                return
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Will recalculate frame differences...")
        
        print("Computing frame differences...")
        analyzer = VideoAnalyzer(self.input_path, verbose=True)
        video_files = analyzer.load_videos()
        self.frame_differences = analyzer.compute_frame_differences()
        
        print("Computing edge-based frame differences...")
        self.edge_frame_differences = analyzer.compute_edge_frame_differences()
        
        # Save to cache
        print("Saving frame differences to cache...")
        cache_data = {
            'regular': [[float(x) for x in cam_diffs] for cam_diffs in self.frame_differences],
            'edges': [[float(x) for x in cam_diffs] for cam_diffs in self.edge_frame_differences]
        }
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Frame differences cached to: {self.cache_file}")
    
    def compute_derivatives(self):
        """Compute all derivatives from frame differences"""
        print("Computing derivatives...")
        num_cameras = len(self.frame_differences)
        
        # Regular derivatives
        self.derivatives = {
            '1st': self.frame_differences,  # velocity (already computed)
            '2nd': [],  # acceleration
            '3rd': []   # jerk
        }
        
        for cam_idx in range(num_cameras):
            velocity_list = self.frame_differences[cam_idx]
            
            # 2nd derivative (acceleration)
            acceleration = []
            for i in range(1, len(velocity_list)):
                accel = velocity_list[i] - velocity_list[i-1]
                acceleration.append(accel)
            
            # 3rd derivative (jerk)
            jerk = []
            for i in range(1, len(acceleration)):
                jerk_val = acceleration[i] - acceleration[i-1]
                jerk.append(jerk_val)
            
            self.derivatives['2nd'].append(acceleration)
            self.derivatives['3rd'].append(jerk)
        
        # Edge-based derivatives
        print("Computing edge-based derivatives...")
        self.edge_derivatives = {
            '1st': self.edge_frame_differences,  # edge velocity
            '2nd': [],  # edge acceleration
            '3rd': []   # edge jerk
        }
        
        for cam_idx in range(num_cameras):
            velocity_list = self.edge_frame_differences[cam_idx]
            
            # 2nd derivative (acceleration)
            acceleration = []
            for i in range(1, len(velocity_list)):
                accel = velocity_list[i] - velocity_list[i-1]
                acceleration.append(accel)
            
            # 3rd derivative (jerk)
            jerk = []
            for i in range(1, len(acceleration)):
                jerk_val = acceleration[i] - acceleration[i-1]
                jerk.append(jerk_val)
            
            self.edge_derivatives['2nd'].append(acceleration)
            self.edge_derivatives['3rd'].append(jerk)
    
    def compute_all_metrics(self):
        """Compute all metrics for each frame"""
        print("Computing all metrics...")
        
        # Define metrics to compute
        metrics_config = {
            '1st': ['min', 'min_ratio', 'std_ratio'],
            '2nd': ['min', 'range'],  # Только минимум (с отрицательным) и разброс
            '3rd': ['max', 'range', 'max_abs']  # Jerk остается как есть - детектит разморозку
        }
        
        self.all_metrics = {}
        
        # Compute regular metrics
        print("Computing regular metrics...")
        for order in ['1st', '2nd', '3rd']:
            derivatives = self.derivatives[order]
            num_cameras = len(derivatives)
            min_length = min(len(deriv) for deriv in derivatives)
            
            metrics = []
            for i in range(min_length):
                # Get values for all cameras at this frame
                values = [float(derivatives[cam][i]) for cam in range(num_cameras)]
                
                # Frame index calculation (adjust based on derivative order)
                if order == '1st':
                    frame_index = i + 1  # velocity: diff[0] corresponds to frame 1 (difference between frames 0 and 1)
                elif order == '2nd':
                    frame_index = i + 2  # acceleration: acc[0] corresponds to frame 2 (difference between diff[0] and diff[1])
                elif order == '3rd':
                    frame_index = i + 2  # jerk peaks align one frame earlier (keep as is - works correctly)
                
                # Compute metrics based on order
                if order == '1st':  # Velocity metrics
                    abs_vals = [abs(v) for v in values]
                    min_abs = min(abs_vals)
                    mean_abs = np.mean(abs_vals)
                    std_abs = np.std(abs_vals)
                    min_ratio = min_abs / (mean_abs + 1e-8)  # Avoid division by zero
                    std_ratio = std_abs / (mean_abs + 1e-8)
                    
                    metrics.append({
                        'frame_index': frame_index,
                        'values': values,
                        f'min_{order}': min_abs,
                        f'min_ratio_{order}': min_ratio,
                        f'std_ratio_{order}': std_ratio,
                        f'cam1_{order}': values[0],
                        f'cam2_{order}': values[1],
                        f'cam3_{order}': values[2]
                    })
                    
                elif order == '2nd':  # Acceleration metrics
                    min_val = min(values)  # Минимум БЕЗ модуля (отрицательные важны!)
                    max_val = max(values)
                    value_range = max_val - min_val  # Разница между мин и макс
                    
                    metrics.append({
                        'frame_index': frame_index,
                        'values': values,
                        f'min_{order}': min_val,
                        f'range_{order}': value_range,
                        f'cam1_{order}': values[0],
                        f'cam2_{order}': values[1],
                        f'cam3_{order}': values[2]
                    })
                    
                else:  # Jerk metrics (остаются как есть)
                    max_val = max(values)
                    min_val = min(values)
                    abs_vals = [abs(v) for v in values]
                    max_abs = max(abs_vals)
                    value_range = max_val - min_val
                    
                    metrics.append({
                        'frame_index': frame_index,
                        'values': values,
                        f'max_{order}': max_val,
                        f'min_{order}': min_val,
                        f'max_abs_{order}': max_abs,
                        f'range_{order}': value_range,
                        f'cam1_{order}': values[0],
                        f'cam2_{order}': values[1],
                        f'cam3_{order}': values[2]
                    })
            
            self.all_metrics[order] = metrics
        
        # Compute edge-based metrics
        print("Computing edge-based metrics...")
        for order in ['1st', '2nd', '3rd']:
            derivatives = self.edge_derivatives[order]
            num_cameras = len(derivatives)
            min_length = min(len(deriv) for deriv in derivatives)
            
            metrics = []
            for i in range(min_length):
                # Get values for all cameras at this frame
                values = [float(derivatives[cam][i]) for cam in range(num_cameras)]
                
                # Frame index calculation (adjust based on derivative order)
                if order == '1st':
                    frame_index = i + 1  # velocity: diff[0] corresponds to frame 1 (difference between frames 0 and 1)
                elif order == '2nd':
                    frame_index = i + 2  # acceleration: acc[0] corresponds to frame 2 (difference between diff[0] and diff[1])
                elif order == '3rd':
                    frame_index = i + 2  # jerk peaks align one frame earlier (keep as is - works correctly)
                
                # Compute metrics based on order
                if order == '1st':  # Velocity metrics
                    abs_vals = [abs(v) for v in values]
                    min_abs = min(abs_vals)
                    mean_abs = np.mean(abs_vals)
                    std_abs = np.std(abs_vals)
                    min_ratio = min_abs / (mean_abs + 1e-8)  # Avoid division by zero
                    std_ratio = std_abs / (mean_abs + 1e-8)
                    
                    metrics.append({
                        'frame_index': frame_index,
                        'values': values,
                        f'min_{order}': min_abs,
                        f'min_ratio_{order}': min_ratio,
                        f'std_ratio_{order}': std_ratio,
                        f'cam1_{order}': values[0],
                        f'cam2_{order}': values[1],
                        f'cam3_{order}': values[2]
                    })
                    
                elif order == '2nd':  # Acceleration metrics
                    min_val = min(values)  # Минимум БЕЗ модуля (отрицательные важны!)
                    max_val = max(values)
                    value_range = max_val - min_val  # Разница между мин и макс
                    
                    metrics.append({
                        'frame_index': frame_index,
                        'values': values,
                        f'min_{order}': min_val,
                        f'range_{order}': value_range,
                        f'cam1_{order}': values[0],
                        f'cam2_{order}': values[1],
                        f'cam3_{order}': values[2]
                    })
                    
                else:  # Jerk metrics (остаются как есть)
                    max_val = max(values)
                    min_val = min(values)
                    abs_vals = [abs(v) for v in values]
                    max_abs = max(abs_vals)
                    value_range = max_val - min_val
                    
                    metrics.append({
                        'frame_index': frame_index,
                        'values': values,
                        f'max_{order}': max_val,
                        f'min_{order}': min_val,
                        f'max_abs_{order}': max_abs,
                        f'range_{order}': value_range,
                        f'cam1_{order}': values[0],
                        f'cam2_{order}': values[1],
                        f'cam3_{order}': values[2]
                    })
            
            # Store edge metrics with 'edge_' prefix
            self.all_metrics[f'edge_{order}'] = metrics
    
    def create_full_metrics_table(self):
        """Create comprehensive table with all frames and metrics"""
        print("Creating full metrics table...")
        
        # Get all unique frame indices
        all_frames = set()
        for order in ['1st', '2nd', '3rd']:
            for metric in self.all_metrics[order]:
                all_frames.add(metric['frame_index'])
        
        # Create lookup dictionaries for each metric type
        metrics_lookup = {}
        for order in ['1st', '2nd', '3rd']:
            metrics_lookup[order] = {m['frame_index']: m for m in self.all_metrics[order]}
        
        # Build full table
        self.full_metrics_table = []
        for frame in sorted(all_frames):
            row = {'frame_index': frame}
            
            # Initialize all metrics to None
            row.update({
                'velocity_min': None,
                'velocity_min_ratio': None,
                'velocity_std_ratio': None,
                'acceleration_min': None,
                'acceleration_range': None,
                'jerk_max': None,
                'jerk_range': None,
                'jerk_max_abs': None,
                'cam1_velocity': None,
                'cam2_velocity': None,
                'cam3_velocity': None,
                'cam1_acceleration': None,
                'cam2_acceleration': None,
                'cam3_acceleration': None,
                'cam1_jerk': None,
                'cam2_jerk': None,
                'cam3_jerk': None
            })
            
            # Add metrics for each derivative order
            if frame in metrics_lookup.get('1st', {}):
                metric_data = metrics_lookup['1st'][frame]
                row.update({
                    'velocity_min': metric_data.get('min_1st', None),
                    'velocity_min_ratio': metric_data.get('min_ratio_1st', None),
                    'velocity_std_ratio': metric_data.get('std_ratio_1st', None),
                    'cam1_velocity': metric_data.get('cam1_1st', None),
                    'cam2_velocity': metric_data.get('cam2_1st', None),
                    'cam3_velocity': metric_data.get('cam3_1st', None)
                })
            
            if frame in metrics_lookup.get('2nd', {}):
                metric_data = metrics_lookup['2nd'][frame]
                row.update({
                    'acceleration_min': metric_data.get('min_2nd', None),
                    'acceleration_range': metric_data.get('range_2nd', None),
                    'cam1_acceleration': metric_data.get('cam1_2nd', None),
                    'cam2_acceleration': metric_data.get('cam2_2nd', None),
                    'cam3_acceleration': metric_data.get('cam3_2nd', None)
                })
            
            if frame in metrics_lookup.get('3rd', {}):
                metric_data = metrics_lookup['3rd'][frame]
                row.update({
                    'jerk_max': metric_data.get('max_3rd', None),
                    'jerk_range': metric_data.get('range_3rd', None),
                    'jerk_max_abs': metric_data.get('max_abs_3rd', None),
                    'cam1_jerk': metric_data.get('cam1_3rd', None),
                    'cam2_jerk': metric_data.get('cam2_3rd', None),
                    'cam3_jerk': metric_data.get('cam3_3rd', None)
                })
            
            # Mark if this frame is a known freeze
            row['is_known_freeze'] = frame in self.known_freezes
            
            self.full_metrics_table.append(row)
    
    def _extract_edge_metric_data(self, metric_name: str) -> List[Dict]:
        """Extract edge metric data from all_metrics for analysis"""
        # Map edge metric names to internal structure
        edge_mapping = {
            'edge_velocity_min': ('edge_1st', 'min_1st'),
            'edge_velocity_min_ratio': ('edge_1st', 'min_ratio_1st'),
            'edge_velocity_std_ratio': ('edge_1st', 'std_ratio_1st'),
            'edge_acceleration_min': ('edge_2nd', 'min_2nd'),
            'edge_acceleration_range': ('edge_2nd', 'range_2nd'),
            'edge_jerk_max': ('edge_3rd', 'max_3rd'),
            'edge_jerk_range': ('edge_3rd', 'range_3rd'),
            'edge_jerk_max_abs': ('edge_3rd', 'max_abs_3rd')
        }
        
        if metric_name not in edge_mapping:
            return []
        
        order_key, value_key = edge_mapping[metric_name]
        if order_key not in self.all_metrics:
            return []
        
        # Convert to format expected by analysis
        result = []
        for metric_data in self.all_metrics[order_key]:
            row = {
                'frame_index': metric_data['frame_index'],
                metric_name: metric_data.get(value_key, None)
            }
            result.append(row)
        
        return result

    
    def analyze_known_freezes(self):
        """Analyze performance on known freeze frames"""
        if not self.known_freezes:
            print("No known freeze frames provided for analysis")
            return {}
        
        print(f"Analyzing performance on {len(self.known_freezes)} known freeze frames...")
        
        # Define metrics to analyze (regular + edge-based)
        metrics_to_analyze = [
            'velocity_min', 'velocity_min_ratio', 'velocity_std_ratio',
            'acceleration_min', 'acceleration_range',
            'jerk_max', 'jerk_range', 'jerk_max_abs',
            'edge_velocity_min', 'edge_velocity_min_ratio', 'edge_velocity_std_ratio',
            'edge_acceleration_min', 'edge_acceleration_range',
            'edge_jerk_max', 'edge_jerk_range', 'edge_jerk_max_abs'
        ]
        
        results = {}
        
        for metric_name in metrics_to_analyze:
            # Get metric data from appropriate source
            if metric_name.startswith('edge_'):
                # Extract edge metrics from all_metrics
                metric_data = self._extract_edge_metric_data(metric_name)
                if not metric_data:
                    continue
                valid_rows = metric_data
            else:
                # Get regular metrics from full_metrics_table
                valid_rows = [row for row in self.full_metrics_table if row.get(metric_name) is not None]
            
            if not valid_rows:
                continue
            
            # Sort by metric value - direction depends on metric type
            if metric_name in ['velocity_min', 'acceleration_min', 'edge_velocity_min', 'edge_acceleration_min']:
                # For minimum metrics, lower values indicate freezes (ascending sort)
                sorted_rows = sorted(valid_rows, key=lambda x: x[metric_name], reverse=False)
            elif metric_name in ['velocity_min_ratio', 'velocity_std_ratio', 'edge_velocity_min_ratio', 'edge_velocity_std_ratio']:
                # For ratio metrics, lower values indicate freezes (ascending sort)
                sorted_rows = sorted(valid_rows, key=lambda x: x[metric_name], reverse=False)
            elif metric_name in ['acceleration_range', 'edge_acceleration_range']:
                # For range metrics, higher values indicate freezes (descending sort)
                sorted_rows = sorted(valid_rows, key=lambda x: x[metric_name], reverse=True)
            else:
                # For jerk metrics, higher values indicate unfreeze events (descending sort)
                sorted_rows = sorted(valid_rows, key=lambda x: x[metric_name], reverse=True)
            
            # Find positions of known freezes
            freeze_positions = []
            for i, row in enumerate(sorted_rows, 1):
                frame_idx = row['frame_index']
                
                # Adjust frame index for jerk metrics (they detect unfreeze, freeze is 1-2 frames earlier)
                if metric_name.startswith('jerk_') or metric_name.startswith('edge_jerk_'):
                    # Check if any known freeze is 1-2 frames before this unfreeze event
                    if (frame_idx - 1) in self.known_freezes or (frame_idx - 2) in self.known_freezes:
                        freeze_positions.append(i)
                else:
                    # For velocity/acceleration metrics, direct match
                    if frame_idx in self.known_freezes:
                        freeze_positions.append(i)
            
            # Calculate average position
            avg_position = sum(freeze_positions) / len(freeze_positions) if freeze_positions else None
            
            # Use number of known freezes as top size
            total_frames = len(sorted_rows)
            top_frame_count = len(self.known_freezes)
            
            # Count detections in top N (where N = number of known freezes)
            top_frames = [row['frame_index'] for row in sorted_rows[:top_frame_count]]
            detected_in_top = len([f for f in self.known_freezes if f in top_frames])
            detection_rate = detected_in_top / len(self.known_freezes) * 100 if self.known_freezes else 0
            
            results[metric_name] = {
                'avg_position': avg_position,
                'top_frame_count': top_frame_count,
                'detected_in_top': detected_in_top,
                'detection_rate': detection_rate,
                'freeze_positions': freeze_positions
            }
        

        
        return results
    
    def save_analysis_to_csv(self, analysis_results: Dict, csv_file: str):
        """Save analysis results to CSV format"""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'Metric',
                'Average Position',
                'Top Frame Count',
                'Detected in Top',
                'Detection Rate',
                'Freeze Positions'
            ])
            # Sort by detection rate descending
            sorted_metrics = sorted(
                analysis_results.items(),
                key=lambda x: x[1]['detection_rate'],
                reverse=True
            )
            # Write data rows
            for metric_name, metric_data in sorted_metrics:
                avg_pos = f"{metric_data['avg_position']:.1f}" if metric_data['avg_position'] else "N/A"
                freeze_positions_str = ', '.join(map(str, metric_data['freeze_positions']))
                writer.writerow([
                    metric_name,
                    avg_pos,
                    metric_data['top_frame_count'],
                    metric_data['detected_in_top'],
                    f"{metric_data['detection_rate']:.1f}%",
                    freeze_positions_str
                ])

    def save_results(self):
        """Save all results to files"""
        print(f"Saving results to {self.output_path}...")
        
        # Save full metrics table as CSV
        csv_file = os.path.join(self.output_path, 'full_metrics_table.csv')
        if self.full_metrics_table:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self.full_metrics_table[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.full_metrics_table)
            print(f"Full metrics table saved to: {csv_file}")
        
        # Save metrics analysis results
        if self.known_freezes:
            analysis_results = self.analyze_known_freezes()
            
            # Save JSON
            json_file = os.path.join(self.output_path, 'freeze_analysis_results.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            print(f"Analysis results saved to: {json_file}")
            
            # Save CSV
            csv_file = os.path.join(self.output_path, 'freeze_analysis_results.csv')
            self.save_analysis_to_csv(analysis_results, csv_file)
            print(f"Analysis results CSV saved to: {csv_file}")
            
            # Print summary
            self.print_analysis_summary(analysis_results)
        
        # Save documentation
        MetricsDocumentation.save_documentation(self.output_path)
    
    def print_analysis_summary(self, analysis_results: Dict):
        """Print analysis summary"""
        print("\n" + "=" * 80)
        print("FREEZE DETECTION ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Known freeze frames: {self.known_freezes}")
        print(f"Total known freezes: {len(self.known_freezes)}")
        print()
        
        # Sort metrics by performance
        sorted_metrics = sorted(
            analysis_results.items(),
            key=lambda x: x[1]['detection_rate'],
            reverse=True
        )
        
        print("PERFORMANCE RANKING (Top N Detection Rate):")
        print("-" * 80)
        print(f"{'Metric':<25} {'Top N':<8} {'Detected':<10} {'Rate':<10} {'Avg Pos':<10}")
        print("-" * 80)
        
        for metric_name, results in sorted_metrics:
            avg_pos = f"{results['avg_position']:.1f}" if results['avg_position'] else "N/A"
            print(f"{metric_name:<25} {results['top_frame_count']:<8} "
                  f"{results['detected_in_top']:<10} {results['detection_rate']:<10.1f} {avg_pos:<10}")
        
        # Find best metric
        best_metric = sorted_metrics[0]
        print(f"\n🏆 BEST METRIC: {best_metric[0]}")
        print(f"   Detection rate (top {best_metric[1]['top_frame_count']}): {best_metric[1]['detection_rate']:.1f}%")
        print(f"   Average position: {best_metric[1]['avg_position']:.1f}")
    
    def run_analysis(self, recalculate: bool = False):
        """Run complete analysis"""
        print("🔬 COMPREHENSIVE METRICS ANALYSIS")
        print("=" * 80)
        print(f"Input path: {self.input_path}")
        print(f"Output path: {self.output_path}")
        print(f"Known freezes: {self.known_freezes}")
        print("-" * 80)
        
        # Print documentation
        MetricsDocumentation.print_documentation()
        
        # Load/compute data
        self.load_or_compute_frame_differences(recalculate)
        self.compute_derivatives()
        self.compute_all_metrics()
        self.create_full_metrics_table()
        
        # Save results
        self.save_results()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Video Freeze Metrics Analyzer')
    parser.add_argument('input_path', help='Path to folder with video files')
    parser.add_argument('--output', '-o', default='metrics_analysis', help='Output directory')
    parser.add_argument('--recalculate', '-r', action='store_true', 
                       help='Force recalculation of frame differences')
    parser.add_argument('--known-freezes', '-f', nargs='*', type=int,
                       help='List of known freeze frame numbers')
    parser.add_argument('--cache-file', '-c', help='Path to cache file for frame differences')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input path {args.input_path} not found")
        sys.exit(1)
    
    # Default known freezes if not provided
    known_freezes = args.known_freezes or [43, 89, 93, 149, 153, 182, 205, 209, 213, 223, 233, 235, 269, 273]
    
    try:
        analyzer = MetricsAnalyzer(
            input_path=args.input_path,
            output_path=args.output,
            cache_file=args.cache_file,
            known_freezes=known_freezes
        )
        analyzer.run_analysis(recalculate=args.recalculate)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 