import numpy as np
import pandas as pd
import csv
import os
from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray


class FreezeDetectorEdge:
    """
    Freeze detector using proper edge-based analysis with structured data arrays
    """
    
    def __init__(self, edge_frame_differences: List[List[float]], 
                 freeze_threshold: float = 0.25):
        """
        Initialize edge-based freeze detector
        
        Args:
            edge_frame_differences: List of edge-based frame differences for each camera
            freeze_threshold: Threshold for freeze detection (default 0.25)
        """
        self.edge_frame_differences = edge_frame_differences
        self.freeze_threshold = freeze_threshold
        self.num_cameras = len(edge_frame_differences)
        
        # Find minimum length across all cameras (synchronized frames)
        self.min_frames = min(len(diffs) for diffs in edge_frame_differences)
        
        # Data arrays
        self.edge_raw = []          # [frame][camera] = raw edge_diff
        self.edge_normalized = []   # [frame][camera] = normalized edge_diff
        self.frame_metric = []      # [frame] = min(normalized values)
        
        # Computed results
        self.frame_data = []        # Complete frame data for CSV/Excel
        self.camera_statistics = []
        self.suspicious_sequences = []
        
        # Compute all data
        self._compute_data_arrays()
        self._compute_statistics()
        
    def _compute_data_arrays(self):
        """Compute the main data arrays: edge_raw, edge_normalized, frame_metric"""
        print("Computing data arrays...")
        
        for frame_idx in range(self.min_frames):
            # Get raw edge differences for all cameras at this frame
            raw_values = [self.edge_frame_differences[cam][frame_idx] 
                         for cam in range(self.num_cameras)]
            
            # Calculate normalized values (raw / max_raw)
            max_raw = max(raw_values)
            if max_raw > 0:
                normalized_values = [raw / max_raw for raw in raw_values]
            else:
                normalized_values = [0.0] * self.num_cameras
            
            # Frame metric = minimum of normalized values
            frame_metric_val = min(normalized_values)
            
            # Store in arrays
            self.edge_raw.append(raw_values)
            self.edge_normalized.append(normalized_values)
            self.frame_metric.append(frame_metric_val)
            
            # Create complete frame data for CSV/Excel
            frame_data = {
                'frame_number': frame_idx + 1,
                'cam1_raw': raw_values[0],
                'cam2_raw': raw_values[1],
                'cam3_raw': raw_values[2],
                'cam1_norm': normalized_values[0],
                'cam2_norm': normalized_values[1],
                'cam3_norm': normalized_values[2],
                'cam1_freeze': 1 if normalized_values[0] < self.freeze_threshold else 0,
                'cam2_freeze': 1 if normalized_values[1] < self.freeze_threshold else 0,
                'cam3_freeze': 1 if normalized_values[2] < self.freeze_threshold else 0,
                'frame_metric': frame_metric_val,
                'frame_freeze': 1 if frame_metric_val < self.freeze_threshold else 0
            }
            self.frame_data.append(frame_data)
    
    def _compute_statistics(self):
        """Compute camera statistics and sequences"""
        print("Computing statistics...")
        
        # Camera statistics
        self.camera_statistics = []
        for cam_idx in range(self.num_cameras):
            # Count freezes for this camera
            freeze_count = sum(1 for frame in self.frame_data 
                             if frame[f'cam{cam_idx + 1}_freeze'] == 1)
            
            # Calculate percentage
            freeze_percentage = (freeze_count / self.min_frames) * 100
            
            # Average values
            avg_raw = np.mean([self.edge_raw[f][cam_idx] for f in range(self.min_frames)])
            avg_norm = np.mean([self.edge_normalized[f][cam_idx] for f in range(self.min_frames)])
            
            self.camera_statistics.append({
                'camera': cam_idx + 1,
                'freeze_count': freeze_count,
                'freeze_percentage': freeze_percentage,
                'avg_raw': avg_raw,
                'avg_normalized': avg_norm,
                'total_frames': self.min_frames
            })
        
        # Find suspicious sequences
        self._find_suspicious_sequences()
    
    def _find_suspicious_sequences(self):
        """Find consecutive sequences of suspicious frames"""
        suspicious_frames = [i for i, frame in enumerate(self.frame_data) 
                           if frame['frame_freeze'] == 1]
        
        if not suspicious_frames:
            self.suspicious_sequences = []
            return
        
        # Group consecutive frames into sequences
        sequences = []
        current_sequence = [suspicious_frames[0]]
        
        for i in range(1, len(suspicious_frames)):
            if suspicious_frames[i] == suspicious_frames[i-1] + 1:
                # Continue current sequence
                current_sequence.append(suspicious_frames[i])
            else:
                # End current sequence, start new one
                sequences.append(current_sequence)
                current_sequence = [suspicious_frames[i]]
        
        # Don't forget the last sequence
        sequences.append(current_sequence)
        
        # Convert to sequence objects with metadata
        self.suspicious_sequences = []
        for seq_frames in sequences:
            start_frame = seq_frames[0] + 1  # +1 for 1-based frame numbering
            end_frame = seq_frames[-1] + 1
            length = len(seq_frames)
            
            # Find minimum frame metric in this sequence
            min_metric = min(self.frame_metric[f] for f in seq_frames)
            
            # Find most suspicious camera in this sequence
            most_suspicious_cam = 0
            min_avg_norm = float('inf')
            for cam_idx in range(self.num_cameras):
                avg_norm = np.mean([self.edge_normalized[f][cam_idx] for f in seq_frames])
                if avg_norm < min_avg_norm:
                    min_avg_norm = avg_norm
                    most_suspicious_cam = cam_idx
            
            self.suspicious_sequences.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'length': length,
                'min_metric': min_metric,
                'camera': most_suspicious_cam,
                'frames': seq_frames
            })
        
        # Sort by length (longest first), then by min_metric (most suspicious first)
        self.suspicious_sequences.sort(key=lambda x: (x['length'], -x['min_metric']), reverse=True)
    
    def detect_freezes(self) -> List[Dict[str, Any]]:
        """
        Detect freeze candidates based on threshold (for statistics)
        
        Returns:
            List of freeze candidates that exceed threshold
        """
        # Get suspicious frames based on threshold
        suspicious_frames = [i for i, frame in enumerate(self.frame_data) 
                           if frame['frame_freeze'] == 1]
        
        # Convert to format expected by visualizer
        freeze_candidates = []
        for rank, frame_idx in enumerate(suspicious_frames):
            frame_data = self.frame_data[frame_idx]
            
            # Find most suspicious camera for this frame
            cam_norms = [frame_data['cam1_norm'], frame_data['cam2_norm'], frame_data['cam3_norm']]
            most_suspicious_cam = cam_norms.index(min(cam_norms))
            
            candidate = {
                'frame_index': frame_idx + 1,  # 1-based for visualizer
                'edge_velocity_min_ratio': frame_data['frame_metric'],
                'most_suspicious_camera': most_suspicious_cam,
                'rank': rank + 1,
                'suspicion_score': 1.0 - frame_data['frame_metric'],
                'detection_method': 'edge_normalized_analysis',
                'edge_velocities': [frame_data['cam1_raw'], frame_data['cam2_raw'], frame_data['cam3_raw']]
            }
            freeze_candidates.append(candidate)
        
        return freeze_candidates
    
    def detect_freezes_for_visualization(self, visualization_count: int) -> List[Dict[str, Any]]:
        """
        Detect top N most suspicious frames for visualization
        
        Args:
            visualization_count: Number of most suspicious frames to return
            
        Returns:
            List of freeze candidates sorted by suspicion level
        """
        # Sort all frames by frame_metric (ascending - lower is more suspicious)
        frame_indices_with_metrics = [(i, frame['frame_metric']) for i, frame in enumerate(self.frame_data)]
        frame_indices_with_metrics.sort(key=lambda x: x[1])
        
        # Take top N as visualization candidates
        num_candidates = min(visualization_count, len(frame_indices_with_metrics))
        top_frame_indices = [frame_idx for frame_idx, metric in frame_indices_with_metrics[:num_candidates]]
        
        # Convert to format expected by visualizer
        freeze_candidates = []
        for rank, frame_idx in enumerate(top_frame_indices):
            frame_data = self.frame_data[frame_idx]
            
            # Find most suspicious camera for this frame
            cam_norms = [frame_data['cam1_norm'], frame_data['cam2_norm'], frame_data['cam3_norm']]
            most_suspicious_cam = cam_norms.index(min(cam_norms))
            
            candidate = {
                'frame_index': frame_idx + 1,  # 1-based for visualizer
                'edge_velocity_min_ratio': frame_data['frame_metric'],
                'most_suspicious_camera': most_suspicious_cam,
                'rank': rank + 1,
                'suspicion_score': 1.0 - frame_data['frame_metric'],
                'detection_method': 'edge_normalized_analysis',
                'edge_velocities': [frame_data['cam1_raw'], frame_data['cam2_raw'], frame_data['cam3_raw']]
            }
            freeze_candidates.append(candidate)
        
        return freeze_candidates
    
    def find_suspicious_sequences(self, ratio_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Find sequences of consecutive suspicious frames
        
        Args:
            ratio_threshold: Not used, kept for compatibility
            
        Returns:
            List of sequences sorted by length
        """
        return self.suspicious_sequences
    
    def find_longest_freeze_sequences_per_camera(self, min_length: int = 2) -> List[Dict[str, Any]]:
        """
        Find longest consecutive freeze sequences for each camera
        
        Args:
            min_length: Minimum sequence length to include (default 2)
            
        Returns:
            List of sequences with camera info, sorted by length descending
        """
        all_sequences = []
        
        # Process each camera separately
        for cam_idx in range(self.num_cameras):
            cam_column = f'cam{cam_idx + 1}_freeze'
            
            # Extract freeze values for this camera
            freeze_values = [frame[cam_column] for frame in self.frame_data]
            
            # Find sequences of consecutive 1s
            sequences = self._find_consecutive_sequences(freeze_values, cam_idx + 1)
            
            # Filter by minimum length
            sequences = [seq for seq in sequences if seq['length'] >= min_length]
            
            all_sequences.extend(sequences)
        
        # Sort by length (descending), then by start frame (ascending)
        all_sequences.sort(key=lambda x: (-x['length'], x['start_frame']))
        
        return all_sequences
    
    def _find_consecutive_sequences(self, binary_list: List[int], camera_num: int) -> List[Dict[str, Any]]:
        """
        Find all consecutive sequences of 1s in a binary list
        
        Args:
            binary_list: List of 0s and 1s
            camera_num: Camera number (1-based)
            
        Returns:
            List of sequence dictionaries
        """
        sequences = []
        current_start = None
        
        for i, value in enumerate(binary_list):
            if value == 1:
                if current_start is None:
                    # Start of new sequence
                    current_start = i
            else:
                if current_start is not None:
                    # End of sequence
                    length = i - current_start
                    sequences.append({
                        'camera': camera_num,
                        'start_frame': current_start + 1,  # 1-based frame numbering
                        'end_frame': i,  # 1-based frame numbering
                        'length': length,
                        'avg_metric': self._calculate_sequence_avg_metric(current_start, i - 1, camera_num - 1)
                    })
                    current_start = None
        
        # Handle case where sequence continues to the end
        if current_start is not None:
            length = len(binary_list) - current_start
            sequences.append({
                'camera': camera_num,
                'start_frame': current_start + 1,  # 1-based frame numbering
                'end_frame': len(binary_list),  # 1-based frame numbering
                'length': length,
                'avg_metric': self._calculate_sequence_avg_metric(current_start, len(binary_list) - 1, camera_num - 1)
            })
        
        return sequences
    
    def _calculate_sequence_avg_metric(self, start_idx: int, end_idx: int, cam_idx: int) -> float:
        """
        Calculate average normalized metric for a sequence
        
        Args:
            start_idx: Start frame index (0-based)
            end_idx: End frame index (0-based, inclusive)
            cam_idx: Camera index (0-based)
            
        Returns:
            Average normalized metric for the sequence
        """
        if start_idx > end_idx:
            return 0.0
        
        total = sum(self.edge_normalized[i][cam_idx] for i in range(start_idx, end_idx + 1))
        count = end_idx - start_idx + 1
        
        return total / count if count > 0 else 0.0
    
    def get_top_longest_sequences(self, top_n: int = 10, min_length: int = 2) -> List[Dict[str, Any]]:
        """
        Get top N longest freeze sequences across all cameras
        
        Args:
            top_n: Number of top sequences to return
            min_length: Minimum sequence length to include (default 2)
            
        Returns:
            List of top sequences with detailed info
        """
        all_sequences = self.find_longest_freeze_sequences_per_camera(min_length=min_length)
        
        # Take top N sequences
        top_sequences = all_sequences[:top_n]
        
        # Add additional info for each sequence
        for seq in top_sequences:
            # Calculate suspicion score (lower avg_metric = higher suspicion)
            seq['suspicion_score'] = 1.0 - seq['avg_metric']
            
            # Add frame range info
            seq['frame_range'] = f"{seq['start_frame']}-{seq['end_frame']}"
            
            # Calculate percentage of sequence length relative to total frames
            seq['sequence_percentage'] = (seq['length'] / self.min_frames) * 100
            
            # Get total suspicious frames for this camera
            total_suspicious_frames = sum(1 for frame in self.frame_data 
                                        if frame[f'cam{seq["camera"]}_freeze'] == 1)
            seq['camera_total_suspicious'] = total_suspicious_frames
            seq['camera_suspicious_percentage'] = (total_suspicious_frames / self.min_frames) * 100
            
            # Add quality assessment based on camera's total suspicious percentage
            if seq['camera_suspicious_percentage'] >= 10.0:
                seq['severity'] = 'CRITICAL'
            elif seq['camera_suspicious_percentage'] >= 5.0:
                seq['severity'] = 'HIGH'
            elif seq['camera_suspicious_percentage'] >= 2.0:
                seq['severity'] = 'MEDIUM'
            else:
                seq['severity'] = 'LOW'
        
        return top_sequences
    
    def count_critical_patterns_per_camera(self) -> int:
        """
        Count critical patterns (sequences of 2+ frames) per camera
        
        Returns:
            Total number of critical patterns across all cameras
        """
        total_critical_patterns = 0
        
        # Process each camera separately
        for cam_idx in range(self.num_cameras):
            cam_column = f'cam{cam_idx + 1}_freeze'
            
            # Extract freeze values for this camera
            freeze_values = [frame[cam_column] for frame in self.frame_data]
            
            # Find sequences of consecutive 1s
            sequences = self._find_consecutive_sequences(freeze_values, cam_idx + 1)
            
            # Count sequences with length >= 2
            critical_sequences = [seq for seq in sequences if seq['length'] >= 2]
            total_critical_patterns += len(critical_sequences)
        
        return total_critical_patterns
    
    def count_total_freezes(self, ratio_threshold: float = None) -> Dict[str, Any]:
        """
        Count total freeze frames across all cameras
        
        Args:
            ratio_threshold: Not used, kept for compatibility
            
        Returns:
            Dictionary with freeze statistics
        """
        # Count per camera
        camera_freezes = []
        for cam_idx in range(self.num_cameras):
            cam_freezes = sum(1 for frame in self.frame_data 
                            if frame[f'cam{cam_idx + 1}_freeze'] == 1)
            camera_freezes.append(cam_freezes)
        
        # Total freeze frames across all cameras
        total_freeze_frames = sum(camera_freezes)
        
        # Total possible camera frames (frames * cameras)
        total_camera_frames = self.min_frames * self.num_cameras
        
        # Calculate percentage relative to all camera frames
        freeze_percentage = (total_freeze_frames / total_camera_frames) * 100
        
        return {
            'total_freezes': total_freeze_frames,
            'camera_freezes': camera_freezes,
            'total_frames': total_camera_frames,
            'freeze_percentage': freeze_percentage
        }
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate detailed statistics about the analysis
        
        Returns:
            Dictionary with analysis statistics
        """
        # Convert camera statistics to expected format
        camera_stats = []
        for cam_stat in self.camera_statistics:
            camera_stats.append({
                'camera_index': cam_stat['camera'] - 1,
                'freeze_count': cam_stat['freeze_count'],
                'freeze_percentage': cam_stat['freeze_percentage'],
                'avg_edge_difference': cam_stat['avg_raw'],
                'avg_normalized': cam_stat['avg_normalized'],
                'total_frames': cam_stat['total_frames']
            })
        
        # Overall metrics - count freeze frames across all cameras
        total_freeze_frames = sum(cam_stat['freeze_count'] for cam_stat in self.camera_statistics)
        total_camera_frames = self.min_frames * self.num_cameras
        overall_freeze_rate = (total_freeze_frames / total_camera_frames) * 100
        
        # Find most problematic camera
        most_problematic = max(range(self.num_cameras), 
                             key=lambda i: self.camera_statistics[i]['freeze_count'])
        
        # Quality score based on freeze rate
        quality_score = max(0, 100 - (overall_freeze_rate * 2))  # Penalize freeze rate
        
        # Get top longest sequences per camera (length >= 2)
        top_sequences = self.get_top_longest_sequences(top_n=10, min_length=2)
        
        return {
            'detection_method': 'edge_normalized_analysis',
            'total_frames': self.min_frames,
            'freeze_threshold': self.freeze_threshold,
            'cameras': camera_stats,
            'setup_metrics': {
                'overall_freeze_rate_percent': overall_freeze_rate,
                'overall_quality_score': quality_score,
                'most_problematic_camera': most_problematic,
                'total_suspicious_frames': total_freeze_frames,
                'critical_sequences': self.count_critical_patterns_per_camera()
            },
            'top_longest_sequences': top_sequences
        }
    
    def save_to_csv(self, output_path: str, filename: str = 'freeze_analysis.csv') -> str:
        """Save frame data to CSV file"""
        csv_path = os.path.join(output_path, filename)
        
        print(f"Saving frame data to CSV: {csv_path}")
        
        fieldnames = [
            'frame_number', 'cam1_raw', 'cam2_raw', 'cam3_raw',
            'cam1_norm', 'cam2_norm', 'cam3_norm',
            'cam1_freeze', 'cam2_freeze', 'cam3_freeze',
            'frame_metric', 'frame_freeze'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.frame_data)
        
        print(f"CSV saved with {len(self.frame_data)} rows")
        return csv_path
    
    def save_to_xlsx(self, output_path: str, filename: str = 'freeze_analysis.xlsx') -> str:
        """Save frame data to Excel file"""
        xlsx_path = os.path.join(output_path, filename)
        
        print(f"Saving frame data to Excel: {xlsx_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.frame_data)
        
        # Save to Excel
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Frame_Analysis', index=False)
            
            # Add statistics sheet
            stats_data = []
            stats_data.append(['Metric', 'Value'])
            stats_data.append(['Total Frames', self.min_frames])
            stats_data.append(['Freeze Threshold', self.freeze_threshold])
            stats_data.append(['Suspicious Frames', sum(1 for f in self.frame_data if f['frame_freeze'] == 1)])
            stats_data.append(['Critical Sequences', len([s for s in self.suspicious_sequences if s['length'] >= 2])])
            
            for cam_stat in self.camera_statistics:
                stats_data.append([f'Camera {cam_stat["camera"]} Freezes', cam_stat['freeze_count']])
                stats_data.append([f'Camera {cam_stat["camera"]} Percentage', f"{cam_stat['freeze_percentage']:.2f}%"])
            
            stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Add longest sequences sheet (length >= 2)
            sequences = self.get_top_longest_sequences(top_n=20, min_length=2)
            if sequences:
                sequences_data = []
                for seq in sequences:
                                    sequences_data.append({
                    'Camera': seq['camera'],
                    'Start Frame': seq['start_frame'],
                    'End Frame': seq['end_frame'],
                    'Length': seq['length'],
                    'Sequence %': f"{seq['sequence_percentage']:.1f}%",
                    'Camera Total': seq['camera_total_suspicious'],
                    'Camera %': f"{seq['camera_suspicious_percentage']:.1f}%",
                    'Frame Range': seq['frame_range'],
                    'Avg Metric': f"{seq['avg_metric']:.4f}",
                    'Suspicion Score': f"{seq['suspicion_score']:.4f}",
                    'Severity': seq['severity']
                })
                
                sequences_df = pd.DataFrame(sequences_data)
                sequences_df.to_excel(writer, sheet_name='Longest_Sequences', index=False)
        
        print(f"Excel saved with {len(self.frame_data)} rows")
        return xlsx_path 