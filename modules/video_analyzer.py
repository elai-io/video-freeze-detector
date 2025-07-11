import os
import cv2
import numpy as np
from typing import List, Tuple, Union, Dict, Any
from numpy.typing import NDArray
from tqdm import tqdm  # Progress bar


class VideoAnalyzer:
    """
    Class for loading and analyzing video files from synchronized cameras
    Supports formats: AVI, MP4
    """
    
    def __init__(self, input_path: str, verbose: bool = False):
        """
        Initialize video analyzer
        
        Args:
            input_path: Path to folder with video files
            verbose: Verbose output flag
        """
        self.input_path = input_path
        self.verbose = verbose
        self.video_files: List[str] = []
        self.frame_counts: List[int] = []
        self.fps_values: List[float] = []
        self.video_captures: List[cv2.VideoCapture] = []


    def load_videos(self) -> List[str]:
        """
        Load and validate video files
        
        Returns:
            List of video file paths
        """
        # Search for AVI and MP4 files
        video_files = []
        supported_extensions = ('.avi', '.mp4')
        
        for filename in os.listdir(self.input_path):
            if filename.lower().endswith(supported_extensions):
                video_files.append(os.path.join(self.input_path, filename))
        
        if len(video_files) != 3:
            raise ValueError(f"Found {len(video_files)} video files (AVI/MP4), expected exactly 3")
        
        self.video_files = sorted(video_files)  # Sort for consistency
        
        # Load metadata
        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.frame_counts.append(frame_count)
            self.fps_values.append(fps)
            self.video_captures.append(cap)
            
            if self.verbose:
                print(f"Loaded: {os.path.basename(video_path)}")
                print(f"  Frames: {frame_count}")
                print(f"  FPS: {fps:.2f}")
        
        return self.video_files


    def validate_synchronization(self) -> bool:
        """
        Validate video file synchronization
        
        Returns:
            True if all videos have the same frame count
        """
        if not self.frame_counts:
            return False
        
        return all(count == self.frame_counts[0] for count in self.frame_counts)


    def compute_laplacian_variance(self, frame: NDArray) -> float:
        """
        Compute Laplacian variance (measure of image sharpness/focus)
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Laplacian variance value (higher = sharper)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Return variance of Laplacian
        return float(np.var(laplacian))


    def compute_tenengrad_variance(self, frame: NDArray) -> float:
        """
        Compute Tenengrad variance (another measure of image sharpness)
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tenengrad variance value (higher = sharper)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel filters
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Return variance of gradient magnitude
        return float(np.var(gradient_magnitude))





    def compute_image_quality_metrics(self) -> Dict[str, List[float]]:
        """
        Compute average image quality metrics for all cameras
        
        Returns:
            Dictionary with quality metrics for each camera
        """
        print("Computing image quality metrics...")
        
        quality_metrics = {
            'laplacian_variance': [],
            'tenengrad_variance': []
        }
        
        min_frames = min(self.frame_counts)
        sample_frames = min(100, min_frames)  # Sample up to 100 frames for efficiency
        frame_indices = np.linspace(0, min_frames - 1, sample_frames, dtype=int)
        
        for camera_idx in range(len(self.video_captures)):
            if self.verbose:
                print(f"Analyzing image quality for camera {camera_idx + 1}...")
            
            cap = self.video_captures[camera_idx]
            
            laplacian_values = []
            tenengrad_values = []
            
            frame_iter = frame_indices
            if self.verbose:
                frame_iter = tqdm(frame_indices, desc=f"Camera {camera_idx + 1} quality", unit="frame", leave=False)
            
            for frame_idx in frame_iter:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Compute Laplacian variance
                laplacian_var = self.compute_laplacian_variance(frame)
                laplacian_values.append(laplacian_var)
                
                # Compute Tenengrad variance
                tenengrad_var = self.compute_tenengrad_variance(frame)
                tenengrad_values.append(tenengrad_var)
            
            # Store average values
            quality_metrics['laplacian_variance'].append(float(np.mean(laplacian_values)) if laplacian_values else 0.0)
            quality_metrics['tenengrad_variance'].append(float(np.mean(tenengrad_values)) if tenengrad_values else 0.0)
            
            if self.verbose:
                print(f"  Camera {camera_idx + 1} quality metrics:")
                print(f"    Laplacian variance: {quality_metrics['laplacian_variance'][-1]:.2f}")
                print(f"    Tenengrad variance: {quality_metrics['tenengrad_variance'][-1]:.2f}")
        
        return quality_metrics


    def compute_frame_differences(self) -> List[List[float]]:
        """
        Compute mean of differences between consecutive frames for all videos
        
        Returns:
            List of difference lists for each video (using mean of pixel differences across all BGR channels)
        """
        all_differences: List[List[float]] = []
        min_frames: int = min(self.frame_counts)
        
        for i, cap in enumerate(self.video_captures):
            if self.verbose:
                print(f"Analyzing camera {i+1} (total {min_frames} frames)...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            differences: List[float] = []
            prev_frame: Union[NDArray, None] = None
            
            frame_iter = range(min_frames)
            if self.verbose:
                frame_iter = tqdm(frame_iter, desc=f"Camera {i+1}", unit="frame", leave=False)
            
            for _ in frame_iter:
                ret, frame = cap.read()
                if not ret:
                    break
                if prev_frame is not None:
                    # Compute mean of differences across all BGR channels
                    # Mean is more intuitive and matches the extract_center_frames approach
                    diff_array = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))
                    diff = np.mean(diff_array)
                    differences.append(diff)
                prev_frame = frame
            
            all_differences.append(differences)
            if self.verbose:
                print(f"  Average difference (BGR): {np.mean(differences):.2f}")
        return all_differences


    def apply_edge_detection(self, frame: NDArray) -> NDArray:
        """
        Apply edge detection to frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Edge-detected frame (grayscale)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges


    def compute_edge_frame_differences(self) -> List[List[float]]:
        """
        Compute mean of differences between consecutive frames using edge detection
        
        Returns:
            List of difference lists for each video (using edge-detected frames)
        """
        all_differences: List[List[float]] = []
        min_frames: int = min(self.frame_counts)
        
        for i, cap in enumerate(self.video_captures):
            if self.verbose:
                print(f"Analyzing camera {i+1} with edge detection (total {min_frames} frames)...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            differences: List[float] = []
            prev_edges: Union[NDArray, None] = None
            
            frame_iter = range(min_frames)
            if self.verbose:
                frame_iter = tqdm(frame_iter, desc=f"Camera {i+1} (edges)", unit="frame", leave=False)
            
            for _ in frame_iter:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply edge detection
                edges = self.apply_edge_detection(frame)
                
                if prev_edges is not None:
                    # Compute mean of differences between edge images
                    diff_array = np.abs(edges.astype(np.float32) - prev_edges.astype(np.float32))
                    diff = np.mean(diff_array)
                    differences.append(diff)
                prev_edges = edges
            
            all_differences.append(differences)
            if self.verbose:
                print(f"  Average edge difference: {np.mean(differences):.2f}")
        return all_differences


    def get_frame_at_index(self, camera_index: int, frame_index: int) -> Union[NDArray, None]:
        """
        Get frame at specific index from specified camera
        
        Args:
            camera_index: Camera index (0, 1, 2)
            frame_index: Frame index
            
        Returns:
            Frame as numpy array or None if failed
        """
        if camera_index >= len(self.video_captures):
            return None
        
        cap = self.video_captures[camera_index]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        return frame if ret else None


    def get_frame_size(self, camera_index: int = 0) -> Tuple[int, int]:
        """
        Get frame size
        
        Args:
            camera_index: Camera index for getting size
            
        Returns:
            Tuple (width, height)
        """
        if camera_index >= len(self.video_captures):
            return (0, 0)
        
        cap = self.video_captures[camera_index]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)


    def close(self):
        """
        Close all video captures
        """
        for cap in self.video_captures:
            cap.release()
        self.video_captures.clear() 