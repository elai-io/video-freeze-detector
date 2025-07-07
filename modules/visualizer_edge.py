import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Union, Dict, Any
from numpy.typing import NDArray
from tqdm import tqdm

from .video_analyzer import VideoAnalyzer


class VisualizerEdge:
    """
    Enhanced visualizer for edge-based freeze detection with edge difference images
    """
    
    def __init__(self, video_analyzer: VideoAnalyzer, output_path: str):
        """
        Initialize edge visualizer
        
        Args:
            video_analyzer: Video analyzer instance
            output_path: Path for saving images
        """
        self.video_analyzer = video_analyzer
        self.output_path = output_path
        self.save_path = output_path
        
        # Get frame dimensions
        self.frame_width, self.frame_height = video_analyzer.get_frame_size()
        
        # Center region size (1/3 of width)
        self.center_width = self.frame_width // 3
        self.center_start = self.frame_width // 3
        
        # Try to load font with larger sizes
        try:
            self.font_large = ImageFont.truetype("arial.ttf", 36)
            self.font_medium = ImageFont.truetype("arial.ttf", 28)
            self.font_small = ImageFont.truetype("arial.ttf", 20)
        except:
            # If system font not found, use default
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def extract_center_region(self, frame: NDArray) -> NDArray:
        """
        Extract center third of frame by width
        
        Args:
            frame: Source frame
            
        Returns:
            Center part of frame
        """
        return frame[:, self.center_start:self.center_start + self.center_width]
    
    def combine_frames_horizontally(self, frames: List[NDArray]) -> NDArray:
        """
        Combine frames horizontally
        
        Args:
            frames: List of frames to combine
            
        Returns:
            Combined frame
        """
        if not frames:
            return np.zeros((self.frame_height, self.center_width * 3, 3), dtype=np.uint8)
        
        # Extract center parts
        center_frames = [self.extract_center_region(frame) for frame in frames]
        
        # Combine horizontally
        combined = np.hstack(center_frames)
        
        return combined
    
    def apply_edge_detection(self, frame: NDArray) -> NDArray:
        """
        Apply edge detection to frame (same as in video_analyzer.py)
        
        Args:
            frame: Input frame
            
        Returns:
            Edge-detected frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to 3-channel for consistency
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_3ch
    
    def compute_edge_difference(self, frame1: NDArray, frame2: NDArray, amplify: float = 1.0) -> NDArray:
        """
        Compute edge difference between two frames
        
        Args:
            frame1: First frame
            frame2: Second frame
            amplify: Amplification factor for differences (not used for edges)
            
        Returns:
            Edge difference image as RGB
        """
        # Apply edge detection to both frames
        edges1 = self.apply_edge_detection(frame1)
        edges2 = self.apply_edge_detection(frame2)
        
        # Convert to float for computation
        e1 = edges1.astype(np.float32)
        e2 = edges2.astype(np.float32)
        
        # Compute absolute difference
        diff = np.abs(e1 - e2)
        
        # For edge differences, no amplification needed (already binary 0/255)
        # Just clip to ensure valid range
        diff_clipped = np.clip(diff, 0, 255)
        
        # Convert back to uint8
        return diff_clipped.astype(np.uint8)
    
    def create_edge_difference_images(self, prev_frames: List[NDArray], current_frames: List[NDArray]) -> NDArray:
        """
        Create edge difference images for all three cameras
        
        Args:
            prev_frames: Previous frames for all cameras
            current_frames: Current frames for all cameras
            
        Returns:
            Combined edge difference image
        """
        diff_frames = []
        
        for i in range(3):
            if i < len(prev_frames) and i < len(current_frames):
                # Extract center regions
                prev_center = self.extract_center_region(prev_frames[i])
                current_center = self.extract_center_region(current_frames[i])
                
                # Compute edge difference
                diff = self.compute_edge_difference(prev_center, current_center, amplify=1.0)
                diff_frames.append(diff)
            else:
                # Create empty frame if missing
                empty_frame = np.zeros((self.frame_height, self.center_width, 3), dtype=np.uint8)
                diff_frames.append(empty_frame)
        
        # Combine horizontally
        combined_diff = np.hstack(diff_frames)
        
        return combined_diff
    
    def add_edge_annotations(self, image: NDArray, candidate: Dict[str, Any], 
                            rank: int, is_current_frame: bool = True, 
                            is_edge_difference: bool = False) -> NDArray:
        """
        Add edge-based annotations over image
        
        Args:
            image: Image to annotate
            candidate: Freeze candidate with edge metrics
            rank: Rank in suspicious frames list
            is_current_frame: Current frame (True) or previous (False)
            is_edge_difference: Whether this is an edge difference image
            
        Returns:
            Annotated image
        """
        # Convert to PIL for easier text work
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Image dimensions
        img_width, img_height = pil_image.size
        section_width = img_width // 3
        
        # Colors for cameras - highlight the most suspicious camera
        colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]  # White by default
        most_suspicious_cam = candidate['most_suspicious_camera']
        colors[most_suspicious_cam] = (255, 0, 0)  # Red for most suspicious camera
        
        # Header
        if is_edge_difference:
            frame_type = "EDGE DIFFERENCE"
        else:
            frame_type = "CURRENT" if is_current_frame else "PREVIOUS"
        
        title = f"Rank #{rank} | Frame {candidate['frame_index']} | {frame_type}"
        draw.text((10, 10), title, fill=(255, 255, 0), font=self.font_large)
        
        # Edge velocity ratio metric (main detection metric)
        metric_text = f"Edge Velocity Min Ratio: {candidate['edge_velocity_min_ratio']:.6f}"
        draw.text((10, 50), metric_text, fill=(255, 255, 0), font=self.font_medium)
        
        # Information for each camera
        for i in range(3):
            x_offset = i * section_width + 10
            y_offset = img_height - 80
            
            # Camera number
            camera_text = f"Camera {i+1}"
            draw.text((x_offset, y_offset), camera_text, fill=colors[i], font=self.font_medium)
            
            # Edge velocity value (main metric)
            edge_vel_val = candidate['edge_velocities'][i]
            edge_vel_text = f"Edge Vel: {edge_vel_val:.4f}"
            draw.text((x_offset, y_offset + 30), edge_vel_text, fill=colors[i], font=self.font_small)
        
        # Separators between cameras
        for i in range(1, 3):
            x_line = i * section_width
            draw.line([(x_line, 0), (x_line, img_height)], fill=(0, 255, 0), width=3)
        
        # Convert back to OpenCV format
        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return annotated_image
    
    def create_freeze_images(self, freeze_candidates: List[Dict[str, Any]]) -> None:
        """
        Create images for all freeze candidates including edge difference images
        
        Args:
            freeze_candidates: List of freeze candidates from edge detector
        """
        print(f"Creating {len(freeze_candidates)} image sets (3 images each)...")
        
        candidate_iter = freeze_candidates
        if len(freeze_candidates) > 20:
            candidate_iter = tqdm(freeze_candidates, desc="Freeze images", unit="img")
        
        for rank, candidate in enumerate(candidate_iter, 1):
            try:
                # Get previous and current frames
                prev_frame_idx = max(0, candidate['frame_index'] - 1)
                current_frame_idx = candidate['frame_index']
                
                # Extract frames from all cameras
                prev_frames: List[NDArray] = []
                current_frames: List[NDArray] = []
                
                for camera_idx in range(3):
                    prev_frame = self.video_analyzer.get_frame_at_index(camera_idx, prev_frame_idx)
                    current_frame = self.video_analyzer.get_frame_at_index(camera_idx, current_frame_idx)
                    if prev_frame is not None and current_frame is not None:
                        prev_frames.append(prev_frame)
                        current_frames.append(current_frame)
                
                if len(prev_frames) == 3 and len(current_frames) == 3:
                    # Create regular frame images
                    prev_combined = self.combine_frames_horizontally(prev_frames)
                    current_combined = self.combine_frames_horizontally(current_frames)
                    
                    # Create edge difference image
                    edge_diff_combined = self.create_edge_difference_images(prev_frames, current_frames)
                    
                    # Add annotations
                    prev_annotated = self.add_edge_annotations(prev_combined, candidate, rank, False, False)
                    current_annotated = self.add_edge_annotations(current_combined, candidate, rank, True, False)
                    edge_diff_annotated = self.add_edge_annotations(edge_diff_combined, candidate, rank, True, True)
                    
                    # Generate filenames
                    base_name = f"{rank:03d}_{candidate['frame_index']:06d}_{candidate['edge_velocity_min_ratio']:.6f}_cam{candidate['most_suspicious_camera']}"
                    prev_filename = f"{base_name}_1.jpg"
                    current_filename = f"{base_name}_2.jpg"
                    edge_diff_filename = f"{base_name}_3.jpg"
                    
                    prev_path = os.path.join(self.save_path, prev_filename)
                    current_path = os.path.join(self.save_path, current_filename)
                    edge_diff_path = os.path.join(self.save_path, edge_diff_filename)
                    
                    cv2.imwrite(prev_path, prev_annotated)
                    cv2.imwrite(current_path, current_annotated)
                    cv2.imwrite(edge_diff_path, edge_diff_annotated)
                    
            except Exception as e:
                print(f"Error processing frame {candidate['frame_index']}: {str(e)}")
                continue
        
        print(f"Freeze images saved to: {self.save_path}")
        print(f"Each freeze has 3 files: _1 (previous), _2 (current), _3 (edge difference)")
    
    def create_summary_image(self, freeze_candidates: List[Dict[str, Any]], 
                           top_count: int = 5) -> str:
        """
        Create summary image with top candidates
        
        Args:
            freeze_candidates: List of freeze candidates
            top_count: Number of top candidates to include
            
        Returns:
            Path to summary image
        """
        if not freeze_candidates:
            return ""
        
        # Limit number of candidates
        top_candidates = freeze_candidates[:top_count]
        
        # Create summary image
        summary_height = len(top_candidates) * (self.frame_height + 200)
        summary_width = self.center_width * 3
        
        summary_image = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
        
        y_offset = 0
        
        for rank, candidate in enumerate(top_candidates, 1):
            try:
                # Get current frame
                current_frames = []
                for camera_idx in range(3):
                    frame = self.video_analyzer.get_frame_at_index(camera_idx, candidate['frame_index'])
                    if frame is not None:
                        current_frames.append(frame)
                
                if len(current_frames) == 3:
                    combined = self.combine_frames_horizontally(current_frames)
                    annotated = self.add_edge_annotations(combined, candidate, rank, True, False)
                    
                    # Insert into summary image
                    end_y = min(y_offset + self.frame_height, summary_height)
                    summary_image[y_offset:end_y, :] = annotated[:end_y - y_offset, :]
                    
                    y_offset += self.frame_height + 50
                
            except Exception as e:
                print(f"Error creating summary for frame {candidate['frame_index']}: {str(e)}")
                continue
        
        # Save summary image
        summary_path = os.path.join(self.output_path, f"summary_edge_top_{top_count}.jpg")
        cv2.imwrite(summary_path, summary_image)
        
        return summary_path 