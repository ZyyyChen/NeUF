import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import json
import os
import cv2

class VolumeData:
    """
    A class to hold volume information and metadata that can be passed between scripts.
    """
    
    def __init__(self, 
                 point_min: Optional[np.ndarray] = None,
                 point_max: Optional[np.ndarray] = None,
                 spacing: Optional[np.ndarray] = None,
                 origin: Optional[np.ndarray] = None,
                 volume_shape: Optional[Tuple[int, int, int]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize VolumeData with volume information.
        
        Args:
            point_min: Minimum point of bounding box (3D coordinates)
            point_max: Maximum point of bounding box (3D coordinates)
            spacing: Voxel spacing in each dimension
            origin: Origin point of the volume
            volume_shape: Shape of the volume (depth, height, width)
            metadata: Additional metadata dictionary
        """
        self.point_min = point_min
        self.point_max = point_max
        self.spacing = spacing
        self.origin = origin
        self.volume_shape = volume_shape
        self.metadata = metadata or {}
        
        # Auto-compute spacing and origin if not provided but bounding box and shape are
        if self.spacing is None and self.point_min is not None and self.point_max is not None and self.volume_shape is not None:
            self.spacing = (self.point_max - self.point_min) / np.array(self.volume_shape)
            
        if self.origin is None and self.point_min is not None:
            self.origin = self.point_min.copy()
    
    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box as a tuple of (point_min, point_max)."""
        return self.point_min, self.point_max
    
    @property
    def volume_size(self) -> np.ndarray:
        """Get the physical size of the volume."""
        if self.point_min is not None and self.point_max is not None:
            return self.point_max - self.point_min
        return None
    
    @property
    def center(self) -> np.ndarray:
        """Get the center point of the volume."""
        if self.point_min is not None and self.point_max is not None:
            return (self.point_min + self.point_max) / 2
        return None
    
    def get_corners(self) -> np.ndarray:
        """Get all 8 corners of the bounding box."""
        if self.point_min is None or self.point_max is None:
            return None
            
        from itertools import product
        x_min, x_max = sorted([self.point_min[0], self.point_max[0]])
        y_min, y_max = sorted([self.point_min[1], self.point_max[1]])
        z_min, z_max = sorted([self.point_min[2], self.point_max[2]])
        
        corners = list(product([x_min, x_max], [y_min, y_max], [z_min, z_max]))
        return np.array(corners)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert VolumeData to a dictionary for serialization."""
        data = {
            'point_min': self.point_min.tolist() if self.point_min is not None else None,
            'point_max': self.point_max.tolist() if self.point_max is not None else None,
            'spacing': self.spacing.tolist() if self.spacing is not None else None,
            'origin': self.origin.tolist() if self.origin is not None else None,
            'volume_shape': self.volume_shape,
            'metadata': self.metadata
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VolumeData':
        """Create VolumeData from a dictionary."""
        return cls(
            point_min=np.array(data['point_min']) if data['point_min'] is not None else None,
            point_max=np.array(data['point_max']) if data['point_max'] is not None else None,
            spacing=np.array(data['spacing']) if data['spacing'] is not None else None,
            origin=np.array(data['origin']) if data['origin'] is not None else None,
            volume_shape=data['volume_shape'],
            metadata=data.get('metadata', {})
        )
    
    def save(self, filepath: str):
        """Save VolumeData to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'VolumeData':
        """Load VolumeData from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of VolumeData."""
        info = []
        if self.point_min is not None:
            info.append(f"Bounding Box: {self.point_min} to {self.point_max}")
        if self.spacing is not None:
            info.append(f"Spacing: {self.spacing}")
        if self.origin is not None:
            info.append(f"Origin: {self.origin}")
        if self.volume_shape is not None:
            info.append(f"Shape: {self.volume_shape}")
        else:
            info.append("Volume: Not loaded")
        
        return f"VolumeData({', '.join(info)})"
    
    def __repr__(self) -> str:
        return self.__str__()


def get_frames(frame_folder: str, prefix: str = "us", suffix: str = ".jpg", reversed: bool = False) -> List[np.ndarray]:
    """Get frames from a folder.
    Args:
        frame_folder: Folder containing the frames
        prefix: Prefix of the frames
        suffix: Suffix of the frames
    Returns:
        List of frames
    """
    frames = []

    file_list = [f for f in os.listdir(frame_folder) if f.endswith(suffix)] # get only images files
    file_list = [prefix + str(num) + suffix for num in range(len(file_list))]
    
    if reversed:
        file_list = file_list[::-1]
    
    for filename in file_list:
        image = cv2.imread(os.path.join(frame_folder, filename), cv2.IMREAD_GRAYSCALE)
        frames.append(image)

    return frames