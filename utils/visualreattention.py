
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

from utils.selfrefinementengine import SelfRefinementEngine
from utils.selfrefinementengine import RefinementResult
from utils.uncertaintymetrics import UncertaintyMetrics
from utils.uncertaintymetrics import UncertaintyAnalyzer


class VisualReAttention:
    """
    Uncertainty-guided visual re-attention mechanism.
    
    Identifies under-explored image regions and generates
    targeted crops for verification.
    """
    
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
    
    def generate_attention_guided_crops(
        self,
        image: Image.Image,
        question: str,
        uncertainty_info: UncertaintyMetrics,
        max_crops: int = 3
    ) -> List[Dict]:
        """
        Generate crops of under-explored regions
        
        Args:
            image: Original image
            question: Question being asked
            uncertainty_info: Uncertainty metrics
            max_crops: Maximum number of crops to generate
        
        Returns:
            List of crop dictionaries with image, bbox, scale
        """
        
        # Extract attention heatmap
        attention_map = self._extract_attention_heatmap(image, question)
        
        if attention_map is None:
            # Fallback: grid-based crops
            return self._generate_grid_crops(image, n=max_crops)
        
        # Identify under-explored regions
        underexplored_regions = self._identify_underexplored_regions(
            attention_map,
            image.size
        )
        
        # Generate multi-scale crops
        crops = []
        for region in underexplored_regions[:max_crops]:
            crop_configs = self._generate_multi_scale_crops(
                image,
                region,
                scales=[1.2, 1.5, 2.0]
            )
            crops.extend(crop_configs)
        
        return crops[:max_crops * 2]  # Limit total crops
    
    def _extract_attention_heatmap(
        self,
        image: Image.Image,
        question: str
    ) -> Optional[np.ndarray]:
        """
        Extract cross-modal attention as 2D heatmap
        """
        try:
            inputs = self.processor(
                text=question,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            if not hasattr(outputs, 'cross_attentions') or not outputs.cross_attentions:
                return None
            
            # Get last layer cross-attention
            cross_attn = outputs.cross_attentions[-1]
            # Shape: [batch, heads, text_seq, img_patches]
            
            # Average over heads and text tokens
            attention_map = cross_attn.mean(dim=1).mean(dim=1).squeeze()
            # Shape: [img_patches]
            
            # Reshape to 2D
            num_patches = attention_map.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size != num_patches:
                return None
            
            attention_2d = attention_map.reshape(grid_size, grid_size)
            
            # Resize to image dimensions
            attention_2d = torch.nn.functional.interpolate(
                attention_2d.unsqueeze(0).unsqueeze(0),
                size=(image.height, image.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            return attention_2d.cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Attention extraction failed: {e}")
            return None
    
    def _identify_underexplored_regions(
        self,
        attention_map: np.ndarray,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Find regions with low attention scores
        """
        # Threshold: regions with attention < 20% of max
        threshold = 0.2 * attention_map.max()
        underexplored_mask = attention_map < threshold
        
        # Connected component analysis
        labeled, num_features = ndimage.label(underexplored_mask)
        
        regions = []
        for region_id in range(1, num_features + 1):
            region_mask = labeled == region_id
            
            # Extract bounding box
            rows = np.any(region_mask, axis=1)
            cols = np.any(region_mask, axis=0)
            
            if not rows.any() or not cols.any():
                continue
            
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            
            ymin, ymax = row_indices[0], row_indices[-1]
            xmin, xmax = col_indices[0], col_indices[-1]
            
            # Filter very small regions (< 5% of image)
            area = (ymax - ymin) * (xmax - xmin)
            total_area = attention_map.shape[0] * attention_map.shape[1]
            
            if area > 0.05 * total_area:
                avg_attention = attention_map[region_mask].mean()
                regions.append({
                    "bbox": (xmin, ymin, xmax, ymax),
                    "attention_score": float(avg_attention),
                    "area": int(area),
                    "id": region_id
                })
        
        # Sort by attention score (lowest first = most under-explored)
        regions.sort(key=lambda r: r['attention_score'])
        
        return regions
    
    def _generate_multi_scale_crops(
        self,
        image: Image.Image,
        region: Dict,
        scales: List[float]
    ) -> List[Dict]:
        """
        Generate crops at multiple scales centered on region
        """
        xmin, ymin, xmax, ymax = region['bbox']
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        
        crops = []
        for scale in scales:
            # Calculate crop size
            w = int((xmax - xmin) * scale)
            h = int((ymax - ymin) * scale)
            
            # Center on region, clip to image bounds
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(image.width, x1 + w)
            y2 = min(image.height, y1 + h)
            
            # Crop image
            cropped = image.crop((x1, y1, x2, y2))
            
            crops.append({
                "image": cropped,
                "bbox": (x1, y1, x2, y2),
                "scale": scale,
                "region_id": region.get("id")
            })
        
        return crops
    
    def _generate_grid_crops(
        self,
        image: Image.Image,
        n: int = 3
    ) -> List[Dict]:
        """
        Fallback: Generate grid-based crops
        """
        crops = []
        grid_size = int(np.ceil(np.sqrt(n)))
        
        cell_w = image.width // grid_size
        cell_h = image.height // grid_size
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = min(x1 + cell_w, image.width)
                y2 = min(y1 + cell_h, image.height)
                
                cropped = image.crop((x1, y1, x2, y2))
                
                crops.append({
                    "image": cropped,
                    "bbox": (x1, y1, x2, y2),
                    "scale": 1.0,
                    "region_id": count
                })
                
                count += 1
        
        return crops

