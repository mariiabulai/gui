"""
Image quality assessment for photogrammetry
"""
import numpy as np
import cv2 as cv
from typing import List, Tuple


class ImageQualityChecker:
    """Check if images are suitable for reconstruction"""
    
    def __init__(self):
        self.min_sharpness = 100  # Laplacian variance threshold
        self.min_features = 500
        self.max_blur_ratio = 0.3
    
    def check_sharpness(self, gray: np.ndarray) -> Tuple[float, bool]:
        """
        Check image sharpness using Laplacian variance
        Higher = sharper
        """
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        variance = laplacian.var()
        is_sharp = variance > self.min_sharpness
        return variance, is_sharp
    
    def check_features(self, gray: np.ndarray) -> Tuple[int, bool]:
        """Check if enough features can be detected"""
        sift = cv.SIFT_create(nfeatures=2000)
        kp, _ = sift.detectAndCompute(gray, None)
        count = len(kp)
        has_enough = count >= self.min_features
        return count, has_enough
    
    def check_exposure(self, gray: np.ndarray) -> Tuple[str, bool]:
        """Check for over/under exposure"""
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        dark_ratio = hist[:30].sum()
        bright_ratio = hist[225:].sum()
        
        if dark_ratio > 0.5:
            return "underexposed", False
        elif bright_ratio > 0.5:
            return "overexposed", False
        else:
            return "good", True
    
    def analyze_image(self, image: np.ndarray) -> dict:
        """Full quality analysis"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        sharpness, is_sharp = self.check_sharpness(gray)
        feature_count, has_features = self.check_features(gray)
        exposure, good_exposure = self.check_exposure(gray)
        
        quality_score = (
            (1.0 if is_sharp else 0.3) *
            (1.0 if has_features else 0.3) *
            (1.0 if good_exposure else 0.5)
        )
        
        return {
            'sharpness': sharpness,
            'is_sharp': is_sharp,
            'feature_count': feature_count,
            'has_features': has_features,
            'exposure': exposure,
            'good_exposure': good_exposure,
            'quality_score': quality_score,
            'usable': quality_score > 0.5
        }
    
    def filter_images(self, images: List[dict], verbose: bool = True) -> List[dict]:
        """Filter out low-quality images"""
        filtered = []
        
        for i, img_data in enumerate(images):
            quality = self.analyze_image(img_data['image'])
            
            if quality['usable']:
                filtered.append(img_data)
            elif verbose:
                print(f"  ⚠ Skipping image {i}: score={quality['quality_score']:.2f}, "
                      f"sharp={quality['is_sharp']}, features={quality['feature_count']}, "
                      f"exposure={quality['exposure']}")
        
        if verbose:
            print(f"  Quality filter: {len(filtered)}/{len(images)} images passed")
        
        return filtered