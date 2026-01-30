"""
SUPERVISED RANDOM FOREST LULC CLASSIFICATION MODULE
Parallel accuracy-enhancement path for the existing unsupervised pipeline.

Key Features:
- Trains on national-scale Sentinel-2 LULC dataset (India)
- Uses identical features as unsupervised pipeline
- Produces probabilistic confidence maps
- Zero modifications to existing pipeline
- No UI components

Author: GeoAI Engineer
Date: January 30, 2026
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Logging utility
def log_message(layer: str, message: str):
    """Structured logging for audit trail"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] RF_{layer}: {message}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Class harmonization mapping (7 Kaggle classes → 5 hackathon classes)
KAGGLE_TO_HACKATHON_MAPPING = {
    'Dense Forest': 'Forest',
    'Sparse Forest': 'Forest', 
    'Agricultural Land': 'Agriculture',
    'Fallow Land': 'Agriculture',
    'Water': 'Water Bodies',
    'Barren Land': 'Barren Land',
    'Built-up': 'Built-up'
}

# Final 5-class system (consistent with existing pipeline)
LULC_CLASSES = {
    0: 'Forest',
    1: 'Water Bodies', 
    2: 'Agriculture',
    3: 'Barren Land',
    4: 'Built-up'
}

# Reverse mapping for encoding
LULC_NAME_TO_ID = {name: id for id, name in LULC_CLASSES.items()}


# ============================================================================
# STEP 1: FEATURE STANDARDIZATION
# ============================================================================

class FeatureExtractor:
    """
    Extracts standardized features for both training and inference.
    Uses identical features as unsupervised pipeline.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_features(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract spectral bands + indices.
        
        Args:
            image: (bands, height, width) or (n_samples, bands) array
            normalize: Whether to apply z-score normalization
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if len(image.shape) == 3:
            # Reshape from (bands, height, width) to (n_samples, bands)
            n_bands, height, width = image.shape
            image = image.transpose(1, 2, 0).reshape(-1, n_bands)
        
        # Extract spectral bands (B2, B3, B4, B8)
        if image.shape[1] != 4:
            raise ValueError(f"Expected 4 bands (B2,B3,B4,B8), got {image.shape[1]}")
            
        blue = image[:, 0]
        green = image[:, 1] 
        red = image[:, 2]
        nir = image[:, 3]
        
        # Calculate spectral indices (identical to unsupervised pipeline)
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8) 
        ndbi = (nir - green) / (nir + green + 1e-8)
        
        # Combine spectral bands + indices
        features = np.column_stack([
            blue, green, red, nir,  # Original bands
            ndvi, ndwi, ndbi        # Derived indices
        ])
        
        # Z-score normalization
        if normalize:
            if not self.fitted:
                features = self.scaler.fit_transform(features)
                self.fitted = True
                log_message("FEATURES", "Feature scaling fitted on training data")
            else:
                features = self.scaler.transform(features)
        
        return features.astype(np.float32)
    
    def save_scaler(self, path: str = "rf_feature_scaler.pkl"):
        """Save fitted scaler"""
        if self.fitted:
            joblib.dump(self.scaler, path)
            log_message("FEATURES", f"Feature scaler saved to {path}")
        else:
            raise ValueError("Scaler not fitted yet")
    
    def load_scaler(self, path: str = "rf_feature_scaler.pkl"):
        """Load fitted scaler"""
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            self.fitted = True
            log_message("FEATURES", f"Feature scaler loaded from {path}")
        else:
            raise FileNotFoundError(f"Scaler file not found: {path}")


# ============================================================================
# STEP 2: CLASS HARMONIZATION
# ============================================================================

class ClassHarmonizer:
    """
    Handles mapping from 7 Kaggle classes to 5 hackathon classes.
    Ensures reproducible and logged harmonization.
    """
    
    def __init__(self):
        self.mapping = KAGGLE_TO_HACKATHON_MAPPING.copy()
        self.reverse_mapping = LULC_NAME_TO_ID.copy()
    
    def harmonize_labels(self, kaggle_labels: pd.Series) -> np.ndarray:
        """
        Convert Kaggle class names to hackathon class IDs.
        
        Args:
            kaggle_labels: Series with Kaggle class names
            
        Returns:
            Array of hackathon class IDs (0-4)
        """
        print("\n" + "="*70)
        print("STEP 2: Class Harmonization (7 → 5 classes)")
        print("="*70)
        
        # Map Kaggle names to hackathon names  
        hackathon_names = kaggle_labels.map(self.mapping)
        
        # Map hackathon names to IDs
        hackathon_ids = hackathon_names.map(self.reverse_mapping)
        
        # Log the mapping
        original_counts = kaggle_labels.value_counts().sort_index()
        final_counts = pd.Series(hackathon_ids).map(LULC_CLASSES).value_counts().sort_index()
        
        print("\nClass Mapping Summary:")
        print("Kaggle Class → Hackathon Class")
        print("-" * 40)
        for kaggle_class, hackathon_class in self.mapping.items():
            print(f"{kaggle_class:<20} → {hackathon_class}")
        
        print("\nFinal Class Distribution:")
        print("-" * 40) 
        for class_name, count in final_counts.items():
            pct = (count / len(hackathon_ids)) * 100
            print(f"{class_name:<15}: {count:>8,} ({pct:>5.1f}%)")
        
        # Check for any unmapped values
        unmapped = hackathon_ids.isnull().sum()
        if unmapped > 0:
            print(f"\n⚠️  WARNING: {unmapped} samples could not be mapped!")
            print("Unmapped classes:", kaggle_labels[hackathon_ids.isnull()].unique())
            raise ValueError("Class harmonization failed - check mapping")
        
        log_message("HARMONIZE", f"Successfully mapped {len(kaggle_labels)} samples to 5 classes")
        
        return hackathon_ids.values.astype(np.int32)
    
    def save_mapping(self, path: str = "class_harmonization_log.json"):
        """Save harmonization mapping with metadata"""
        mapping_log = {
            'timestamp': datetime.now().isoformat(),
            'strategy': 'Kaggle 7-class to hackathon 5-class mapping',
            'kaggle_to_hackathon': self.mapping,
            'hackathon_to_id': self.reverse_mapping,
            'final_classes': LULC_CLASSES
        }
        
        with open(path, 'w') as f:
            json.dump(mapping_log, f, indent=2)
        
        log_message("HARMONIZE", f"Class mapping logged to {path}")


# ============================================================================
# STEP 3: RANDOM FOREST TRAINING
# ============================================================================

class RandomForestTrainer:
    """
    Trains Random Forest on national-scale Sentinel-2 LULC dataset.
    Represents India-level spectral intelligence.
    """
    
    def __init__(self, 
                 n_estimators: int = 150,
                 max_depth: int = 25,
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1):
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_split=10,
            min_samples_leaf=5,
            bootstrap=True,
            oob_score=True
        )
        
        self.feature_extractor = FeatureExtractor()
        self.class_harmonizer = ClassHarmonizer()
        self.training_metrics = {}
        
    def train_from_kaggle_dataset(self, dataset_path: str) -> Dict:
        """
        Train RF model on Kaggle Sentinel-2 LULC dataset.
        
        Args:
            dataset_path: Path to Kaggle dataset directory
            
        Returns:
            Training metrics dictionary
        """
        print("\n" + "="*70)
        print("STEP 3: Random Forest Training on National Dataset")
        print("Dataset: Sentinel-2 LULC (India) - Kaggle")
        print("="*70)
        
        # Load dataset
        print("Loading Kaggle dataset...")
        X, y = self._load_kaggle_dataset(dataset_path)
        
        # Class harmonization (7 → 5 classes)
        y_harmonized = self.class_harmonizer.harmonize_labels(pd.Series(y))
        self.class_harmonizer.save_mapping()
        
        # Feature extraction and scaling
        print("\nExtracting and scaling features...")
        X_features = self.feature_extractor.extract_features(X, normalize=True)
        self.feature_extractor.save_scaler()
        
        print(f"✓ Training data prepared:")
        print(f"  Samples: {len(X_features):,}")
        print(f"  Features: {X_features.shape[1]} (bands + indices)")  
        print(f"  Classes: {len(np.unique(y_harmonized))}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y_harmonized,
            test_size=0.2,
            random_state=42,
            stratify=y_harmonized
        )
        
        print(f"\nTraining split:")
        print(f"  Training: {len(X_train):,} samples")
        print(f"  Validation: {len(X_val):,} samples")
        
        # Train Random Forest
        print(f"\nTraining Random Forest...")
        print(f"  n_estimators: {self.model.n_estimators}")
        print(f"  max_depth: {self.model.max_depth}")
        print(f"  class_weight: {self.model.class_weight}")
        
        self.model.fit(X_train, y_train)
        
        # Validation
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        oob_score = self.model.oob_score_
        
        print(f"\n✓ Training completed!")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        print(f"  OOB Score: {oob_score:.4f}")
        
        # Detailed metrics
        class_names = [LULC_CLASSES[i] for i in sorted(LULC_CLASSES.keys())]
        report = classification_report(y_val, y_pred, 
                                     target_names=class_names,
                                     output_dict=True)
        
        print(f"\nPer-class Performance:")
        for class_name in class_names:
            metrics = report[class_name]
            print(f"  {class_name:<15}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        feature_names = ['Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI', 'NDBI']
        
        print(f"\nFeature Importance:")
        for name, importance in zip(feature_names, importances):
            print(f"  {name:<8}: {importance:.4f}")
        
        # Store metrics
        self.training_metrics = {
            'validation_accuracy': float(accuracy),
            'oob_score': float(oob_score),
            'per_class_metrics': report,
            'feature_importance': dict(zip(feature_names, importances.tolist())),
            'training_samples': int(len(X_train)),
            'validation_samples': int(len(X_val)),
            'n_classes': int(len(class_names))
        }
        
        log_message("TRAIN", f"RF training complete - accuracy: {accuracy:.4f}")
        
        return self.training_metrics
    
    def _load_kaggle_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and parse Kaggle Sentinel-2 LULC dataset.
        Expected format: CSV with spectral bands and class labels.
        
        This is a placeholder - actual implementation depends on dataset format.
        """
        # TODO: Implement actual Kaggle dataset loading
        # For now, return dummy data for structure
        print("⚠️  Loading Kaggle dataset - placeholder implementation")
        print("   Replace with actual dataset loading logic")
        
        # Placeholder: Generate synthetic data matching expected structure
        n_samples = 50000
        X = np.random.randn(n_samples, 4).astype(np.float32)  # 4 bands (B2,B3,B4,B8)
        
        # Simulate realistic class distribution using correct Kaggle class names
        kaggle_classes = list(KAGGLE_TO_HACKATHON_MAPPING.keys())
        class_probs = [0.15, 0.15, 0.3, 0.1, 0.05, 0.15, 0.1]  # Probabilities for 7 classes
        y = np.random.choice(kaggle_classes, size=n_samples, p=class_probs)
        
        print(f"✓ Loaded dataset: {n_samples:,} samples")
        print(f"  Kaggle classes: {len(kaggle_classes)} classes")
        return X, y
    
    def save_model(self, path: str = "rf_lulc_india_model.pkl"):
        """Save trained Random Forest model"""
        model_package = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'class_harmonizer': self.class_harmonizer,
            'training_metrics': self.training_metrics,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'dataset': 'Sentinel-2 LULC (India) - Kaggle',
                'classes': LULC_CLASSES
            }
        }
        
        joblib.dump(model_package, path)
        print(f"✓ Model saved: {path}")
        log_message("TRAIN", f"RF model package saved to {path}")
        
    def save_training_report(self, path: str = "rf_training_report.json"):
        """Save comprehensive training report"""
        with open(path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        log_message("TRAIN", f"Training report saved to {path}")


# ============================================================================
# STEP 4: RF INFERENCE ON TIRUPATI
# ============================================================================

class RandomForestPredictor:
    """
    Applies trained RF model to Tirupati imagery for LULC classification.
    Generates both classification and confidence maps.
    """
    
    def __init__(self, model_path: str = "rf_lulc_india_model.pkl"):
        """Load trained RF model package"""
        if os.path.exists(model_path):
            package = joblib.load(model_path)
            self.model = package['model']
            self.feature_extractor = package['feature_extractor'] 
            self.class_harmonizer = package['class_harmonizer']
            self.metadata = package['metadata']
            log_message("PREDICT", f"RF model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def predict_tirupati(self, image_path: str, boundary_path: str, 
                        output_lulc_path: str, output_conf_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply RF model to Tirupati imagery.
        
        Args:
            image_path: Path to preprocessed GeoTIFF (4 bands)
            boundary_path: Path to Tirupati boundary shapefile
            output_lulc_path: Output path for LULC map
            output_conf_path: Output path for confidence map
            
        Returns:
            Tuple of (lulc_map, confidence_map)
        """
        print(f"\n{'='*70}")
        print(f"STEP 4: RF Inference on Tirupati")
        print(f"Input: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        # Load and clip image to boundary
        with rasterio.open(image_path) as src:
            image = src.read()
            image_meta = src.meta.copy()
            image_crs = src.crs
            image_transform = src.transform
        
        print(f"✓ Loaded image: {image.shape}")
        
        # Load and reproject boundary if needed
        boundary = gpd.read_file(boundary_path)
        if boundary.crs != image_crs:
            boundary = boundary.to_crs(image_crs)
            log_message("PREDICT", f"Boundary reprojected to {image_crs}")
        
        # Clip image to boundary
        geometries = boundary.geometry.values
        with rasterio.open(image_path) as src:
            clipped_image, clipped_transform = mask(src, geometries, crop=True)
        
        original_shape = clipped_image.shape
        print(f"✓ Clipped to boundary: {original_shape}")
        
        # Extract features (bands + indices)
        X_features = self.feature_extractor.extract_features(clipped_image, normalize=True)
        print(f"✓ Extracted features: {X_features.shape}")
        
        # RF Prediction
        print("Applying Random Forest model...")
        y_pred = self.model.predict(X_features)
        y_proba = self.model.predict_proba(X_features)
        
        # Calculate confidence as max class probability
        confidence = np.max(y_proba, axis=1).astype(np.float32)
        
        print(f"✓ RF prediction completed")
        print(f"  Mean confidence: {np.mean(confidence):.4f}")
        print(f"  Confidence range: [{np.min(confidence):.4f}, {np.max(confidence):.4f}]")
        
        # Reshape to spatial dimensions
        height, width = original_shape[1], original_shape[2]
        lulc_map = y_pred.reshape(height, width).astype(np.uint8)
        confidence_map = confidence.reshape(height, width).astype(np.float32)
        
        # Class distribution
        print("\nRF LULC Distribution:")
        unique_classes, counts = np.unique(y_pred, return_counts=True)
        for class_id, count in zip(unique_classes, counts):
            class_name = LULC_CLASSES.get(class_id, 'Unknown')
            pct = (count / len(y_pred)) * 100
            print(f"  {class_name:<15}: {count:>8,} ({pct:>5.1f}%)")
        
        # Save LULC map
        lulc_meta = image_meta.copy()
        lulc_meta.update({
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8',
            'crs': image_crs,
            'transform': clipped_transform
        })
        
        with rasterio.open(output_lulc_path, 'w', **lulc_meta) as dst:
            dst.write(lulc_map, 1)
        
        print(f"✓ Saved LULC map: {output_lulc_path}")
        
        # Save confidence map
        conf_meta = lulc_meta.copy()
        conf_meta['dtype'] = 'float32'
        
        with rasterio.open(output_conf_path, 'w', **conf_meta) as dst:
            dst.write(confidence_map, 1)
            
        print(f"✓ Saved confidence map: {output_conf_path}")
        
        log_message("PREDICT", f"RF inference complete for {os.path.basename(image_path)}")
        
        return lulc_map, confidence_map


# ============================================================================
# STEP 5: CHANGE DETECTION INTEGRATION
# ============================================================================

class RandomForestChangeDetector:
    """
    Performs change detection using RF-generated LULC maps.
    Reuses existing change detection logic for consistency.
    """
    
    def detect_changes(self, lulc_t1: np.ndarray, lulc_t2: np.ndarray,
                      metadata_t1: Dict, metadata_t2: Dict = None,
                      output_path: str = "rf_change_map.tif") -> Tuple[np.ndarray, float]:
        """
        Detect changes between RF-generated LULC maps.
        Uses same logic as existing unsupervised pipeline.
        """
        print(f"\n{'='*70}")
        print(f"STEP 5: RF Change Detection")
        print(f"Strategy: Pixel-wise comparison with geospatial alignment")
        print(f"{'='*70}")
        
        if metadata_t2 is None:
            metadata_t2 = metadata_t1
            
        # Ensure spatial alignment (same logic as existing pipeline)
        if lulc_t1.shape != lulc_t2.shape:
            print(f"Aligning T2 to T1 grid...")
            print(f"  T1: {lulc_t1.shape}, T2: {lulc_t2.shape}")
            
            # Use rasterio WarpedVRT for alignment (same as existing pipeline)
            h1, w1 = lulc_t1.shape
            
            # Create temporary T2 file
            temp_t2_path = "_temp_rf_t2.tif"
            with rasterio.open(
                temp_t2_path, 'w',
                driver='GTiff',
                height=lulc_t2.shape[0],
                width=lulc_t2.shape[1], 
                count=1,
                dtype=lulc_t2.dtype,
                crs=metadata_t2['crs'],
                transform=metadata_t2['transform']
            ) as dst:
                dst.write(lulc_t2, 1)
            
            # Reproject to T1 grid
            with rasterio.open(temp_t2_path) as src:
                with WarpedVRT(
                    src,
                    crs=metadata_t1['crs'],
                    resampling=rasterio.enums.Resampling.nearest,
                    transform=metadata_t1['transform'],
                    width=w1,
                    height=h1
                ) as vrt:
                    lulc_t2_aligned = vrt.read(1)
            
            os.remove(temp_t2_path)
            lulc_t2 = lulc_t2_aligned
            print(f"  Aligned T2: {lulc_t2.shape}")
        
        # Pixel-wise change detection
        changed = (lulc_t1 != lulc_t2).astype(np.uint8)
        num_changed = np.sum(changed)
        total_pixels = changed.size
        pct_changed = (num_changed / total_pixels) * 100
        
        print(f"\n✓ Change Detection Results:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Changed: {num_changed:,} ({pct_changed:.2f}%)")
        print(f"  Unchanged: {total_pixels - num_changed:,} ({100 - pct_changed:.2f}%)")
        
        # Save change map
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=changed.shape[0],
            width=changed.shape[1],
            count=1,
            dtype=changed.dtype,
            crs=metadata_t1['crs'],
            transform=metadata_t1['transform']
        ) as dst:
            dst.write(changed, 1)
            
        print(f"✓ Saved change map: {output_path}")
        log_message("CHANGE", f"RF change detection complete: {pct_changed:.2f}% changed")
        
        return changed, pct_changed
    
    def analyze_transitions(self, lulc_t1: np.ndarray, lulc_t2: np.ndarray,
                          output_path: str = "rf_transition_statistics.json") -> Dict:
        """
        Analyze land cover transitions using RF results.
        Same logic as existing pipeline for consistency.
        """
        print(f"\nAnalyzing RF-based transitions...")
        
        # Ensure same size
        if lulc_t1.size != lulc_t2.size:
            min_size = min(lulc_t1.size, lulc_t2.size)
            lulc_t1_flat = lulc_t1.flatten()[:min_size]
            lulc_t2_flat = lulc_t2.flatten()[:min_size]
        else:
            lulc_t1_flat = lulc_t1.flatten()
            lulc_t2_flat = lulc_t2.flatten()
        
        # Build transition matrix
        n_classes = len(LULC_CLASSES)
        transition_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(n_classes):
            for j in range(n_classes):
                mask = (lulc_t1_flat == i) & (lulc_t2_flat == j)
                transition_matrix[i, j] = np.sum(mask)
        
        # Find significant transitions (> 0.5%)
        total_pixels = len(lulc_t1_flat)
        significant_changes = []
        
        print(f"\nRF Transition Matrix:")
        print("-" * 70)
        
        class_names = [LULC_CLASSES[i] for i in sorted(LULC_CLASSES.keys())]
        print(f"{'From/To':<15}", end="")
        for name in class_names:
            print(f"{name[:12]:>14}", end="")
        print()
        print("-" * 70)
        
        for i, from_name in enumerate(class_names):
            print(f"{from_name:<15}", end="")
            for j in range(n_classes):
                count = transition_matrix[i, j]
                print(f"{count:>14,}", end="")
                
                # Check for significant changes
                if i != j:
                    pct = (count / total_pixels) * 100
                    if pct > 0.5:
                        significant_changes.append({
                            'from': LULC_CLASSES[i],
                            'to': LULC_CLASSES[j],
                            'pixels': int(count),
                            'percentage': float(pct)
                        })
            print()
        
        print(f"\nSignificant RF Transitions (> 0.5%):")
        for change in significant_changes:
            print(f"  {change['from']:<15} → {change['to']:<15}: "
                  f"{change['pixels']:>10,} ({change['percentage']:>5.2f}%)")
        
        # Save statistics
        rf_stats = {
            'method': 'Random Forest supervised classification',
            'transition_matrix': transition_matrix.tolist(),
            'significant_changes': significant_changes,
            'total_pixels': int(total_pixels),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(rf_stats, f, indent=2)
            
        print(f"✓ RF transition statistics saved: {output_path}")
        log_message("TRANSITION", f"RF transition analysis complete")
        
        return rf_stats


# ============================================================================
# VALIDATION & VERIFICATION
# ============================================================================

class RandomForestValidator:
    """
    Validates RF results against unsupervised results.
    Checks for improvements in stability and realism.
    """
    
    def validate_against_unsupervised(self, 
                                    rf_lulc_t1: np.ndarray, rf_lulc_t2: np.ndarray,
                                    unsup_lulc_t1: np.ndarray, unsup_lulc_t2: np.ndarray,
                                    rf_conf_t1: np.ndarray, rf_conf_t2: np.ndarray) -> Dict:
        """
        Compare RF results with unsupervised results.
        
        Returns validation report with key metrics.
        """
        print(f"\n{'='*70}")
        print(f"VALIDATION: RF vs Unsupervised Comparison")
        print(f"{'='*70}")
        
        validation_results = {}
        
        # 1. Class area distributions
        print(f"\n1. Class Area Distribution Comparison:")
        print(f"{'Class':<15} {'RF-T1 (%)':<10} {'Unsup-T1 (%)':<12} {'RF-T2 (%)':<10} {'Unsup-T2 (%)':<12}")
        print("-" * 65)
        
        class_comparison = {}
        for class_id, class_name in LULC_CLASSES.items():
            rf_t1_pct = (np.sum(rf_lulc_t1 == class_id) / rf_lulc_t1.size) * 100
            rf_t2_pct = (np.sum(rf_lulc_t2 == class_id) / rf_lulc_t2.size) * 100
            unsup_t1_pct = (np.sum(unsup_lulc_t1 == class_id) / unsup_lulc_t1.size) * 100
            unsup_t2_pct = (np.sum(unsup_lulc_t2 == class_id) / unsup_lulc_t2.size) * 100
            
            print(f"{class_name:<15} {rf_t1_pct:<10.2f} {unsup_t1_pct:<12.2f} {rf_t2_pct:<10.2f} {unsup_t2_pct:<12.2f}")
            
            class_comparison[class_name] = {
                'rf_t1_percent': float(rf_t1_pct),
                'rf_t2_percent': float(rf_t2_pct), 
                'unsup_t1_percent': float(unsup_t1_pct),
                'unsup_t2_percent': float(unsup_t2_pct)
            }
        
        # 2. Total change comparison
        rf_change_pct = (np.sum(rf_lulc_t1 != rf_lulc_t2) / rf_lulc_t1.size) * 100
        unsup_change_pct = (np.sum(unsup_lulc_t1 != unsup_lulc_t2) / unsup_lulc_t1.size) * 100
        
        print(f"\n2. Total Change Comparison:")
        print(f"  RF Change: {rf_change_pct:.2f}%")
        print(f"  Unsupervised Change: {unsup_change_pct:.2f}%")
        print(f"  Difference: {rf_change_pct - unsup_change_pct:+.2f} percentage points")
        
        # 3. Confidence analysis
        print(f"\n3. Confidence Analysis:")
        print(f"  RF Mean Confidence: {np.mean(rf_conf_t1):.4f}")
        print(f"  RF Confidence Range: [{np.min(rf_conf_t1):.4f}, {np.max(rf_conf_t1):.4f}]")
        print(f"  RF Std Dev: {np.std(rf_conf_t1):.4f}")
        
        # 4. Stability assessment
        rf_stable = rf_change_pct < unsup_change_pct
        reasonable_change = 5 <= rf_change_pct <= 20  # Reasonable change range
        
        print(f"\n4. Stability Assessment:")
        print(f"  RF more stable than unsupervised: {'✓' if rf_stable else '✗'}")
        print(f"  RF change in reasonable range (5-20%): {'✓' if reasonable_change else '✗'}")
        
        validation_results = {
            'class_distributions': class_comparison,
            'total_change': {
                'rf_percent': float(rf_change_pct),
                'unsupervised_percent': float(unsup_change_pct),
                'improvement': float(unsup_change_pct - rf_change_pct)
            },
            'confidence_stats': {
                'mean': float(np.mean(rf_conf_t1)),
                'std': float(np.std(rf_conf_t1)),
                'min': float(np.min(rf_conf_t1)),
                'max': float(np.max(rf_conf_t1))
            },
            'stability_assessment': {
                'rf_more_stable': bool(rf_stable),
                'change_in_reasonable_range': bool(reasonable_change)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save validation report
        with open('rf_validation_report.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
            
        print(f"\n✓ Validation report saved: rf_validation_report.json")
        log_message("VALIDATE", "RF vs unsupervised comparison complete")
        
        return validation_results


# ============================================================================
# MAIN RF INTEGRATION FUNCTION
# ============================================================================

def download_kaggle_dataset():
    """
    Download Kaggle Sentinel-2 LULC dataset.
    This is a placeholder - replace with actual download logic.
    """
    print("⚠️  Kaggle dataset download - placeholder implementation")
    print("   Please implement actual dataset download or provide dataset path")
    
    # For development, create a dummy dataset structure
    dataset_dir = "kaggle_sentinel2_lulc"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created placeholder dataset directory: {dataset_dir}")
    
    return dataset_dir