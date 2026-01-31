"""
Unsupervised LULC Classification System
Step 2 & 3: K-Means Clustering + Spectral Interpretation
GeoAI Engineer: No labeled data required
"""

import numpy as np
from sklearn.cluster import KMeans
import joblib
import rasterio
from typing import Dict, Tuple
import json

# Import preprocessing pipeline
from satellite_preprocessing_pipeline import preprocess_satellite_image


# LULC Class Definitions
LULC_CLASSES = {
    0: 'Forest',
    1: 'Water Bodies',
    2: 'Agriculture',
    3: 'Barren Land',
    4: 'Built-up'
}


def calculate_spectral_indices(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate spectral indices from normalized band values.
    Bands: [Blue, Green, Red, NIR]
    
    Args:
        X: Feature matrix (num_pixels, 4) with bands [Blue, Green, Red, NIR]
        
    Returns:
        Dictionary of spectral indices
    """
    print("\n" + "=" * 60)
    print("Calculating Spectral Indices")
    print("=" * 60)
    
    blue = X[:, 0]
    green = X[:, 1]
    red = X[:, 2]
    nir = X[:, 3]
    
    # NDVI: Normalized Difference Vegetation Index
    # High values = vegetation (Forest, Agriculture)
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    # NDWI: Normalized Difference Water Index
    # High values = water bodies
    ndwi = (green - nir) / (green + nir + 1e-8)
    
    # NDBI: Normalized Difference Built-up Index
    # High values = built-up areas
    ndbi = (nir - green) / (nir + green + 1e-8)
    
    # Additional: Brightness (overall reflectance)
    brightness = (blue + green + red + nir) / 4.0
    
    indices = {
        'ndvi': ndvi,
        'ndwi': ndwi,
        'ndbi': ndbi,
        'brightness': brightness
    }
    
    print(f"✓ NDVI range: [{np.min(ndvi):.3f}, {np.max(ndvi):.3f}]")
    print(f"✓ NDWI range: [{np.min(ndwi):.3f}, {np.max(ndwi):.3f}]")
    print(f"✓ NDBI range: [{np.min(ndbi):.3f}, {np.max(ndbi):.3f}]")
    
    return indices


def fit_kmeans_model(X: np.ndarray, 
                     n_clusters: int = 5,
                     random_state: int = 42) -> KMeans:
    """
    Fit K-Means clustering model (ONE-TIME on reference year data).
    
    Args:
        X: Normalized feature matrix from reference year (num_pixels, 4)
        n_clusters: Number of LULC clusters (default: 5)
        random_state: Random seed for reproducibility
        
    Returns:
        Fitted K-Means model
    """
    print("\n" + "=" * 60)
    print("STEP 2: Fitting Unsupervised K-Means Model")
    print("=" * 60)
    print(f"Training on reference year data: {X.shape[0]:,} pixels")
    print(f"Number of clusters: {n_clusters}")
    
    # Use MiniBatch K-Means for efficiency with large datasets
    from sklearn.cluster import MiniBatchKMeans
    
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=10000,
        max_iter=100,
        n_init=10,
        verbose=1
    )
    
    print("\nTraining K-Means model...")
    model.fit(X)
    
    print("✓ Model training complete!")
    print(f"✓ Inertia (within-cluster sum of squares): {model.inertia_:.2f}")
    
    # Display cluster centers
    print("\nCluster Centers (Band Reflectance):")
    print(f"{'Cluster':<10} {'Blue':<10} {'Green':<10} {'Red':<10} {'NIR':<10}")
    print("-" * 50)
    for i, center in enumerate(model.cluster_centers_):
        print(f"{i:<10} {center[0]:<10.4f} {center[1]:<10.4f} {center[2]:<10.4f} {center[3]:<10.4f}")
    
    return model


def predict_clusters(model: KMeans, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict cluster assignments for a given year's data.
    
    Args:
        model: Fitted K-Means model
        X: Feature matrix (num_pixels, 4)
        
    Returns:
        Tuple of (cluster_labels, distances_to_centers)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Predicting Cluster Assignments")
    print("=" * 60)
    
    # Predict cluster labels
    labels = model.predict(X)
    
    # Calculate distances to cluster centers (confidence metric)
    distances = model.transform(X)
    min_distances = np.min(distances, axis=1)
    
    # Display cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Calculate average confidence
    avg_distance = np.mean(min_distances)
    print(f"\nAverage distance to nearest cluster center: {avg_distance:.4f}")
    print("  (Lower distance = higher confidence)")
    
    return labels, min_distances


def interpret_clusters(model: KMeans, X_sample: np.ndarray) -> Dict[int, str]:
    """
    Automatically interpret clusters as LULC classes using spectral characteristics.
    
    Args:
        model: Fitted K-Means model
        X_sample: Sample feature matrix for calculating indices
        
    Returns:
        Dictionary mapping cluster_id → LULC class name
    """
    print("\n" + "=" * 60)
    print("Interpreting Clusters as LULC Classes")
    print("=" * 60)
    
    # Get cluster centers
    centers = model.cluster_centers_
    
    # Calculate indices for each cluster center
    cluster_characteristics = []
    
    for i, center in enumerate(centers):
        blue, green, red, nir = center
        
        # Calculate indices
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8)
        ndbi = (nir - green) / (nir + green + 1e-8)
        brightness = (blue + green + red + nir) / 4.0
        
        cluster_characteristics.append({
            'cluster_id': i,
            'ndvi': ndvi,
            'ndwi': ndwi,
            'ndbi': ndbi,
            'brightness': brightness,
            'nir': nir,
            'red': red
        })
    
    # Rule-based classification
    cluster_to_lulc = {}
    assigned_classes = set()
    
    # Sort by characteristics for assignment
    chars = sorted(cluster_characteristics, key=lambda x: x['ndwi'], reverse=True)
    
    for char in chars:
        cid = char['cluster_id']
        
        # Water Bodies: High NDWI, low brightness
        if 'Water Bodies' not in assigned_classes and char['ndwi'] > 0.0 and char['brightness'] < 0.3:
            cluster_to_lulc[cid] = 'Water Bodies'
            assigned_classes.add('Water Bodies')
            
        # Forest: High NDVI, high NIR
        elif 'Forest' not in assigned_classes and char['ndvi'] > 0.3 and char['nir'] > 0.3:
            cluster_to_lulc[cid] = 'Forest'
            assigned_classes.add('Forest')
            
        # Built-up: Low NDVI, moderate brightness, higher red
        elif 'Built-up' not in assigned_classes and char['ndvi'] < 0.2 and char['brightness'] > 0.3:
            cluster_to_lulc[cid] = 'Built-up'
            assigned_classes.add('Built-up')
            
        # Agriculture: Moderate NDVI (between forest and barren)
        elif 'Agriculture' not in assigned_classes and 0.2 <= char['ndvi'] <= 0.4:
            cluster_to_lulc[cid] = 'Agriculture'
            assigned_classes.add('Agriculture')
            
        # Barren Land: Low NDVI, variable brightness
        elif 'Barren Land' not in assigned_classes:
            cluster_to_lulc[cid] = 'Barren Land'
            assigned_classes.add('Barren Land')
    
    # Fill any remaining clusters
    remaining_classes = set(LULC_CLASSES.values()) - assigned_classes
    for cid in range(len(centers)):
        if cid not in cluster_to_lulc:
            if remaining_classes:
                cluster_to_lulc[cid] = remaining_classes.pop()
            else:
                cluster_to_lulc[cid] = 'Unknown'
    
    # Display mapping
    print("\nCluster → LULC Mapping:")
    print("-" * 60)
    print(f"{'Cluster':<10} {'LULC Class':<20} {'NDVI':<10} {'NDWI':<10} {'NDBI':<10}")
    print("-" * 60)
    for char in sorted(cluster_characteristics, key=lambda x: x['cluster_id']):
        cid = char['cluster_id']
        lulc = cluster_to_lulc[cid]
        print(f"{cid:<10} {lulc:<20} {char['ndvi']:<10.3f} {char['ndwi']:<10.3f} {char['ndbi']:<10.3f}")
    
    return cluster_to_lulc


def map_clusters_to_lulc(cluster_labels: np.ndarray, 
                        cluster_to_lulc: Dict[int, str]) -> np.ndarray:
    """
    Map cluster IDs to LULC class IDs.
    
    Args:
        cluster_labels: Array of cluster assignments
        cluster_to_lulc: Mapping from cluster_id to LULC class name
        
    Returns:
        Array of LULC class IDs
    """
    # Reverse LULC_CLASSES dict for lookup
    lulc_name_to_id = {name: id for id, name in LULC_CLASSES.items()}
    
    # Map clusters to LULC IDs
    lulc_labels = np.zeros_like(cluster_labels)
    for cluster_id, lulc_name in cluster_to_lulc.items():
        lulc_id = lulc_name_to_id.get(lulc_name, 0)
        lulc_labels[cluster_labels == cluster_id] = lulc_id
    
    return lulc_labels


def save_lulc_map(lulc_labels: np.ndarray,
                 metadata: Dict,
                 output_path: str):
    """
    Save LULC classification map as GeoTIFF.
    
    Args:
        lulc_labels: 1D array of LULC class IDs
        metadata: Spatial metadata from preprocessing
        output_path: Path to save the LULC map
    """
    print(f"\nSaving LULC map to: {output_path}")
    
    # Reshape to 2D spatial grid
    height = metadata['height']
    width = metadata['width']
    lulc_map = lulc_labels.reshape(height, width).astype(np.uint8)
    
    # Display LULC statistics
    print("\nLULC Distribution:")
    unique, counts = np.unique(lulc_labels, return_counts=True)
    for lulc_id, count in zip(unique, counts):
        lulc_name = LULC_CLASSES.get(lulc_id, 'Unknown')
        percentage = (count / len(lulc_labels)) * 100
        print(f"  {lulc_name:<15}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Save as GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=lulc_map.dtype,
        crs=metadata['crs'],
        transform=metadata['transform']
    ) as dst:
        dst.write(lulc_map, 1)
    
    print(f"✓ Saved: {output_path}")


def save_model_and_mapping(model: KMeans, 
                          cluster_to_lulc: Dict[int, str],
                          model_path: str = 'kmeans_lulc_model.pkl',
                          mapping_path: str = 'cluster_mapping.json'):
    """
    Save the fitted model and cluster mapping to disk.
    
    Args:
        model: Fitted K-Means model
        cluster_to_lulc: Cluster to LULC class mapping
        model_path: Path to save model
        mapping_path: Path to save mapping
    """
    print("\n" + "=" * 60)
    print("Saving Model and Mapping")
    print("=" * 60)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save mapping
    with open(mapping_path, 'w') as f:
        json.dump(cluster_to_lulc, f, indent=2)
    print(f"✓ Cluster mapping saved to: {mapping_path}")


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("UNSUPERVISED LULC CLASSIFICATION SYSTEM")
    print("K-Means Clustering + Spectral Interpretation")
    print("=" * 70)
    print("\nThis system requires NO labeled data.")
    print("\nWorkflow:")
    print("  1. Preprocess both images (2016, 2018)")
    print("  2. Fit K-Means model on 2016 data (reference year)")
    print("  3. Apply model to both years")
    print("  4. Interpret clusters as LULC classes")
    print("  5. Generate LULC maps for both years")
    print("\n" + "=" * 70)
    
    # Get inputs from user
    print("\nPlease provide the following files:\n")
    
    # Boundary shapefile (same for both years)
    print("1. Boundary Shapefile (AOI)")
    shapefile_path = input("   Enter path to boundary shapefile (.shp): ").strip()
    
    # Image for 2016 (reference year)
    print("\n2. Satellite Image - 2016 (Reference Year)")
    image_2016_path = input("   Enter path to 2016 satellite image (.tif): ").strip()
    
    # Image for 2018
    print("\n3. Satellite Image - 2018")
    image_2018_path = input("   Enter path to 2018 satellite image (.tif): ").strip()
    
    print("\n" + "=" * 70)
    
    try:
        # ===== PREPROCESSING (Uses existing Layer 1) =====
        
        print("\n" + "=" * 70)
        print("PREPROCESSING: Year 2016")
        print("=" * 70)
        X_2016, metadata_2016 = preprocess_satellite_image(
            image_path=image_2016_path,
            shapefile_path=shapefile_path
        )
        
        print("\n" + "=" * 70)
        print("PREPROCESSING: Year 2018")
        print("=" * 70)
        X_2018, metadata_2018 = preprocess_satellite_image(
            image_path=image_2018_path,
            shapefile_path=shapefile_path
        )
        
        # ===== STEP 2: FIT MODEL (ONE-TIME on 2016 data) =====
        
        model = fit_kmeans_model(X_2016, n_clusters=5)
        
        # ===== INTERPRET CLUSTERS =====
        
        cluster_to_lulc = interpret_clusters(model, X_2016)
        
        # ===== STEP 3: PREDICT FOR BOTH YEARS =====
        
        print("\n" + "=" * 70)
        print("Applying Model to 2016 Data")
        print("=" * 70)
        clusters_2016, distances_2016 = predict_clusters(model, X_2016)
        lulc_2016 = map_clusters_to_lulc(clusters_2016, cluster_to_lulc)
        
        print("\n" + "=" * 70)
        print("Applying Model to 2018 Data")
        print("=" * 70)
        clusters_2018, distances_2018 = predict_clusters(model, X_2018)
        lulc_2018 = map_clusters_to_lulc(clusters_2018, cluster_to_lulc)
        
        # ===== SAVE OUTPUTS =====
        
        print("\n" + "=" * 70)
        print("Saving LULC Maps")
        print("=" * 70)
        
        save_lulc_map(lulc_2016, metadata_2016, 'lulc_map_2016.tif')
        save_lulc_map(lulc_2018, metadata_2018, 'lulc_map_2018.tif')
        
        # Save model and mapping
        save_model_and_mapping(model, cluster_to_lulc)
        
        # ===== FINAL SUMMARY =====
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE!")
        print("=" * 70)
        print("\n✓ Generated Files:")
        print("  - lulc_map_2016.tif    (LULC classification for 2016)")
        print("  - lulc_map_2018.tif    (LULC classification for 2018)")
        print("  - kmeans_lulc_model.pkl (fitted model)")
        print("  - cluster_mapping.json  (cluster interpretation)")
        
        print("\n✓ LULC Classes:")
        for lulc_id, lulc_name in LULC_CLASSES.items():
            print(f"  {lulc_id}: {lulc_name}")
        
        print("\n✓ Next Step:")
        print("  Use these LULC maps for change detection analysis")
        print("  Run change detection script to identify transitions")
        
    except Exception as e:
        print(f"\n" + "=" * 70)
        print("ERROR OCCURRED")
        print("=" * 70)
        print(f"{e}")
        import traceback
        traceback.print_exc()
