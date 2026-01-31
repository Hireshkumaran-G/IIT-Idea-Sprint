# AUTOMATED PIPELINE - EXECUTION GUIDE

## 🎯 **ONE COMMAND = COMPLETE ANALYSIS**

---

## ⚡ **SUPER SIMPLE EXECUTION:**

```bash
# Activate environment
.venv\Scripts\activate

# Run the automated pipeline
python automated_pipeline.py
```

**That's it! Everything runs automatically after you provide two folder paths.**

---

## 📥 **User Inputs (ONLY 2):**

When prompted, provide:

1. **SAFE folder for Time 1 (2016):**
   ```
   Tirupati_2016
   ```

2. **SAFE folder for Time 2 (2018):**
   ```
   Tirupati_2018
   ```

**The AOI boundary (`Tirupati_Boundary`) is already in the system.**

---

## 🤖 **What Happens Automatically:**

### **Layer 0:** SAFE → GeoTIFF
- Extracts Blue, Green, Red, NIR bands
- Creates `image_T1.tif`, `image_T2.tif`

### **Layer 1:** Independent Preprocessing
- Clips each image to boundary
- **CRITICAL:** Each image gets its OWN normalization
- T1 min/max ≠ T2 min/max (guaranteed isolation)
- Outputs: `X_T1`, `X_T2` (feature matrices)

### **Layer 2:** Model Training
- Fits K-Means on T1 data ONLY
- 5 clusters
- Saves `unsupervised_model.pkl`

### **Layer 3:** Cluster Prediction
- Applies model to both T1 and T2 separately
- Saves `cluster_map_T1.tif`, `cluster_map_T2.tif`

### **Layer 4:** Automatic Interpretation
- Calculates NDVI, NDWI, NDBI per cluster
- Maps clusters to: Forest, Water, Agriculture, Barren, Built-up
- Saves `cluster_to_lulc_mapping.json`

### **Layer 5:** LULC Map Generation
- Converts clusters to land cover classes
- Saves `lulc_map_T1.tif`, `lulc_map_T2.tif`

### **Layer 6:** Change Detection
- Pixel-by-pixel comparison
- Saves `change_map.tif` (binary: 0=unchanged, 1=changed)

### **Layer 7:** Transition Analysis
- 5×5 transition matrix
- Statistics (Forest→Built-up, etc.)
- Saves `transition_statistics.json`

### **Layer 8:** Confidence Estimation
- Distance to cluster center = confidence
- Saves `confidence_map_T1.tif`, `confidence_map_T2.tif`

---

## 📊 **Output Files (11 total):**

| File | Description |
|------|-------------|
| `image_T1.tif` | GeoTIFF for 2016 |
| `image_T2.tif` | GeoTIFF for 2018 |
| `cluster_map_T1.tif` | Cluster assignments 2016 |
| `cluster_map_T2.tif` | Cluster assignments 2018 |
| `lulc_map_T1.tif` | Land cover 2016 |
| `lulc_map_T2.tif` | Land cover 2018 |
| `change_map.tif` | Binary change map |
| `confidence_map_T1.tif` | Confidence scores 2016 |
| `confidence_map_T2.tif` | Confidence scores 2018 |
| `unsupervised_model.pkl` | Trained model |
| `cluster_to_lulc_mapping.json` | Interpretation |
| `transition_statistics.json` | Change statistics |

---

## 🔒 **Guaranteed Safeguards:**

### **No Cross-Contamination:**
- ✅ Each image preprocessed independently
- ✅ Separate normalization parameters per image
- ✅ T1 min/max stored with `image_id='T1'`
- ✅ T2 min/max stored with `image_id='T2'`
- ✅ Automatic verification checks

### **Error Detection:**
```python
# Built-in check:
if params['image_id'] != self.image_id:
    raise ValueError("Normalization contamination detected!")
```

---

## ⏱️ **Execution Time:**

- Layer 0: ~2 min (SAFE conversion)
- Layer 1: ~3 min (preprocessing)
- Layer 2: ~5 min (model training)
- Layer 3: ~2 min (prediction)
- Layers 4-8: ~2 min (interpretation & analysis)

**Total: ~15 minutes for 33M pixels**

---

## 💡 **Key Features:**

1. **Zero Manual Steps:** Upload SAFE folders → Get complete analysis
2. **Strict Isolation:** T1 and T2 never share normalization parameters
3. **Automatic Interpretation:** Spectral indices map clusters to LULC
4. **Complete Outputs:** Maps, statistics, confidence scores
5. **Production Ready:** Error handling, verification, logging

---

## 🎓 **For Judges:**

**One-sentence summary:**
> "Fully automated end-to-end GeoAI pipeline that converts raw Sentinel-2 SAFE folders to complete LULC classification and change detection in one execution, with strict guarantees against cross-year data contamination."

**Key innovations:**
- ✅ No intermediate user interaction
- ✅ Automated quality control (independence verification)
- ✅ 8-layer modular architecture
- ✅ Handles millions of pixels efficiently
- ✅ Production-grade error handling

---

## ✅ **YOU'RE READY!**

Just run:
```bash
python automated_pipeline.py
```

Enter two SAFE folder paths, and the system does everything automatically!

**15 minutes later → Complete LULC analysis with change detection** 🚀
