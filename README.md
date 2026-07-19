# Handwritten Character Tracing and Recognition Using Computer Vision

![Tracing Canvas Example](assets/tracing_preview.png)

---

## **Project Overview**

This project implements an **interactive handwritten character tracing and evaluation engine** using Computer Vision (CV) and Deep Learning. The system analyzes raw sketch canvas strokes, classifies the character intent, evaluates geometric alignment precision against standard templates, and returns target feedback.

The system utilizes a hybrid pipeline:
1. A custom **Convolutional Neural Network (CNN)** trained on the **EMNIST Balanced dataset** to identify the structural intent of the drawing.
2. A **Classical Geometry Engine** that uses pixel-wise matrix evaluation to calculate precise accuracy scores and isolate exact coordinates where drawing mistakes occurred.

The system can recognize and score 47 distinct balanced character classes, including:
- Digits (0–9)
- Uppercase Letters (A–Z)
- Distinct Lowercase Letters (a, b, d, e, f, g, h, n, q, r, t)

This project demonstrates an end-to-end machine learning and computer vision workflow, covering automated preprocessing, multi-model comparative training, metrics evaluation, and geometric error mapping.

---

## **Technologies and Libraries**

- **Python 3.10+**
- **TensorFlow / Keras** – Deep learning framework and inference engine
- **OpenCV (opencv-python)** – Advanced image transformation and classical matrix operations
- **NumPy** – High-performance matrix array computations
- **Pandas** – Dataframe analysis and CSV manipulation
- **Matplotlib & Seaborn** – Loss curves, accuracy charts, and confusion matrix rendering
- **Scikit-learn** – Core evaluation metrics (Macro Precision, Recall, F1-Score)

---

## **Dataset**

The project uses the **EMNIST (Extended MNIST) Balanced Dataset**, an industry benchmark for handwritten character categorization.

Dataset statistics:
- **131,600 total structural images**
- **47 perfectly balanced classes** (Digits + Alphabets combined)
- **2,800 uniform samples per class**
- **28 × 28 pixel dimensions** per image sample

Dataset Link:
https://www.kaggle.com/datasets/crawford/emnist

---

## **Image Preprocessing**

Raw stroke drawings captured from a user interface require normalization before feeding into the evaluation models to match historical training constraints.

The image processing unit executes these sequential steps:

1. **Grayscale Conversion:** Strips out auxiliary RGB profile channels to reduce footprint.
2. **Inversion:** Flips white canvas backgrounds to black, transforming stroke marks into high-intensity white pixels.
3. **Otsu's Binarization:** Applies automated thresholding to erase gray anti-aliasing brush shadows.
4. **Bounding Box Isolation:** Crops tightly along extreme stroke edge coordinates to isolate the target shape.
5. **Resizing:** Scales the extracted box area down to uniform dimensions while maintaining aspect ratios.

### **Inference Input Parameters**

| Parameter | Value |
|------------|-------|
| Target Matrix Dimensions | 28 × 28 Pixels |
| Color Channels | 1 (Grayscale) |
| Active Value Scale | Normalized [0.0 - 1.0] |
| Input Array Structure | 1 × 28 × 28 × 1 |

---

## **Dataset Split**

The dataset is divided cleanly to safeguard against data leakage and validate real-world reliability:

- **Training Subsample:** 95,880 structural images (85% of training file allocation)
- **Validation Subsample:** 16,920 structural images (15% internal evaluation split)
- **Testing Set:** 18,800 separate standalone images (Complete official EMNIST Test file)

---

## **Model Architecture Selection**

The research environment evaluates and compares three unique network architectures to select the optimal production model:

1. **Shallow CNN Baseline:** A lightweight 2-layer network used to establish baseline performance floors.
2. **Deep Regularized CNN:** An optimized, deeper network containing Batch Normalization and Dropout steps to maximize stability and prevent overfitting.
3. **Residual CNN:** A high-efficiency network using shortcut skip-connections to retain fine spatial edge features across pooling filters.

### **Why Deep Regularized CNN?**
- Batch Normalization stabilizes internal covariate shifts during training acceleration.
- Layer dropouts prevent dependency on single pixel regions, forcing the model to look at the entire character profile.
- High structural flexibility allows it to adapt to highly varied, erratic hand movements and shaky lines.

---

## **Training Details**

- **Loss Metric:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Base Learning Rate:** 1e-3
- **Batch Processing Size:** 128
- **Evaluation Lifecycles:** 20 Epochs

### **Inference Validation Features**
- Dynamic validation split metrics tracking
- Real-time performance evaluation logs
- Standardized binary model serialization checkpointing

---

## **Performance Metrics**

### **Model Architecture Comparison**

| Architecture Profile | Test Accuracy | Macro Precision | Macro F1-Score |
|----------------------|---------------|-----------------|----------------|
| Shallow Baseline     | 82.1%         | 0.814           | 0.817          |
| Deep Regularized CNN | 88.4%         | 0.881           | 0.882          |
| Residual CNN         | 86.9%         | 0.865           | 0.866          |

### **Evaluation Verification**
Comprehensive architectural sign-off requires parsing via:
- Class-by-class Precision, Recall, and F1 calculations.
- Macro-averaged metric overviews across all 47 target categories.
- Generated heatmaps via a full Confusion Matrix to pinpoint specific character confusion points (e.g., distinguishing '0' vs 'O' or '1' vs 'l').

---

## **Saved Model Engine**

The optimal, highest-performing architecture model instance is automatically bundled, serialized, and exported upon validation completion:

best_tracing_model.keras


This self-contained structure can be integrated directly into production API layers for real-time inference without requiring dependency adjustments.

---

## **Prediction & Scoring Pipeline**

Raw User Drawing (Canvas / PNG)
               │
               ▼
   [ Grayscale & Inversion ]
               │
               ▼
 [ Binarization & Thresholding ]
               │
               ▼
 [ Bounding Box Crop & Resize ] ─── Normalizes image to 28x28x1
               │
               ▼
    ┌──────────┴──────────┐
    ▼                     ▼
[ CNN Classifier ]   [ Geometry Engine ]
(Validates Intent)   (Calculates IoU Match)
│                     │
└──────────┬──────────┘
│
▼
[ Pixel-Wise Subtraction ] ───── Isolates missing stroke spots
│
▼
Final Results: Score %, Match Validated, Error Coordinates


---

## **Project Structure**

Handwritten-Character-Tracing/
│
├── assets/
│   └── tracing_preview.png
│
├── dataset/
│   ├── emnist-balanced-train.csv
│   └── emnist-balanced-test.csv
│
├── notebook/
│   └── character_training_comparison.ipynb
│
├── production_models/
│   └── best_tracing_model.keras
│
├── README.md
└── requirements.txt


---

## **Future Improvements**

- Add specialized text stroke data augmentations (elastic distortions, erosion, and dilation simulations).
- Implement Spatial Transformer Networks (STNs) to automatically align skewed drawings before classification.
- Integrate structural skeletonization (thinning algorithms) to evaluate line directions rather than raw pixel width.
- Incorporate lightweight edge-optimized model deployment setups (ONNX / TF-Lite conversions) to minimize API latency.

---

## **Applications**

This tracing evaluation technology can be applied to:
- Interactive early childhood alphabet and digit learning applications.
- Accessibility tools for individuals practicing fine motor skills.
- Gamified handwriting and calligraphy correction systems.
- Automated document signature consistency verification engines.

---

## **Results Summary**

This project delivers a complete deep learning and computer vision solution for stroke tracing evaluation. By pa