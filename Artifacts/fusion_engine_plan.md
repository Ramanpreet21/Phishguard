# Fusion Engine Implementation Plan (Visual + Tabular)

## Overview
Modern zero-text phishing pages (e.g., login forms rendered entirely as images) easily bypass NLP and tabular URL analysis. To combat this, we will introduce a **Visual + Tabular Fusion Engine**. This engine will combine our existing RF/XGBoost tabular models with a lightweight Convolutional Neural Network (CNN) analyzing screenshots of the rendered page.

---

## Step 1: Chrome Extension Modifications
We need to capture the rendered page's visual state to send to the backend.
1. **Background Service Worker**: 
   - Before calling the API, use `chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 50 })` to get a lightweight base64 representation of the page.
2. **Payload Update**: 
   - Modify the `PREDICT` payload to include the new `screenshot_b64` parameter.

---

## Step 2: API Schema Updates (`api.py`)
1. Update `PredictRequest` to accept an optional `screenshot_b64: Optional[str]`.
2. Update the `PredictResponse` to include the `visual_cnn` vote in `model_votes`.

---

## Step 3: CNN Integration (`src/models/dl_models.py`)
1. **MobileNetV3 Definition**:
   - Add a `PhishingVisualCNN` class inheriting from PyTorch's `nn.Module`.
   - Load a pre-trained `mobilenet_v3_small` architecture.
   - Replace the classification head with a binary output (`safe` vs `phishing`).
2. **Image Preprocessing**:
   - Add a utility to decode the base64 string, resize it (e.g., 224x224), apply standard ImageNet normalization, and convert it to a tensor.

---

## Step 4: The Fusion Layer (`predict.py`)
1. **Inference Pipeline**:
   - Update `PhishingPredictor` to route the screenshot through the CNN and retrieve a `visual_prob` score.
2. **Meta-Classifier (Fusion Engine)**:
   - Introduce a logistic regression layer (or deterministic heuristics if preferred before training).
   - The layer will take the output probabilities from the Tabular models (`RF`, `XGB`) and the Visual model (`CNN`).
   - **Override Rule**: If the URL looks benign (Tabular score < 0.3) but the visual model detects a 99% match for a known login screen (Visual score > 0.95), the Fusion layer will trigger a `phishing` block to mitigate zero-text evasion.

---

## Next Steps
This plan handles the pipeline plumbing. Retraining the CNN on a dataset of phishing screenshots and fitting the logistic regression meta-classifier will require a dataset update. For now, the code will implement the architectural framework.
