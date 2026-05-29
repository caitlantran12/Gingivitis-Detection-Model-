# Gingivitis-Detection-Model-
An interpretable computer vision pipeline utilising SAM2 automated masking and lightweight EfficientNet-B0 segmentation architectures to provide low cost, offline gingivitis screening for rural regional areas

---
## Data Engineering & System Architecture Pipeline
The application processes raw oral photographic data through a structured five stage automated pipeline:

```

┌──────────────────────┐      ┌──────────────────────┐      ┌───────────────────────────────┐
│   1. IMAGE INPUT     ├─────►│  2. SAM2 SEGMENTATION├─────►│ 3. CLASS MGI RESHAPE          │
│ (Low-res/Smartphone) │      │ (Pixel-level Masks)  │      │ (Healthy/Questionable/Disease)│
└──────────────────────┘      └──────────────────────┘      └───────────────────────────-───┘
│
┌──────────────────────┐      ┌──────────────────────┐                 │
│ 5. ACCESSIBILITY UX  │◄─────┤ 4. ENCODER-DECODER   │◄────────────────┘
│ (Colour Overlay Map) │      │ (EfficientNet-B0)    │
└──────────────────────┘      └──────────────────────┘

```

### 1. Data Preprocessing & Semantic Upgrading
* **Mask Generation via SAM2:** Transformed a public dataset of 1,096 intraoral records by upgrading loose YOLO bounding box coordinates into highly precise, pixel-level semantic masks utilising the **Segment Anything Model 2 (SAM2)**.
* **Class Consolidation:** To smooth out real world visual variances and combat dataset class imbalances, the original 5 tier Modified Gingival Index (MGI) was remapped into 3 distinct operational business layers: `Healthy`, `Questionable`, and `Diseased`.
* **Robust Augmentation:** Introduced random scaling, translation, brightness/contrast filters, and Gaussian noise layers to train the model to handle poor lighting and uneven framing typical of amateur smartphone captures.

### 2. Evaluated Framework Architectures
To identify the absolute best operational balance between compute footprint and regional boundary accuracy, the pipeline evaluated four distinct deep learning configurations. To preserve device memory constraints, all four pipelines were standardised with a lightweight **EfficientNet-B0 backbone encoder**:
* **PAN (Pyramid Attention Network):** High speed, low footprint framework optimising edge runtime efficiency.
* **UNet:** Standard medical framework utilising fundamental feature retaining skip connections.
* **UNet++:** Complex architecture introducing nested, dense skip pathways for highly granular feature collection.
* **DeepLabV3+:** High tier model utilising Atrous Spatial Pyramid Pooling (ASPP) to map macro contextual images.

---
## Analytical Trade-Off & Performance Metrics
System benchmarking revealed a distinct balance between raw spatial precision (Mean IoU) and computational speed:

| Model | Test Accuracy | Mean IoU| Mean Dice | Mean Recall | Runtime
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PAN** | *53%* | *20%* | *27%* | *35%* | *23 minutes* |
| **UNet** | **59%** | **27%** | **38%** | **39%** | **26 minutes** |
| **UNet++** | **65%** | **32%** | **43%** | **44%** | **51 minutes** |
| **DeepLabV3+** | **63%** | **29%** | **39%** | **41%** | **36 minutes** |

*Systems Analysis Note:* Boundary precision was slightly limited by the fuzzy edges generated via automated SAM2 masks versus manual dental labeling, alongside minor classification overlap between inflamed tissues and natural facial colour boundaries (e.g., lip tissue).

---
## Strategic User Experience Framework
The core delivery feature of this platform is its user focused interpretation layer. Instead of showing complex statistical probabilities or technical performance coefficients, the system uses a real-time pixel-level **Colour Overlay Filter** directly on top of the user's photo:

* 🟢 **Green Overlay:** Healthy tissue structure — No intervention required.
* 🟠 **Orange Overlay:** Questionable/Moderate severity — Monitor and schedule routine assessment.
* 🔴 **Dark Red Overlay:** High inflammation/Diseased tissue — Immediate recommendation for local specialist care.

This visual strategy eliminates language barriers and technical confusion, giving local workers an actionable screening framework.

---
## Technical Stack & Environment
* **Core Platform:** Python, PyTorch Deep Learning Ecosystem
* **Computer Vision Tooling:** Segment Anything Model 2 (SAM2), Segmentation Models Pytorch (SMP)
* **Backbone Backbone:** EfficientNet-B0 
* **Training Platform:** Cloud GPU Infrastructure (NVIDIA Tesla T4 Runtime Environment)
* **Performance Telemetry:** Pixel Accuracy, Intersection over Union (IoU), Dice Coefficient, Recall
