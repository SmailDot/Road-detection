# Road Surface Detection: Gravel vs Asphalt

Semantic segmentation pipeline that detects road regions and classifies them as **gravel** or **asphalt** using a hybrid deep-learning + texture-analysis approach.

## Architecture

```
Input Image
    │
    ▼
SegFormer-B2 (ADE20K)          ← road/path/dirt-track pixel mask
    │
    ▼
Multi-Feature Texture Analysis
    ├── FFT High-Frequency Energy Ratio  (30%)
    ├── Sobel Edge Density               (25%)
    ├── LBP Variance                     (20%)
    ├── Colour Standard Deviation        (15%)
    └── Mean Intensity                   (10%)
    │
    ▼
Weighted Score → Gravel / Asphalt
    │
    ▼
Colour-Overlay Output Image
```

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.10. GPU recommended (CUDA 12+), falls back to CPU.

## Usage

```bash
# Place images in  images/  directory, then:
python road_detection.py
```

Output images are written to `output/`:

| File | Description |
|------|-------------|
| `{name}_result.jpg` | Original with colour overlay (OpenCV) |
| `{name}_panel.png`  | 3-panel: original / mask / result (matplotlib) |
| `summary.png`       | 2×3 grid of all results |

## Results

- **Green overlay** → Gravel road
- **Orange overlay** → Asphalt road

Feature scores printed per image; accuracy table shown at the end.

## Test Images

| File | Ground Truth |
|------|-------------|
| asphalt road_1.jpg | Asphalt |
| asphalt road_2.jpg | Asphalt |
| asphalt road_3.jpg | Asphalt |
| gravel road_1.jpg  | Gravel  |
| gravel road_2.jpeg | Gravel  |
| gravel road_3.jpg  | Gravel  |

## Model

Pre-trained **SegFormer-B2** fine-tuned on ADE20K 150-class dataset  
(`nvidia/segformer-b2-finetuned-ade-512-512` via HuggingFace Transformers).

Road-type classification is performed by a hand-crafted weighted scorer applied to texture features extracted from the segmented road region — no additional training data required.
