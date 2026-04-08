# 道路表面偵測：碎石路 vs 柏油路

利用語意分割（Semantic Segmentation）技術自動識別道路表面類型（gravel / asphalt），並以不同顏色標示天空、植被與道路區域。

---

## 系統架構

```
輸入圖片
    │
    ▼
SegFormer-B2（ADE20K 預訓練）     ← 語意分割：道路 / 天空 / 植被 / 其他
    │
    ▼
多特徵紋理分析（Texture Analysis）
    ├── lv_ratio（細/粗局部方差比）  ← 主要判斷依據（權重 35%）
    ├── Sobel 邊緣密度               （權重 25%）
    ├── Laplacian 方差               （權重 25%）
    └── 平均亮度                     （權重 15%）
    │
    ▼
加權分數 → Gravel（碎石路）或 Asphalt（柏油路）
    │
    ▼
多類別彩色疊加輸出圖片
```

---

## 顏色對照表

| 區域 | 顏色 |
|------|------|
| 天空（Sky） | 天藍色 |
| 植被 / 路邊（Vegetation） | 深綠色 |
| 柏油路（Asphalt） | 橘色 |
| 碎石路（Gravel） | 萊姆綠 |
| 其他（Other） | 灰色 |

---

## 安裝環境

```bash
pip install -r requirements.txt
```

> 需求：Python ≥ 3.10，建議使用 GPU（CUDA 12+），無 GPU 時自動切換 CPU 執行。

---

## 使用方式

1. 將待測圖片放入 `images/` 資料夾
2. 執行主程式：

```bash
python road_detection.py
```

結果圖片輸出至 `output/` 資料夾：

| 輸出檔案 | 說明 |
|----------|------|
| `{name}_result.jpg` | 彩色疊加圖（含圖例與信心值） |
| `{name}_panel.png`  | 三欄對照圖：原圖 / 語意遮罩 / 結果 |
| `summary.png`       | 6 張圖統整總結（2×3 格）       |

---

## 測試圖片

| 檔案 | 正確答案 |
|------|---------|
| asphalt road_1.jpg | 柏油路 |
| asphalt road_2.jpg | 柏油路 |
| asphalt road_3.jpg | 柏油路 |
| gravel road_1.jpg  | 碎石路 |
| gravel road_2.jpeg | 碎石路 |
| gravel road_3.jpg  | 碎石路 |

**測試準確率：6/6 = 100%**

---

## 輸出結果對照

### 整體總結（6 張）

![Summary](output/summary.png)

---

### 逐張前後對照（原圖 / 語意遮罩 / 結果）

#### 柏油路（Asphalt）

![asphalt_road_1](output/asphalt_road_1_panel.png)

![asphalt_road_2](output/asphalt_road_2_panel.png)

![asphalt_road_3](output/asphalt_road_3_panel.png)

#### 碎石路（Gravel）

![gravel_road_1](output/gravel_road_1_panel.png)

![gravel_road_2](output/gravel_road_2_panel.png)

![gravel_road_3](output/gravel_road_3_panel.png)

---

## 模型說明

- **分割模型**：`nvidia/segformer-b2-finetuned-ade-512-512`（HuggingFace Transformers）
  - 使用 ADE20K 150 類預訓練，識別 road / sidewalk / path / dirt track / sky / tree / grass 等類別
- **分類方法**：不需額外訓練資料，透過手工設計的加權紋理評分判斷路面類型

### 核心特徵：lv_ratio

$$\text{lv\_ratio} = \frac{\overline{\sigma_{5 \times 5}}}{\overline{\sigma_{15 \times 15}}}$$

- 計算路面區域在細尺度（5×5）與粗尺度（15×15）局部方差的比值
- 碎石路（gravel）的細尺度紋理相對豐富 → lv_ratio 較高（≥ 0.876）
- 柏油路（asphalt）表面平滑 → lv_ratio 較低（< 0.876）

---

## 技術細節

- 道路遮罩限制在圖片下方 65%，排除天空遠景誤判
- 形態學開閉運算（kernel 11×11）去除小雜訊區域
- 保留最大連通分量，確保只標記主要道路
- 支援 EXIF 旋轉校正，避免手機直拍圖片方向錯誤
