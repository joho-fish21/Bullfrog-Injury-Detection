# EC-BIMS / BID-YOLO: Bullfrog Injury Detection

Official implementation of the paper:

**EC-BIMS: An edge–cloud collaborative IoT system for real-time bullfrog injury monitoring via lightweight deep learning**

This repository contains the bullfrog injury dataset, the BID-YOLO detector (YOLOv8n + CBAM), training and evaluation scripts, and edge-deployment code for the NVIDIA Jetson Orin NX.

## Overview

- **Dataset**: 1128 annotated images (3816 instances) covering three injury categories: head injury, tail injury, and limb injury.
- **Model**: BID-YOLO = YOLOv8n + two CBAM modules (before the backbone SPPF layer and in the P3 branch of the neck), with only 3.02 M parameters.
- **Performance**: mAP@0.5 of 79.83 ± 3.19% over five independent runs (seeds 42, 123, 256, 512, 1024), +3.05 percentage points (p < 0.01) over the YOLOv8n baseline.
- **Deployment**: 78.7 FPS on Jetson Orin NX with TensorRT FP16; end-to-end alert latency of 0.606 s over a 4G link.

## Repository structure

```
├── custom_layers.py          # CBAM module definition
├── yolov8n_CBAM.yaml         # BID-YOLO model configuration
├── train.py                  # training script (five seeds)
├── data/
│   ├── seed_42/              # images/labels + data.yaml for each partition
│   └── ...                   # seed_123, seed_256, seed_512, seed_1024
└── weights/                  # download links for trained checkpoints (see Releases)
```

## Installation

```bash
# Python 3.8, CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Training

```bash
python 1-train.py --seeds 42 123 256 512 1024 --data-root data/
```

All models are trained with a unified protocol: input resolution 640 × 640, 200 epochs, batch size 32, SGD optimizer, cosine annealing, and no early stopping (patience = 0). Metrics are then averaged over the five runs (mean ± SD), as reported in the paper.

## Evaluation

```bash

```

## Main results

| Model               | mAP@0.5 (%)      | Recall (%)       | Params (M) | Size (MB) |
| ------------------- | ---------------- | ---------------- | ---------- | --------- |
| YOLOv8n (baseline)  | 76.78 ± 2.74     | 73.92 ± 3.05     | 3.01       | 5.97      |
| **BID-YOLO (ours)** | **79.83 ± 3.19** | **77.00 ± 3.95** | 3.02       | 5.99      |

Metrics are reported as mean ± SD over five independent runs.

## Edge deployment (Jetson Orin NX)

The trained checkpoint is exported to ONNX and compiled into a TensorRT FP16 engine, reaching 78.7 FPS (vs. 38.2 FPS in PyTorch FP32) with only 0.30 pp mAP degradation. TensorRT is provided by JetPack on the Jetson device; see the deployment scripts for the export and inference workflow.

## Trained weights

Pre-trained checkpoints for all five seeds are available in (通过网盘分享的文件：Bullfrog-Injury-Detection.zip
链接: https://pan.baidu.com/s/1_n8aXxkr3-BlPPKZJYVP2w 提取码: 8h25).

## Citation

If you find this dataset or code useful, please cite:

```
Wang, J., Shi, Y., Wang, L., Zhang, X., Wang, Y., 2026. EC-BIMS: An edge–cloud
collaborative IoT system for real-time bullfrog injury monitoring via lightweight
deep learning. (in press)
```

## License

- Code: 
- Dataset: 

## Acknowledgements

This work was supported by the Jiangsu Science and Technology Planning Project of China (Grant No. BE2021362).
