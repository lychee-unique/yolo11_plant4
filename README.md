# YOLO11 Optimization for Lightweight and Accurate Plant Detection in UAV Aerial Imagery

[![DOI](https://zenodo.org/badge/907699807.svg)](https://doi.org/10.5281/zenodo.17163310)

## 1. Overview

This repository contains the code supporting the article **"You Look Only Once 11 (YOLO11) Optimization for Lightweight and Accurate Plant Detection in unmanned aerial vehicle (UAV) imagery"**, which is currently under peer review.

YOLO configuration files supporting the experiments in the paper can be found in the `cfgs` folder.

The `cfgs` folder contains two subfolders:

- **`Ablation Study`**: Includes configuration files for all models used in the **Ablation Study** section of the paper.
- **`SOTA Comparison`**: Configuration files for all models used in the paper’s **Comparison with State-of-the-Art Methods** and **Transfer Learning Study** sections.

The paper introduces three optimization strategies. The P2AR strategy can be located in the YAML model configuration files under the `cfgs` folder. The integration and use of the CBAM module and Shape-IoU Loss require modifications to the code in the `ultralytics/nn/modules` directory within the YOLO11 project. Detailed instructions can be found in the **User Guide** section.

## 2. Dataset

The four original single-class datasets referenced in the paper can be downloaded from the authors’ repositories:

- **CBDA (Cotton Boll Detection Augmented)** and **WEDU (Wheat Ears Detection Update)** — [Ye-Sk / Plant-dataset](https://github.com/Ye-Sk/Plant-dataset) [1]
- **MTDC-UAV (Maize Tassel Detection and Counting-UAV)** — [Ye-Sk/MTDC-UAV](https://github.com/Ye-Sk/MTDC-UAV) [2]
- **RFRB (Rape Flower Rectangular Box Labeling)** — [CV-Wang / RapeNet](https://github.com/CV-Wang/RapeNet) [3]

The YOLO-format annotations used in this paper for the Plant4 dataset can be found in this repository’s `data/` directory, along with the YAML configuration for training and evaluation.  
Use the following folder structure to recreate the Plant4 dataset as described in the paper:

```text
data/ 
├── labels/        # YOLO-format annotation files  
├── plant4.yaml    # dataset config file for training and evaluation of YOLO models; please set `path:` to your dataset root
└── images/        # please create the `images` folder and its two subfolders `train` and `test`
    ├──── train/   # place the TRAIN split files from the original CBDA, WEDU, MTDC-UAV, and RFRB datasets here  
    └──── test/    # place the VAL and TEST split files from the original CBDA, WEDU, MTDC-UAV, and RFRB datasets here
```

## 3. User Guide

### 3.1 Training and Evaluating YOLO Models

To train and evaluate a YOLO model using the configuration files provided in this repository, follow the example below. This script loads a custom model defined in `ours.yaml` and trains it on the `plant4.yaml` dataset.

```python
import sys
sys.path.insert(0, r'../ultralytics')  # Add YOLO11 repository path to system path
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"./ours.yaml")  # Load model architecture from specific config
    train_results = model.train(
        data=r"./plant4.yaml",      # Path to dataset YAML file
        epochs=300,                 # Number of training epochs
        imgsz=640,                  # Input image size
        batch=8,                    # Batch size
        project="plant4",           # Root directory for training outputs
        name="ours"                 # Subdirectory name for this experiment
    )
```

For additional options and full parameter documentation, refer to the [official Ultralytics YOLO documentation](https://docs.ultralytics.com/usage/cfg/).

### 3.2 Integrating the CBAM Module into YOLO11

To integrate the CBAM module into the YOLO11 framework, follow these steps:

1. In the `ultralytics/nn/tasks.py` file of the YOLO11 project, add the following import statement:
   
   ```python
   from ultralytics.nn.modules.conv import CBAM
   ```

2. modify the `parse_model` function in `tasks.py` by adding the following code under the `if m in ...` branch:
   
   ```python
   elif m is CBAM:
       c1 = ch[f]
       args = [c1, *args[1:]]
   ```
   
   This ensures proper handling and integration of the `CBAM` module within the YOLO11 model architecture.

3. Update the YOLO11 model's YAML configuration file to include the `CBAM` module. Example configurations can be found in the `cfgs` directory of this repository.

### 3.3 Using Shape-IoU Loss

In this study, we replaced the original CIoU loss function in YOLO11 with the Shape-IoU loss [4] function to enhance the accuracy of bounding box regression for small plant targets. To implement Shape-IoU in YOLO11, follow the instructions below.

- Import `shape_iou` (the `shape_iou_loss.py` file can be found in the `code` directory of this repository) into `ultralytics/utils/loss.py`.

- Modify the `forward` function in the `BboxLoss` class located in `ultralytics/utils/loss.py` to make the IoU loss selectable via a configuration flag (for example, a `bbox_loss` key in your config file). Inside `forward` method, add an if–else (or a small dispatch map) that switches among the supported IoU losses based on this flag. The snippet below is provided as a reference example.
  
  ```python
  if config.bbox_loss == 'Shape-IoU':
      iou = shape_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
      loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
  else:   # CIoU
      iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
      loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
  ```

## References

[1] Lu, Dunlu, et al. "Plant detection and counting: Enhancing precision agriculture in UAV and general scenes." *IEEE Access* (2023).

[2] Ye, Jianxiong, and Zhenghong Yu. "Fusing global and local information network for tassel detection in UAV imagery." *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 17 (2024): 4100-4108.

[3] Li, Jie, et al. "Automatic rape flower cluster counting method based on low-cost labelling and UAV-RGB images." *Plant Methods* 19.1 (2023): 40.

[4] Zhang, Hao, and Shuaijie Zhang. "Shape-iou: More accurate metric considering bounding box shape and scale." *arXiv preprint arXiv:2312.17663* (2023).